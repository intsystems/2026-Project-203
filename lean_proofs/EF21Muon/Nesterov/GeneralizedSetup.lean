import EF21Muon.Nesterov.Setup
import EF21Muon.Nesterov.RadiusEnergy
import EF21Muon.Distributed.GeneralizedDrift

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]

abbrev RadiusStates (N : ℕ) (m n : Fin p → ℕ) := States N m n × Model m n

def radiusPoints (c : Fin p → Parameters) (s : States N m n) : Model m n :=
  Distributed.radiusPoints c (core s)

def radiusNext {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (z : RadiusStates N m n)
    (ω : Ω) : RadiusStates N m n :=
  (fun i => radiusUpdate (c i) (A.CU i) (z.1 i)
      (fun j => P.reply (radiusPoints c z.1) ω j i),
    fun i => (1-(c i).β) • z.2 i+(c i).β •
      (average (fun j => P.reply (radiusPoints c z.1) ω j i)-P.grad (radiusPoints c z.1) i))

def radiusStart (x : Model m n) : RadiusStates N m n := (initial x,0)

def radiusFrozen {μ : Measure Ω} [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (c : Fin p → Parameters)
    (s : States N m n) (i : Fin p) := P.radiusFrozen c (core s) i

theorem points_radiusNext {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (z : RadiusStates N m n) (ω : Ω) :
    points (radiusNext P A z ω).1 = radiusPoints c z.1 := rfl

theorem radiusNext_frozen {μ : Measure Ω} [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters} (A : Scheme N m n c)
    (z : RadiusStates N m n) (ω : Ω) (i : Fin p) :
    (radiusNext P A z ω).1 i = radiusUpdate (c i) (A.CU i) (z.1 i)
      ((radiusFrozen P c z.1 i).replies ω) ∧
    (radiusNext P A z ω).2 i = newMomentum (c i).β (z.2 i) 0
      (radiusFrozen P c z.1 i).mean ω := by
  have he : (radiusFrozen P c z.1 i).replies ω =
      (fun j => P.reply (radiusPoints c z.1) ω j i) := by
    funext j
    dsimp [Noise.replies,radiusFrozen,GeneralizedProblem.radiusFrozen,radiusPoints]
    module
  constructor
  · rw [he]; rfl
  · dsimp [radiusNext,newMomentum,Noise.mean,radiusFrozen,GeneralizedProblem.radiusFrozen,radiusPoints]
    rw [average_sub,P.mean_gradient]
    simp only [zero_add]

theorem measurable_radiusPoints (c : Fin p → Parameters) :
    Measurable (radiusPoints c : States N m n → Model m n) :=
  (Distributed.measurable_radiusPoints c).comp measurable_core

theorem measurable_radiusNext {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) :
    Measurable (Function.uncurry (radiusNext P A)) := by
  have hx : Measurable (fun z : RadiusStates N m n × Ω => radiusPoints c z.1.1) :=
    (measurable_radiusPoints c).comp measurable_fst.fst
  have hreply : Measurable (fun z : RadiusStates N m n × Ω => P.reply (radiusPoints c z.1.1) z.2) :=
    P.measurable_reply.comp (hx.prodMk measurable_snd)
  have hold (i : Fin p) (j : Fin N) : Measurable (fun z : RadiusStates N m n × Ω => (z.1.1 i).1.M j) :=
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst.fst).fst.snd.snd.fst
  have hrep (i : Fin p) (j : Fin N) : Measurable (fun z : RadiusStates N m n × Ω =>
      P.reply (radiusPoints c z.1.1) z.2 j i) :=
    (measurable_pi_apply i).comp ((measurable_pi_apply j).comp hreply)
  have hM (i : Fin p) (j : Fin N) : Measurable (fun z : RadiusStates N m n × Ω =>
      (1-(c i).β) • (z.1.1 i).1.M j+(c i).β • P.reply (radiusPoints c z.1.1) z.2 j i) := by
    simpa only [Pi.smul_def,Pi.add_def] using
      ((hold i j).const_smul (1-(c i).β)).add ((hrep i j).const_smul (c i).β)
  have hV (i : Fin p) (j : Fin N) : Measurable (fun z : RadiusStates N m n × Ω =>
      (1-(c i).β) • ((1-(c i).β) • (z.1.1 i).1.M j+(c i).β • P.reply (radiusPoints c z.1.1) z.2 j i)+
        (c i).β • P.reply (radiusPoints c z.1.1) z.2 j i) := by
    simpa only [Pi.smul_def,Pi.add_def] using
      ((hM i j).const_smul (1-(c i).β)).add ((hrep i j).const_smul (c i).β)
  apply Measurable.prodMk
  · apply Measurable.of_eval
    intro i
    have hi := (measurable_pi_apply i).comp hx
    have hG j : Measurable (fun z : RadiusStates N m n × Ω => (z.1.1 i).1.G j) :=
      (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst.fst).fst.snd.snd.snd
    exact (hi.prodMk (hi.prodMk ((Measurable.of_eval (hM i)).prodMk
      (Measurable.of_eval fun j => (hG j).add ((A.measurable_CU i).comp ((hV i j).sub (hG j))))))).prodMk
        (Measurable.of_eval (hold i))
  · apply Measurable.of_eval
    intro i
    have hZ : Measurable (fun z : RadiusStates N m n × Ω => z.1.2 i) :=
      (measurable_pi_apply i).comp measurable_fst.snd
    have hmean := measurable_average (fun j => hrep i j)
    have hg := (measurable_pi_apply i).comp (P.gradient_continuous.measurable.comp hx)
    simpa only [Function.comp_def,Pi.smul_def,Pi.add_def,Pi.sub_def] using
      (hZ.const_smul (1-(c i).β)).add ((hmean.sub hg).const_smul (c i).β)

def generalizedPotential {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) (z : RadiusStates N m n) : ℝ :=
  P.f (points z.1)-P.floor+∑ i, radiusEnergy (c i) (z.1 i)
    (P.grad (points z.1) i) (z.2 i) (fun j => P.localGrad j (points z.1) i)

def generalizedCost {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) (z : RadiusStates N m n) : ℝ :=
  ∑ i, (c i).γ*nuclearNorm (P.grad (points z.1) i)

theorem generalizedPotential_nonneg {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) (z : RadiusStates N m n) : 0 ≤ generalizedPotential P c z :=
  add_nonneg (sub_nonneg.mpr (P.lower_bound _))
    (Finset.sum_nonneg fun _ _ => radiusEnergy_nonneg _ _ _ _ _)

theorem generalizedCost_nonneg {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) (z : RadiusStates N m n) : 0 ≤ generalizedCost P c z :=
  Finset.sum_nonneg fun i _ => mul_nonneg (c i).γ_pos.le (nuclearNorm_nonneg _)

theorem generalizedCost_measurable {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) : Measurable (generalizedCost P c) :=
  Finset.measurable_sum _ (fun i _ =>
    (continuous_nuclearNorm.measurable.comp ((measurable_pi_apply i).comp
      (P.gradient_continuous.measurable.comp (measurable_points.comp measurable_fst)))).const_mul _)

theorem generalizedPotential_measurable {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) : Measurable (generalizedPotential P c) := by
  have hx : Measurable (fun z : RadiusStates N m n => points z.1) := measurable_points.comp measurable_fst
  have hf : Measurable (fun z : RadiusStates N m n => P.f (points z.1)-P.floor) :=
    (continuous_iff_continuousAt.mpr (fun x => (P.derivative x).continuousAt)).measurable.comp hx
      |>.sub measurable_const
  apply hf.add
  apply Finset.measurable_sum
  intro i _
  have hM j : Measurable (fun z : RadiusStates N m n => (z.1 i).1.M j) :=
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst).fst.snd.snd.fst
  have hG j : Measurable (fun z : RadiusStates N m n => (z.1 i).1.G j) :=
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst).fst.snd.snd.snd
  have hprev j : Measurable (fun z : RadiusStates N m n => (z.1 i).2 j) :=
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst).snd
  have hV j : Measurable (fun z : RadiusStates N m n => target (c i) (z.1 i) j) := by
    simpa only [target,Pi.smul_def,Pi.sub_def] using
      ((hM j).const_smul (2-(c i).β)).sub ((hprev j).const_smul (1-(c i).β))
  have hg := (measurable_pi_apply i).comp (P.gradient_continuous.measurable.comp hx)
  have hZ : Measurable (fun z : RadiusStates N m n => z.2 i) := (measurable_pi_apply i).comp measurable_snd
  have hD := (((measurable_average hM).sub hg).sub hZ).norm.const_mul (radiusDWeight (c i))
  have hF := (measurable_average fun j => ((hM j).sub
    ((measurable_pi_apply i).comp ((P.local_gradient_continuous j).measurable.comp hx))).norm).const_mul
      (radiusRWeight (c i))
  have hH := (measurable_average fun j => ((hV j).sub (hG j)).norm).const_mul (radiusHWeight (c i))
  have hJ := (measurable_average fun j => ((hM j).sub (hprev j)).norm).const_mul (radiusJWeight (c i))
  exact (((hD.add hF).add hH).add hJ).add ((hZ.norm.pow_const 2).const_mul _)

theorem generalizedPotential_integrable {μ : Measure Ω} [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters} (A : Scheme N m n c)
    (z : RadiusStates N m n) : Integrable (fun ω => generalizedPotential P c (radiusNext P A z ω)) μ := by
  have he (ω : Ω) : generalizedPotential P c (radiusNext P A z ω) =
      P.f (radiusPoints c z.1)-P.floor+∑ i,
        radiusEnergy (c i) (radiusUpdate (c i) (A.CU i) (z.1 i)
          ((radiusFrozen P c z.1 i).replies ω)) (P.grad (radiusPoints c z.1) i)
          (newMomentum (c i).β (z.2 i) 0 (radiusFrozen P c z.1 i).mean ω)
          (fun j => P.localGrad j (radiusPoints c z.1) i) := by
    simp only [generalizedPotential,points_radiusNext,
      (radiusNext_frozen P A z ω _).1,(radiusNext_frozen P A z ω _).2]
  simp_rw [he]
  exact (integrable_const _).add (integrable_finsetSum _ fun i _ =>
    radiusEnergy_integrable (c i) (A.measurable_CU i) (A.contracts_CU i) (z.1 i)
      (z.2 i) (P.grad (radiusPoints c z.1) i) (radiusFrozen P c z.1 i))

end EF21Muon.Nesterov
