import EF21Muon.Distributed.RadiusEnergy

/-! The generalized-smooth problem and the exact-downlink constant-radius
algorithm. The problem data are independent of the momentum and step controls.
The second state component is a ghost noise EMA used only in the proof. -/

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]

structure GeneralizedProblem (N : ℕ) (m n : Fin p → ℕ) (μ : Measure Ω) where
  workers_pos : 0 < N
  f : Model m n → ℝ
  grad : Model m n → Model m n
  localF : Fin N → Model m n → ℝ
  localGrad : Fin N → Model m n → Model m n
  localMin : Fin N → ℝ
  local_lower : ∀ j x, localMin j ≤ localF j x
  mean_objective : ∀ x, average (fun j => localF j x) = f x
  mean_gradient : ∀ x i, average (fun j => localGrad j x i) = grad x i
  derivative : ∀ x, HasFDerivAt f (gradientMap (grad x)) x
  local_derivative : ∀ j x, HasFDerivAt (localF j) (gradientMap (localGrad j x)) x
  L₀ : Fin p → ℝ
  L₁ : Fin p → ℝ
  localL₀ : Fin N → Fin p → ℝ
  localL₁ : Fin N → Fin p → ℝ
  L₀_nonneg : ∀ i, 0 ≤ L₀ i
  L₁_nonneg : ∀ i, 0 ≤ L₁ i
  localL₀_nonneg : ∀ j i, 0 ≤ localL₀ j i
  localL₁_nonneg : ∀ j i, 0 ≤ localL₁ j i
  smooth : GeneralizedSmooth grad L₀ L₁
  localSmooth : ∀ j, GeneralizedSmooth (localGrad j) (localL₀ j) (localL₁ j)
  σ : Fin p → ℝ
  σ_nonneg : ∀ i, 0 ≤ σ i
  reply : Model m n → Ω → Fin N → Model m n
  measurable_reply : Measurable (Function.uncurry reply)
  noise_memLp : ∀ x i j, MemLp (fun ω => reply x ω j i-localGrad j x i) 2 μ
  noise_centered : ∀ x i j, (∫ ω, reply x ω j i-localGrad j x i ∂μ) = 0
  noise_variance : ∀ x i j, (∫ ω, ‖reply x ω j i-localGrad j x i‖^2 ∂μ) ≤ σ i^2
  worker_independence : ∀ x i, ProbabilityTheory.iIndepFun
    (fun j ω => reply x ω j i-localGrad j x i) μ

abbrev RadiusStates (N : ℕ) (m n : Fin p → ℕ) := States N m n × Model m n

def radiusPoints (c : Fin p → Parameters) (s : States N m n) : Model m n :=
  fun i => radiusPoint (c i) (s i)

def radiusNext {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (z : RadiusStates N m n)
    (ω : Ω) : RadiusStates N m n :=
  (fun i => radiusUpdate (c i) (A.CU i) (z.1 i)
      (fun j => P.reply (radiusPoints c z.1) ω j i),
    fun i => (1-(c i).β) • z.2 i + (c i).β •
      (average (fun j => P.reply (radiusPoints c z.1) ω j i)-P.grad (radiusPoints c z.1) i))

def GeneralizedProblem.floor {μ : Measure Ω} (P : GeneralizedProblem N m n μ) : ℝ :=
  average P.localMin

theorem GeneralizedProblem.lower_bound {μ : Measure Ω}
    (P : GeneralizedProblem N m n μ) (x : Model m n) : P.floor ≤ P.f x := by
  rw [← P.mean_objective x]
  exact average_mono (fun j => P.local_lower j x)

theorem GeneralizedProblem.gradient_continuous {μ : Measure Ω}
    (P : GeneralizedProblem N m n μ) : Continuous P.grad :=
  generalized_gradient_continuous P.smooth P.L₀_nonneg P.L₁_nonneg

theorem GeneralizedProblem.local_gradient_continuous {μ : Measure Ω}
    (P : GeneralizedProblem N m n μ) (j : Fin N) : Continuous (P.localGrad j) :=
  generalized_gradient_continuous (P.localSmooth j) (P.localL₀_nonneg j) (P.localL₁_nonneg j)

def GeneralizedProblem.radiusFrozen {μ : Measure Ω} [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (c : Fin p → Parameters)
    (s : States N m n) (i : Fin p) :
    Noise N (m i) (n i) μ (fun j => P.localGrad j (radiusPoints c s) i) (P.σ i) where
  value j ω := P.reply (radiusPoints c s) ω j i-P.localGrad j (radiusPoints c s) i
  memLp := P.noise_memLp (radiusPoints c s) i
  centered := P.noise_centered (radiusPoints c s) i
  variance := P.noise_variance (radiusPoints c s) i
  independent := P.worker_independence (radiusPoints c s) i

theorem points_radiusNext {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (z : RadiusStates N m n) (ω : Ω) :
    points (radiusNext P A z ω).1 = radiusPoints c z.1 := rfl

theorem radiusNext_frozen {μ : Measure Ω} [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters} (A : Scheme N m n c)
    (z : RadiusStates N m n) (ω : Ω) (i : Fin p) :
    (radiusNext P A z ω).1 i = radiusUpdate (c i) (A.CU i) (z.1 i)
      ((P.radiusFrozen c z.1 i).replies ω) ∧
    (radiusNext P A z ω).2 i = newMomentum (c i).β (z.2 i) 0
      (P.radiusFrozen c z.1 i).mean ω := by
  have he : (P.radiusFrozen c z.1 i).replies ω =
      (fun j => P.reply (radiusPoints c z.1) ω j i) := by
    funext j
    dsimp [Noise.replies, GeneralizedProblem.radiusFrozen]
    module
  constructor
  · rw [he]; rfl
  · dsimp [radiusNext, newMomentum, Noise.mean, GeneralizedProblem.radiusFrozen]
    rw [average_sub, P.mean_gradient]
    simp only [zero_add]

theorem measurable_radiusPoints (c : Fin p → Parameters) :
    Measurable (radiusPoints c : States N m n → Model m n) := by
  apply Measurable.of_eval
  intro i
  have hx : Measurable (fun s : States N m n => (s i).x) := (measurable_pi_apply i).fst
  have hG : Measurable (fun s : States N m n => average (s i).G) :=
    measurable_average (fun j => (measurable_pi_apply j).comp (measurable_pi_apply i).snd.snd.snd)
  simpa only [radiusPoints, radiusPoint, Function.comp_def, Pi.smul_def, Pi.sub_def] using
    hx.sub (((measurable_polar (m i) (n i)).comp hG).const_smul (c i).γ)

theorem measurable_radiusNext {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) :
    Measurable (Function.uncurry (radiusNext P A)) := by
  have hx : Measurable (fun z : RadiusStates N m n × Ω => radiusPoints c z.1.1) :=
    (measurable_radiusPoints c).comp measurable_fst.fst
  have hreply : Measurable (fun z : RadiusStates N m n × Ω =>
      P.reply (radiusPoints c z.1.1) z.2) :=
    P.measurable_reply.comp (hx.prodMk measurable_snd)
  have hM (i : Fin p) (j : Fin N) : Measurable (fun z : RadiusStates N m n × Ω =>
      (1-(c i).β) • (z.1.1 i).M j + (c i).β • P.reply (radiusPoints c z.1.1) z.2 j i) := by
    have hold : Measurable (fun z : RadiusStates N m n × Ω => (z.1.1 i).M j) :=
      (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst.fst).snd.snd.fst
    have hrep := (measurable_pi_apply i).comp ((measurable_pi_apply j).comp hreply)
    simpa only [Function.comp_def, Pi.smul_def, Pi.add_def] using
      (hold.const_smul (1-(c i).β)).add (hrep.const_smul (c i).β)
  apply Measurable.prodMk
  · apply Measurable.of_eval
    intro i
    have hi := (measurable_pi_apply i).comp hx
    have hG (j : Fin N) : Measurable (fun z : RadiusStates N m n × Ω => (z.1.1 i).G j) :=
      (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst.fst).snd.snd.snd
    exact hi.prodMk (hi.prodMk ((Measurable.of_eval (hM i)).prodMk
      (Measurable.of_eval fun j => (hG j).add ((A.measurable_CU i).comp ((hM i j).sub (hG j))))))
  · apply Measurable.of_eval
    intro i
    have hZ : Measurable (fun z : RadiusStates N m n × Ω => z.1.2 i) :=
      (measurable_pi_apply i).comp measurable_fst.snd
    have hrep := measurable_average (fun j =>
      (measurable_pi_apply i).comp ((measurable_pi_apply j).comp hreply))
    have hg := (measurable_pi_apply i).comp (P.gradient_continuous.measurable.comp hx)
    simpa only [Function.comp_def, Pi.smul_def, Pi.add_def, Pi.sub_def] using
      (hZ.const_smul (1-(c i).β)).add ((hrep.sub hg).const_smul (c i).β)

def GeneralizedProblem.radiusPotential {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) (z : RadiusStates N m n) : ℝ :=
  P.f (points z.1)-P.floor + ∑ i, radiusEnergy (c i) (z.1 i)
    (P.grad (points z.1) i) (z.2 i) (fun j => P.localGrad j (points z.1) i)

def GeneralizedProblem.radiusCost {μ : Measure Ω} (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) (z : RadiusStates N m n) : ℝ :=
  ∑ i, (c i).γ*nuclearNorm (P.grad (points z.1) i)

theorem GeneralizedProblem.radiusPotential_nonneg {μ : Measure Ω}
    (P : GeneralizedProblem N m n μ) (c : Fin p → Parameters) (z : RadiusStates N m n) :
    0 ≤ P.radiusPotential c z :=
  add_nonneg (sub_nonneg.mpr (P.lower_bound _))
    (Finset.sum_nonneg fun _ _ => radiusEnergy_nonneg _ _ _ _ _)

theorem GeneralizedProblem.radiusCost_nonneg {μ : Measure Ω}
    (P : GeneralizedProblem N m n μ) (c : Fin p → Parameters) (z : RadiusStates N m n) :
    0 ≤ P.radiusCost c z :=
  Finset.sum_nonneg fun i _ => mul_nonneg (c i).γ_pos.le (nuclearNorm_nonneg _)

theorem GeneralizedProblem.measurable_radiusCost {μ : Measure Ω}
    (P : GeneralizedProblem N m n μ) (c : Fin p → Parameters) :
    Measurable (P.radiusCost c) := by
  apply Finset.measurable_sum
  intro i _
  exact (continuous_nuclearNorm.measurable.comp ((measurable_pi_apply i).comp
    (P.gradient_continuous.measurable.comp (measurable_points.comp measurable_fst)))).const_mul _

theorem GeneralizedProblem.measurable_radiusPotential {μ : Measure Ω}
    (P : GeneralizedProblem N m n μ) (c : Fin p → Parameters) :
    Measurable (P.radiusPotential c) := by
  have hx : Measurable (fun z : RadiusStates N m n => points z.1) :=
    measurable_points.comp measurable_fst
  have hf : Measurable (fun z : RadiusStates N m n => P.f (points z.1)-P.floor) :=
    (continuous_iff_continuousAt.mpr (fun x => (P.derivative x).continuousAt)).measurable.comp hx
      |>.sub measurable_const
  apply hf.add
  apply Finset.measurable_sum
  intro i _
  have hM j : Measurable (fun z : RadiusStates N m n => (z.1 i).M j) :=
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst).snd.snd.fst
  have hG j : Measurable (fun z : RadiusStates N m n => (z.1 i).G j) :=
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst).snd.snd.snd
  have hg := (measurable_pi_apply i).comp (P.gradient_continuous.measurable.comp hx)
  have hZ : Measurable (fun z : RadiusStates N m n => z.2 i) :=
    (measurable_pi_apply i).comp measurable_snd
  have hD := (((measurable_average hM).sub hg).sub hZ).norm.const_mul (radiusDWeight (c i))
  have hL := (measurable_average fun j => ((hM j).sub
    ((measurable_pi_apply i).comp ((P.local_gradient_continuous j).measurable.comp hx))).norm).const_mul
      (radiusLWeight (c i))
  have hS := (measurable_average fun j => ((hM j).sub (hG j)).norm).const_mul (radiusSWeight (c i))
  exact ((hD.add hL).add hS).add ((hZ.norm.pow_const 2).const_mul _)

theorem GeneralizedProblem.radiusPotential_integrable {μ : Measure Ω} [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters} (A : Scheme N m n c)
    (z : RadiusStates N m n) :
    Integrable (fun ω => P.radiusPotential c (radiusNext P A z ω)) μ := by
  have he (ω : Ω) : P.radiusPotential c (radiusNext P A z ω) =
      P.f (radiusPoints c z.1)-P.floor + ∑ i,
        radiusEnergy (c i) (radiusUpdate (c i) (A.CU i) (z.1 i)
          ((P.radiusFrozen c z.1 i).replies ω)) (P.grad (radiusPoints c z.1) i)
          (newMomentum (c i).β (z.2 i) 0 (P.radiusFrozen c z.1 i).mean ω)
          (fun j => P.localGrad j (radiusPoints c z.1) i) := by
    simp only [GeneralizedProblem.radiusPotential, points_radiusNext,
      (radiusNext_frozen P A z ω _).1, (radiusNext_frozen P A z ω _).2]
  simp_rw [he]
  exact (integrable_const _).add (integrable_finsetSum _ fun i _ =>
    radiusEnergy_integrable (c i) (A.measurable_CU i) (A.contracts_CU i) (z.1 i)
      (z.2 i) (P.grad (radiusPoints c z.1) i) (P.radiusFrozen c z.1 i))

end EF21Muon.Distributed
