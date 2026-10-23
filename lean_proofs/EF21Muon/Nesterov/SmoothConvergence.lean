import EF21Muon.Nesterov.Setup
import EF21Muon.Probability.GeometricDrift
import EF21Muon.Distributed.Bounds

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def smoothPotential (P : Problem N m n μ c) (s : States N m n) : ℝ :=
  P.f (points s)-P.fmin+∑ i, energy (c i) (P.ell2 i) (s i)
    (fun j => P.localGrad j (points s) i)

def smoothCost (P : Problem N m n μ c) (s : States N m n) : ℝ :=
  ∑ i, (c i).γ/4*nuclearNorm (P.grad (points s) i)^2

def smoothNoise (P : Problem N m n μ c) : ℝ := ∑ i, (c i).γ*noise (c i)*P.σ i^2

theorem smoothPotential_nonneg (P : Problem N m n μ c) (s : States N m n) :
    0 ≤ smoothPotential P s :=
  add_nonneg (sub_nonneg.mpr (P.lower_bound _))
    (Finset.sum_nonneg (fun i _ => energy_nonneg (c i) (P.ell2_nonneg i) _ _))

theorem smoothCost_nonneg (P : Problem N m n μ c) (s : States N m n) :
    0 ≤ smoothCost P s :=
  Finset.sum_nonneg (fun i _ => mul_nonneg (div_nonneg (c i).γ_pos.le (by norm_num)) (sq_nonneg _))

theorem smoothNoise_nonneg (P : Problem N m n μ c) : 0 ≤ smoothNoise P := by
  apply Finset.sum_nonneg
  intro i _
  rcases weights_nonneg (c i) (P.ell2_nonneg i) with ⟨_,_,_,hr,hf,_,_⟩
  exact mul_nonneg (mul_nonneg (c i).γ_pos.le
    (mul_nonneg (add_nonneg hf hr) (sq_nonneg _))) (sq_nonneg _)

theorem smoothEnergy_measurable (P : Problem N m n μ c) (i : Fin p) :
    Measurable (fun s : States N m n => energy (c i) (P.ell2 i) (s i)
      (fun j => P.localGrad j (points s) i)) :=
  measurable_energy (measurable_pi_apply i)
    (fun j => (measurable_pi_apply i).comp ((P.localGrad_measurable j).comp measurable_points)) c _

theorem smoothPotential_measurable (P : Problem N m n μ c) : Measurable (smoothPotential P) := by
  have hf : Continuous P.f := continuous_iff_continuousAt.mpr (fun x => (P.derivative x).continuousAt)
  exact ((hf.measurable.comp measurable_points).sub measurable_const).add
    (Finset.measurable_sum _ (fun i _ => smoothEnergy_measurable P i))

theorem smoothCost_measurable (P : Problem N m n μ c) : Measurable (smoothCost P) :=
  Finset.measurable_sum _ (fun i _ => ((continuous_nuclearNorm.measurable.comp
    ((measurable_pi_apply i).comp (P.grad_measurable.comp measurable_points))).pow_const 2).const_mul _)

theorem smoothLayer_drift (P : Problem N m n μ c) (A : Scheme N m n c)
    (hsmall : ∀ i, 2*(c i).L*(c i).γ+4*zeta (c i) (P.ell2 i) (A.τ i)*(c i).γ^2 ≤ 1)
    (s : States N m n) (i : Fin p) :
    inner ℝ (P.grad (points s) i) (nextX (c i) (s i).1-(s i).1.x)+
      (c i).L/2*spectralNorm (nextX (c i) (s i).1-(s i).1.x)^2+
      (∫ ω, energy (c i) (P.ell2 i) (next A P.reply s ω i)
        (fun j => P.localGrad j (serverPoints c s) i) ∂μ)+
      (c i).γ/4*nuclearNorm (P.grad (points s) i)^2 ≤
    energy (c i) (P.ell2 i) (s i) (fun j => P.localGrad j (points s) i)+
      (c i).γ*noise (c i)*P.σ i^2 := by
  have hmov : spectralNorm (serverPoints c s i-points s i) ≤
      (c i).γ*nuclearNorm (average (s i).1.G) := movement_spectral (c i) (s i).1.server
  have hbro : spectralNorm (queries A s i-serverPoints c s i) ≤
      ‖nextW (c i) (A.CP i) (s i).1-nextX (c i) (s i).1‖ := spectralNorm_le_frobenius _
  have ha := average_mono (fun j => gradient_difference_sq (P.localL_nonneg i j)
    (fun x y => P.localSmooth j x y i) _ _ hmov)
  have hb := average_mono (fun j => gradient_difference_sq (P.localL_nonneg i j)
    (fun x y => P.localSmooth j x y i) _ _ hbro)
  simp only [norm_sub_rev (P.localGrad _ (serverPoints c s) i)] at ha
  have ha' : average (fun j => ‖P.localGrad j (points s) i-P.localGrad j (serverPoints c s) i‖^2) ≤
      P.ell2 i*(c i).γ^2*nuclearNorm (average (s i).1.G)^2 := by
    convert ha using 1
    simp only [mul_pow, mul_assoc, mul_comm _ ((c i).γ^2*_), average_mul, Problem.ell2]
  have hb' : average (fun j => ‖P.localGrad j (queries A s) i-P.localGrad j (serverPoints c s) i‖^2) ≤
      P.ell2 i*‖nextW (c i) (A.CP i) (s i).1-nextX (c i) (s i).1‖^2 := by
    convert hb using 1
    simp only [mul_comm _ (‖_‖^2), average_mul, Problem.ell2]
  simp only [next_frozen P A s]
  exact layer_descent (c i) P.workers_pos (P.ell2_nonneg i) (A.measurable_CU i)
    (A.contracts_CU i) (A.broadcast i) (A.dual i) (hsmall i) (s i)
    (P.grad (points s) i) (fun j => P.localGrad j (points s) i)
    (fun j => P.localGrad j (serverPoints c s) i) (fun j => P.localGrad j (queries A s) i)
    (P.mean_gradient _ i) (frozenNoise P A s i) ha' hb'

theorem smoothEnergy_next_integrable (P : Problem N m n μ c) (A : Scheme N m n c)
    (s : States N m n) (i : Fin p) :
    Integrable (fun ω => energy (c i) (P.ell2 i) (next A P.reply s ω i)
      (fun j => P.localGrad j (serverPoints c s) i)) μ := by
  simp only [next_frozen P A s]
  exact energy_integrable (c i) _ (A.measurable_CU i) (A.contracts_CU i) _ _ _ _ (frozenNoise P A s i)

theorem smoothPotential_next_integrable (P : Problem N m n μ c) (A : Scheme N m n c)
    (s : States N m n) : Integrable (fun ω => smoothPotential P (next A P.reply s ω)) μ := by
  simp only [smoothPotential, points_next]
  exact (integrable_const _).add (integrable_finsetSum _ (fun i _ => smoothEnergy_next_integrable P A s i))

theorem smoothPotential_descent (P : Problem N m n μ c) (A : Scheme N m n c)
    (hsmall : ∀ i, 2*(c i).L*(c i).γ+4*zeta (c i) (P.ell2 i) (A.τ i)*(c i).γ^2 ≤ 1)
    (s : States N m n) :
    (∫ ω, smoothPotential P (next A P.reply s ω) ∂μ)+smoothCost P s ≤
      smoothPotential P s+smoothNoise P := by
  have hd := smooth_upper P.derivative P.smooth (points s) (serverPoints c s)
  have h := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => smoothLayer_drift P A hsmall s i)
  simp only [smoothPotential, points_next]
  rw [integral_add (integrable_const _) (integrable_finsetSum _
    (fun i _ => smoothEnergy_next_integrable P A s i)), integral_const, probReal_univ, one_smul,
    integral_finsetSum _ (fun i _ => smoothEnergy_next_integrable P A s i)]
  simp only [Finset.sum_add_distrib] at h hd
  dsimp only [smoothCost, smoothNoise, points, serverPoints, Distributed.points,
    Distributed.serverPoints, core] at *
  linarith only [h,hd]

/-- Full finite-time stochastic convergence of the Nesterov algorithm, including
both uplink and downlink compression, for arbitrary finite-variance noise laws. -/
theorem smooth_finite_time (P : Problem N m n μ c) (A : Scheme N m n c)
    (hsmall : ∀ i, 2*(c i).L*(c i).γ+4*zeta (c i) (P.ell2 i) (A.τ i)*(c i).γ^2 ≤ 1)
    (T : ℕ) (s : States N m n) :
    Integrable (fun ξ => ((run (next A P.reply) T s ξ).map (smoothCost P)).sum) (pathMeasure μ T) ∧
    (∫ ξ, ((run (next A P.reply) T s ξ).map (smoothCost P)).sum ∂pathMeasure μ T) ≤
      smoothPotential P s+(T:ℝ)*smoothNoise P := by
  simpa only [one_pow, one_mul] using accumulated_cost_geometric_bound μ
    (a := 1) (by norm_num) (smoothNoise_nonneg P) (measurable_next A P.measurable_reply)
    (smoothCost_measurable P) (smoothPotential_measurable P) (smoothCost_nonneg P)
    (smoothPotential_nonneg P) (smoothPotential_next_integrable P A)
    (by simpa only [one_mul] using smoothPotential_descent P A hsmall) T s

def smoothStationarity (P : Problem N m n μ c) (A : Scheme N m n c)
    (γbar : ℝ) (s : States N m n) (T : ℕ) : ℝ :=
  (∫ ξ, ((run (next A P.reply) T s ξ).map (fun s =>
    ∑ i, (c i).γ/γbar*nuclearNorm (P.grad (points s) i)^2)).sum ∂pathMeasure μ T)/T

theorem smooth_average (P : Problem N m n μ c) (A : Scheme N m n c)
    (hsmall : ∀ i, 2*(c i).L*(c i).γ+4*zeta (c i) (P.ell2 i) (A.τ i)*(c i).γ^2 ≤ 1)
    {γbar : ℝ} (hγ : 0 < γbar) (s : States N m n) (T : ℕ) (hT : 0 < T) :
    smoothStationarity P A γbar s T ≤
      4*smoothPotential P s/(γbar*T)+4*smoothNoise P/γbar := by
  have he (s : States N m n) : (∑ i, (c i).γ/γbar*nuclearNorm (P.grad (points s) i)^2) =
      4/γbar*smoothCost P s := by
    simp only [smoothCost, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _; ring
  have h := (smooth_finite_time P A hsmall T s).2
  have hTr : (0:ℝ) < T := by exact_mod_cast hT
  simp only [smoothStationarity, he, sum_map_scale, integral_const_mul]
  apply (div_le_iff₀ hTr).mpr
  have hh := mul_le_mul_of_nonneg_left h (by positivity : 0 ≤ 4/γbar)
  have heq : (4*smoothPotential P s/(γbar*T)+4*smoothNoise P/γbar)*T =
      4/γbar*(smoothPotential P s+T*smoothNoise P) := by field_simp
  rw [heq]; exact hh

end EF21Muon.Nesterov
