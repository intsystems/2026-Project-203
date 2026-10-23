import EF21Muon.Distributed.Setup
import EF21Muon.Distributed.GeneralizedGeometry
import EF21Muon.Probability.FirstMoment

/-! First-moment stochastic tracking for the constant-radius, exact-downlink
algorithm. The additional noise EMA is an analysis variable; it does not alter
the position, worker momentum, or compressed estimates. -/

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N m n : ℕ}

def radiusPoint (c : Parameters) (s : LayerState N m n) : Space m n :=
  s.x - c.γ • polar (average s.G)

def radiusUpdate (c : Parameters) (CU : Space m n → Space m n)
    (s : LayerState N m n) (reply : Fin N → Space m n) : LayerState N m n :=
  let M := fun j => (1-c.β) • s.M j + c.β • reply j
  (radiusPoint c s, radiusPoint c s, M, fun j => s.G j + CU (M j-s.G j))

def firstLocal (s : LayerState N m n) (g : Fin N → Space m n) : ℝ :=
  average (fun j => ‖s.M j-g j‖)

def firstCompression (s : LayerState N m n) : ℝ := average (fun j => ‖s.M j-s.G j‖)

def debiasedError (s : LayerState N m n) (g Z : Space m n) : ℝ :=
  ‖average s.M-g-Z‖

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

def Noise.mean {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ)
    (ω : Ω) : Space m n := average (fun j => O.value j ω)

omit [IsProbabilityMeasure μ] in
theorem Noise.mean_memLp {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    MemLp O.mean 2 μ := average_memLp O.memLp

theorem Noise.mean_centered {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    (∫ ω, O.mean ω ∂μ) = 0 := by
  simp only [Noise.mean, average, integral_smul,
    integral_finsetSum _ (fun j _ => (O.memLp j).integrable (by norm_num)),
    O.centered, Finset.sum_const_zero, smul_zero]

theorem Noise.mean_variance (hN : 0 < N) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) : (∫ ω, ‖O.mean ω‖^2 ∂μ) ≤ σ^2/N :=
  independent_average_variance hN O.memLp O.independent O.centered O.variance

omit [IsProbabilityMeasure μ] in
theorem radius_averaged_momentum (c : Parameters) (CU : Space m n → Space m n)
    (s : LayerState N m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) (ω : Ω) :
    average (radiusUpdate c CU s (O.replies ω)).M =
      newMomentum c.β (average s.M) (average q) O.mean ω := by
  change average (fun j => (1-c.β) • s.M j + c.β • (q j+O.value j ω)) = _
  rw [average_add, average_smul, average_smul, average_add]
  rfl

theorem radius_local_integrable (c : Parameters) (CU : Space m n → Space m n)
    (s : LayerState N m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) :
    Integrable (fun ω => firstLocal (radiusUpdate c CU s (O.replies ω)) q) μ := by
  have hi (j : Fin N) := (((newMomentum_memLp c.β (s.M j) (q j) (O.memLp j)).sub
    (memLp_const (q j))).integrable (by norm_num)).norm
  change Integrable (fun ω => (N:ℝ)⁻¹ * ∑ j, ‖newMomentum c.β (s.M j) (q j) (O.value j) ω-q j‖) μ
  exact (integrable_finsetSum _ (fun j _ => hi j)).const_mul _

theorem radius_local_recurrence (c : Parameters) (hN : 0 < N)
    (CU : Space m n → Space m n) (s : LayerState N m n)
    (old q : Fin N → Space m n) {σ : ℝ} (hσ : 0 ≤ σ)
    (O : Noise N m n μ q σ) :
    (∫ ω, firstLocal (radiusUpdate c CU s (O.replies ω)) q ∂μ) ≤
      (1-c.β)*firstLocal s old +
        (1-c.β)*average (fun j => ‖old j-q j‖) + c.β*σ := by
  have hi (j : Fin N) := (((newMomentum_memLp c.β (s.M j) (q j) (O.memLp j)).sub
    (memLp_const (q j))).integrable (by norm_num)).norm
  change (∫ ω, average (fun j => ‖newMomentum c.β (s.M j) (q j) (O.value j) ω-q j‖) ∂μ) ≤ _
  simp only [Pi.sub_apply] at hi
  rw [integral_average hi]
  have h := average_mono (fun j => momentum_first_moment c.β_pos.le c.β_le_one
    (s.M j) (old j) (q j) (O.memLp j) hσ (O.variance j))
  simpa only [average_add, average_mul, average_const hN, firstLocal] using h

omit [IsProbabilityMeasure μ] in
theorem radius_compression_residual (c : Parameters) (CU : Space m n → Space m n)
    (s : LayerState N m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) (ω : Ω) (j : Fin N) :
    ‖(radiusUpdate c CU s (O.replies ω)).M j - (radiusUpdate c CU s (O.replies ω)).G j‖ =
      ‖CU (newMomentum c.β (s.M j) (q j) (O.value j) ω-s.G j) -
        (newMomentum c.β (s.M j) (q j) (O.value j) ω-s.G j)‖ := by
  rw [norm_sub_rev]
  congr 1
  dsimp [radiusUpdate, Noise.replies, newMomentum, LayerState.M, LayerState.G]
  module

theorem radius_compression_integrable (c : Parameters) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU) (s : LayerState N m n)
    {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    Integrable (fun ω => firstCompression (radiusUpdate c CU s (O.replies ω))) μ := by
  have hi (j : Fin N) := ((residual_memLp hm c.αU_pos hc
    ((newMomentum_memLp c.β (s.M j) (q j) (O.memLp j)).sub
      (memLp_const (s.G j)))).integrable (by norm_num)).norm
  simp only [firstCompression, radius_compression_residual, average, smul_eq_mul]
  exact (integrable_finsetSum _ (fun j _ => hi j)).const_mul _

theorem radius_compression_recurrence (c : Parameters) (hN : 0 < N)
    {CU : Space m n → Space m n} (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (s : LayerState N m n) (old q : Fin N → Space m n) {σ : ℝ} (hσ : 0 ≤ σ)
    (O : Noise N m n μ q σ) :
    (∫ ω, firstCompression (radiusUpdate c CU s (O.replies ω)) ∂μ) ≤
      Real.sqrt (1-c.αU)*firstCompression s + Real.sqrt (1-c.αU)*c.β*firstLocal s old +
        Real.sqrt (1-c.αU)*c.β*average (fun j => ‖old j-q j‖) +
        Real.sqrt (1-c.αU)*c.β*σ := by
  have hi (j : Fin N) := ((residual_memLp hm c.αU_pos hc
    ((newMomentum_memLp c.β (s.M j) (q j) (O.memLp j)).sub
      (memLp_const (s.G j)))).integrable (by norm_num)).norm
  simp only [firstCompression, radius_compression_residual]
  simp only [Pi.sub_apply] at hi
  rw [integral_average hi]
  have h := average_mono (fun j => compression_first_moment c.β_pos.le c.αU_pos c.αU_le_one
    (s.M j) (s.G j) (old j) (q j) hm hc (O.memLp j) hσ (O.variance j))
  simpa only [average_add, average_mul, average_const hN, firstLocal, firstCompression] using h

omit [IsProbabilityMeasure μ] in
theorem radius_debiased_identity (c : Parameters) (CU : Space m n → Space m n)
    (s : LayerState N m n) (Z old new : Space m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) (hmean : average q = new) (ω : Ω) :
    average (radiusUpdate c CU s (O.replies ω)).M - new - newMomentum c.β Z 0 O.mean ω =
      (1-c.β) • ((average s.M-old-Z)+(old-new)) := by
  rw [radius_averaged_momentum, hmean]
  exact debiased_momentum_identity _ _ _ _ _ _ _

theorem radius_debiased_integrable (c : Parameters) (CU : Space m n → Space m n)
    (s : LayerState N m n) (Z new : Space m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) :
    Integrable (fun ω => debiasedError (radiusUpdate c CU s (O.replies ω)) new
      (newMomentum c.β Z 0 O.mean ω)) μ := by
  simp only [debiasedError, radius_averaged_momentum]
  exact ((((newMomentum_memLp c.β (average s.M) (average q) O.mean_memLp).sub
    (memLp_const new)).sub (newMomentum_memLp c.β Z 0 O.mean_memLp)).integrable (by norm_num)).norm

theorem radius_debiased_recurrence (c : Parameters) (CU : Space m n → Space m n)
    (s : LayerState N m n) (Z old new : Space m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) (hmean : average q = new) :
    (∫ ω, debiasedError (radiusUpdate c CU s (O.replies ω)) new
      (newMomentum c.β Z 0 O.mean ω) ∂μ) ≤
      (1-c.β)*debiasedError s old Z + (1-c.β)*‖old-new‖ := by
  simp only [debiasedError, radius_debiased_identity c CU s Z old new O hmean,
    integral_const, probReal_univ, one_smul, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (sub_nonneg.mpr c.β_le_one)]
  have h := mul_le_mul_of_nonneg_left (norm_add_le (average s.M-old-Z) (old-new))
    (sub_nonneg.mpr c.β_le_one)
  simpa only [mul_add] using h

theorem radius_noise_recurrence (c : Parameters) (hN : 0 < N)
    (Z : Space m n) {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    (∫ ω, ‖newMomentum c.β Z 0 O.mean ω‖^2 ∂μ) ≤
      (1-c.β)*‖Z‖^2 + c.β^2*σ^2/N := by
  have h := noise_ema_recurrence c.β_pos.le c.β_le_one Z O.mean_memLp
    O.mean_centered (O.mean_variance hN)
  convert h using 1; ring

end EF21Muon.Distributed
