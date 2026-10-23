import EF21Muon.Nesterov.Tracking

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N m n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]

theorem first_local_integrable (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    Integrable (fun ω => firstLocal (update c CU CP s (O.replies ω)).1 q) μ :=
  radius_local_integrable c CU s.1 O

theorem first_local_recurrence (c : Parameters) (hN : 0 < N)
    (CU CP : Space m n → Space m n) (s : Layer N m n)
    (old q : Fin N → Space m n) {σ : ℝ} (hσ : 0 ≤ σ) (O : Noise N m n μ q σ) :
    (∫ ω, firstLocal (update c CU CP s (O.replies ω)).1 q ∂μ) ≤
      (1-c.β)*firstLocal s.1 old+(1-c.β)*average (fun j => ‖old j-q j‖)+c.β*σ :=
  radius_local_recurrence c hN CU s.1 old q hσ O

theorem first_increment_integrable (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    Integrable (fun ω => firstIncrement (update c CU CP s (O.replies ω))) μ := by
  exact (integrable_finsetSum _ (fun j _ =>
    (((momentum_memLp c CU CP s O j).sub (memLp_const (s.1.M j))).integrable (by norm_num)).norm)).const_mul _

theorem first_increment_recurrence (c : Parameters) (hN : 0 < N)
    (CU CP : Space m n → Space m n) (s : Layer N m n)
    (old q : Fin N → Space m n) {σ : ℝ} (hσ : 0 ≤ σ) (O : Noise N m n μ q σ) :
    (∫ ω, firstIncrement (update c CU CP s (O.replies ω)) ∂μ) ≤
      c.β*(firstLocal s.1 old+average (fun j => ‖old j-q j‖)+σ) := by
  have hi j := (((momentum_memLp c CU CP s O j).sub
    (memLp_const (s.1.M j))).integrable (by norm_num)).norm
  simp only [Pi.sub_apply] at hi
  have hj (j : Fin N) :
      (∫ ω, ‖(update c CU CP s (O.replies ω)).1.M j-s.1.M j‖ ∂μ) ≤
        c.β*(‖s.1.M j-old j‖+‖old j-q j‖+σ) := by
    have he (ω : Ω) : (update c CU CP s (O.replies ω)).1.M j-s.1.M j =
        c.β • ((q j-s.1.M j)+O.value j ω) := by
      dsimp [update, LayerState.M, Noise.replies]; module
    have hnoise := integral_norm_le_of_sq (O.memLp j) hσ (O.variance j)
    have hn := (O.memLp j).integrable (by norm_num) |>.norm
    have hb := c.β_pos.le
    have h := integral_mono (hi j)
      (((integrable_const ‖q j-s.1.M j‖).add hn).const_mul c.β)
      (fun ω => by
        rw [he, norm_smul, Real.norm_eq_abs, abs_of_nonneg hb]
        exact mul_le_mul_of_nonneg_left (norm_add_le _ _) hb)
    simp only [Pi.add_apply] at h
    rw [integral_const_mul, integral_add (integrable_const _) hn] at h
    simp only [integral_const, probReal_univ, one_smul] at h
    have ht := norm_add_le (q j-old j) (old j-s.1.M j)
    rw [sub_add_sub_cancel, norm_sub_rev (q j) (old j), norm_sub_rev (old j) (s.1.M j)] at ht
    nlinarith only [h,mul_le_mul_of_nonneg_left ht hb,mul_le_mul_of_nonneg_left hnoise hb]
  change (∫ ω, average (fun j => ‖(update c CU CP s (O.replies ω)).1.M j-s.1.M j‖) ∂μ) ≤ _
  rw [integral_average hi]
  simpa only [average_add,average_mul,average_const hN,firstLocal] using average_mono hj

theorem first_compression_integrable (c : Parameters) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (CP : Space m n → Space m n) (s : Layer N m n)
    {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    Integrable (fun ω => firstCompression c (update c CU CP s (O.replies ω))) μ := by
  have hi j := ((residual_memLp hm c.αU_pos hc
    ((target_memLp c CU CP s O j).sub (memLp_const (s.1.G j)))).integrable (by norm_num)).norm
  simp only [firstCompression,residual_identity,average,smul_eq_mul]
  exact (integrable_finsetSum _ (fun j _ => hi j)).const_mul _

theorem first_compression_step (c : Parameters) {CU : Space m n → Space m n}
    (hc : SquaredContracts CU c.αU) (CP : Space m n → Space m n)
    (s : Layer N m n) (reply : Fin N → Space m n) :
    firstCompression c (update c CU CP s reply) ≤
      Real.sqrt (1-c.αU)*firstCompression c s+
        2*Real.sqrt (1-c.αU)*firstIncrement (update c CU CP s reply)+
        Real.sqrt (1-c.αU)*firstIncrement s := by
  have hj (j : Fin N) :
      ‖target c (update c CU CP s reply) j-(update c CU CP s reply).1.G j‖ ≤
        Real.sqrt (1-c.αU)*‖target c s j-s.1.G j‖+
        2*Real.sqrt (1-c.αU)*‖(update c CU CP s reply).1.M j-s.1.M j‖+
        Real.sqrt (1-c.αU)*‖s.1.M j-s.2 j‖ := by
    rw [residual_identity]
    have h := sqrt_contraction c.αU_le_one hc (target c (update c CU CP s reply) j-s.1.G j)
    have ht := norm_add_le (target c s j-s.1.G j)
      (target c (update c CU CP s reply) j-target c s j)
    have he : (target c s j-s.1.G j)+(target c (update c CU CP s reply) j-target c s j) =
        target c (update c CU CP s reply) j-s.1.G j := by module
    rw [he] at ht
    have hd := norm_sub_le ((2-c.β) • ((update c CU CP s reply).1.M j-s.1.M j))
      ((1-c.β) • (s.1.M j-s.2 j))
    rw [← target_increment] at hd
    simp only [norm_smul,Real.norm_eq_abs,abs_of_nonneg (by linarith [c.β_le_one] : 0 ≤ 2-c.β),
      abs_of_nonneg (sub_nonneg.mpr c.β_le_one)] at hd
    have hβ := c.β_pos
    have hΔ : ‖target c (update c CU CP s reply) j-target c s j‖ ≤
        2*‖(update c CU CP s reply).1.M j-s.1.M j‖+‖s.1.M j-s.2 j‖ := by
      nlinarith [norm_nonneg ((update c CU CP s reply).1.M j-s.1.M j), norm_nonneg (s.1.M j-s.2 j)]
    nlinarith only [h,mul_le_mul_of_nonneg_left ht (Real.sqrt_nonneg (1-c.αU)),
      mul_le_mul_of_nonneg_left hΔ (Real.sqrt_nonneg (1-c.αU))]
  simpa only [firstCompression,firstIncrement,average_add,average_mul,update] using average_mono hj

theorem first_compression_recurrence (c : Parameters) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (CP : Space m n → Space m n) (s : Layer N m n)
    {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    (∫ ω, firstCompression c (update c CU CP s (O.replies ω)) ∂μ) ≤
      Real.sqrt (1-c.αU)*firstCompression c s+
        2*Real.sqrt (1-c.αU)*(∫ ω, firstIncrement (update c CU CP s (O.replies ω)) ∂μ)+
        Real.sqrt (1-c.αU)*firstIncrement s := by
  have hi := (first_increment_integrable c CU CP s O).const_mul (2*Real.sqrt (1-c.αU))
  have hc0 : Integrable (fun _ : Ω => Real.sqrt (1-c.αU)*firstCompression c s) μ := integrable_const _
  have hc1 : Integrable (fun _ : Ω => Real.sqrt (1-c.αU)*firstIncrement s) μ := integrable_const _
  have hi0 : Integrable (fun ω => Real.sqrt (1-c.αU)*firstCompression c s+
      2*Real.sqrt (1-c.αU)*firstIncrement (update c CU CP s (O.replies ω))) μ := hc0.add hi
  have h := integral_mono (first_compression_integrable c hm hc CP s O)
    (hi0.add hc1) (fun ω => first_compression_step c hc CP s (O.replies ω))
  simp only [Pi.add_apply] at h
  rw [integral_add hi0 hc1, integral_add hc0 hi] at h
  simpa only [integral_const_mul,integral_const,probReal_univ,one_smul] using h

end EF21Muon.Nesterov
