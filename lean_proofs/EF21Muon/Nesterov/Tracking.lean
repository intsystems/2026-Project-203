import EF21Muon.Nesterov.Base

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N m n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]

theorem momentum_memLp (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) (j : Fin N) :
    MemLp (fun ω => (update c CU CP s (O.replies ω)).1.M j) 2 μ :=
  newMomentum_memLp c.β (s.1.M j) (q j) (O.memLp j)

theorem target_memLp (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) (j : Fin N) :
    MemLp (fun ω => target c (update c CU CP s (O.replies ω)) j) 2 μ := by
  simp only [target]
  exact ((momentum_memLp c CU CP s O j).const_smul (2-c.β)).sub
    (memLp_const ((1-c.β) • s.1.M j))

theorem local_integrable (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) (g : Fin N → Space m n) :
    Integrable (fun ω => localError (update c CU CP s (O.replies ω)).1 g) μ := by
  exact (integrable_finsetSum _ (fun j _ => square_integrable
    ((momentum_memLp c CU CP s O j).sub (memLp_const (g j))))).const_mul _

theorem increment_integrable (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) :
    Integrable (fun ω => increment (update c CU CP s (O.replies ω))) μ := by
  exact (integrable_finsetSum _ (fun j _ => square_integrable
    ((momentum_memLp c CU CP s O j).sub (memLp_const (s.1.M j))))).const_mul _

theorem compression_integrable (c : Parameters) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (CP : Space m n → Space m n) (s : Layer N m n)
    {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    Integrable (fun ω => compression c (update c CU CP s (O.replies ω))) μ := by
  have hi j := square_integrable (residual_memLp hm c.αU_pos hc
    ((target_memLp c CU CP s O j).sub (memLp_const (s.1.G j))))
  simp only [compression, residual_identity, average, smul_eq_mul]
  exact (integrable_finsetSum _ (fun j _ => hi j)).const_mul _

theorem local_recurrence (c : Parameters) (hN : 0 < N) (CU CP : Space m n → Space m n)
    (s : Layer N m n) (old new q : Fin N → Space m n) {σ : ℝ}
    (O : Noise N m n μ q σ) {a b : ℝ}
    (ha : average (fun j => ‖old j-new j‖^2) ≤ a)
    (hb : average (fun j => ‖q j-new j‖^2) ≤ b) :
    (∫ ω, localError (update c CU CP s (O.replies ω)).1 new ∂μ) ≤
      (1-c.β/2)*localError s.1 old+2/c.β*a+c.β^2*(1+2/c.β)*b+c.β^2*σ^2 :=
  Distributed.local_recurrence c hN CU CP s.1 old new q O ha hb

theorem increment_recurrence (c : Parameters) (hN : 0 < N)
    (CU CP : Space m n → Space m n) (s : Layer N m n)
    (old new q : Fin N → Space m n) {σ : ℝ} (O : Noise N m n μ q σ)
    {a b : ℝ} (ha : average (fun j => ‖old j-new j‖^2) ≤ a)
    (hb : average (fun j => ‖q j-new j‖^2) ≤ b) :
    (∫ ω, increment (update c CU CP s (O.replies ω)) ∂μ) ≤
      3*c.β^2*(localError s.1 old+a+b)+c.β^2*σ^2 := by
  have hi j := square_integrable ((momentum_memLp c CU CP s O j).sub
    (memLp_const (s.1.M j)))
  simp only [Pi.sub_apply] at hi
  have hj (j : Fin N) :
      (∫ ω, ‖(update c CU CP s (O.replies ω)).1.M j-s.1.M j‖^2 ∂μ) ≤
        3*c.β^2*(‖s.1.M j-old j‖^2+‖old j-new j‖^2+‖q j-new j‖^2)+c.β^2*σ^2 := by
    have he (ω : Ω) : (update c CU CP s (O.replies ω)).1.M j-s.1.M j =
        c.β • (q j-s.1.M j)+c.β • O.value j ω := by
      dsimp [update, LayerState.M, Noise.replies]; module
    simp_rw [he]
    rw [affine_variance_identity _ _ (O.memLp j) (O.centered j),
      norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
    have ht := three_sq (q j-new j) (new j-old j) (old j-s.1.M j)
    rw [sub_add_sub_cancel, sub_add_sub_cancel, norm_sub_rev (new j) (old j),
      norm_sub_rev (old j) (s.1.M j)] at ht
    nlinarith only [mul_le_mul_of_nonneg_left ht (sq_nonneg c.β),
      mul_le_mul_of_nonneg_left (O.variance j) (sq_nonneg c.β)]
  change (∫ ω, average (fun j => ‖(update c CU CP s (O.replies ω)).1.M j-s.1.M j‖^2) ∂μ) ≤ _
  rw [integral_average hi]
  have h := average_mono hj
  simp only [average_add, average_mul, average_const hN] at h
  have hab := mul_le_mul_of_nonneg_left (add_le_add ha hb) (by positivity : 0 ≤ 3*c.β^2)
  dsimp only [localError]
  linarith only [h, hab]

theorem compression_step (c : Parameters) {CU : Space m n → Space m n}
    (hc : SquaredContracts CU c.αU) (CP : Space m n → Space m n)
    (s : Layer N m n) (reply : Fin N → Space m n) :
    compression c (update c CU CP s reply) ≤
      (1-c.αU/2)*compression c s+16/c.αU*increment (update c CU CP s reply)+
        4/c.αU*increment s := by
  have hj (j : Fin N) :
      ‖target c (update c CU CP s reply) j-(update c CU CP s reply).1.G j‖^2 ≤
        (1-c.αU/2)*‖target c s j-s.1.G j‖^2+
        16/c.αU*‖(update c CU CP s reply).1.M j-s.1.M j‖^2+
        4/c.αU*‖s.1.M j-s.2 j‖^2 := by
    rw [residual_identity]
    have h := hc (target c (update c CU CP s reply) j-s.1.G j)
    have ht := contracted_add_sq (target c s j-s.1.G j)
      (target c (update c CU CP s reply) j-target c s j) c.αU_pos c.αU_le_one
    have he : (target c s j-s.1.G j)+
        (target c (update c CU CP s reply) j-target c s j) =
        target c (update c CU CP s reply) j-s.1.G j := by module
    rw [he] at ht
    have hU := c.αU_pos
    have hd := mul_le_mul_of_nonneg_left (target_increment_sq c CU CP s reply j)
      (by positivity : 0 ≤ 2/c.αU)
    simp only [div_eq_mul_inv] at h ht hd ⊢
    nlinarith only [h, ht, hd]
  simpa only [compression, increment, average_add, average_mul, update] using average_mono hj

theorem compression_recurrence (c : Parameters) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (CP : Space m n → Space m n) (s : Layer N m n)
    {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    (∫ ω, compression c (update c CU CP s (O.replies ω)) ∂μ) ≤
      (1-c.αU/2)*compression c s+
      16/c.αU*(∫ ω, increment (update c CU CP s (O.replies ω)) ∂μ)+4/c.αU*increment s := by
  have hi := (increment_integrable c CU CP s O).const_mul (16/c.αU)
  have hc0 : Integrable (fun _ : Ω => (1-c.αU/2)*compression c s) μ := integrable_const _
  have hc1 : Integrable (fun _ : Ω => 4/c.αU*increment s) μ := integrable_const _
  have h := integral_mono (compression_integrable c hm hc CP s O)
    ((hc0.add hi).add hc1)
    (fun ω => compression_step c hc CP s (O.replies ω))
  simp only [Pi.add_apply] at h
  have hi0 : Integrable (fun ω => (1-c.αU/2)*compression c s+
      16/c.αU*increment (update c CU CP s (O.replies ω))) μ := hc0.add hi
  rw [integral_add hi0 hc1, integral_add hc0 hi] at h
  simpa only [integral_const_mul, integral_const, probReal_univ, one_smul] using h

end EF21Muon.Nesterov
