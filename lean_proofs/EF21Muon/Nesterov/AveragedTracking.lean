import EF21Muon.Nesterov.Tracking

/-! Average the momentum and its increment before squaring. Only the
compression recurrences use the workerwise squared errors. -/

noncomputable section

namespace EF21Muon.Nesterov.Averaged

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N m n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]

def meanIncrement (s : Layer N m n) : ℝ := ‖average s.1.M - average s.2‖ ^ 2

theorem mean_error_sq (c : Parameters) (hN : 0 < N) (s : Layer N m n)
    (g : Space m n) :
    ‖average s.1.G-g‖^2 ≤ 3*(globalError s.1 g+compression c s+meanIncrement s) := by
  have he : average s.1.G-g = (average s.1.M-g) +
      average (fun j => s.1.G j-target c s j) +
      (1-c.β) • (average s.1.M-average s.2) := by
    simp only [target, average_sub, average_smul]
    module
  have hh := average_sq_le hN (fun j => s.1.G j-target c s j)
  have hh' : ‖average (fun j => s.1.G j-target c s j)‖^2 ≤ compression c s := by
    simpa only [compression, Probability.average, smul_eq_mul, norm_sub_rev] using hh
  have h := three_sq (average s.1.M-g) (average (fun j => s.1.G j-target c s j))
    ((1-c.β) • (average s.1.M-average s.2))
  have ha : (1-c.β)^2 ≤ 1 := by nlinarith [c.β_pos, c.β_le_one]
  rw [← he, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs] at h
  dsimp [globalError, meanIncrement]
  nlinarith [mul_le_mul_of_nonneg_right ha (sq_nonneg ‖average s.1.M-average s.2‖)]

theorem global_integrable (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) (g : Space m n) :
    Integrable (fun ω => globalError (update c CU CP s (O.replies ω)).1 g) μ :=
  square_integrable ((average_memLp (fun j => momentum_memLp c CU CP s O j)).sub (memLp_const g))

theorem meanIncrement_integrable (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) {q : Fin N → Space m n} {σ : ℝ}
    (O : Noise N m n μ q σ) :
    Integrable (fun ω => meanIncrement (update c CU CP s (O.replies ω))) μ :=
  square_integrable ((average_memLp (fun j => momentum_memLp c CU CP s O j)).sub
    (memLp_const (average s.1.M)))

theorem global_recurrence (c : Parameters) (hN : 0 < N) (CU CP : Space m n → Space m n)
    (s : Layer N m n) (old new : Space m n) (q : Fin N → Space m n) {σ : ℝ}
    (O : Noise N m n μ q σ) {a b : ℝ}
    (ha : ‖old-new‖^2 ≤ a) (hb : ‖average q-new‖^2 ≤ b) :
    (∫ ω, globalError (update c CU CP s (O.replies ω)).1 new ∂μ) ≤
      (1-c.β/2)*globalError s.1 old+2/c.β*a+c.β^2*(1+2/c.β)*b+c.β^2*σ^2/N :=
  Distributed.global_recurrence c hN CU CP s.1 old new q O ha hb

theorem meanIncrement_recurrence (c : Parameters) (hN : 0 < N)
    (CU CP : Space m n → Space m n) (s : Layer N m n)
    (old new : Space m n) (q : Fin N → Space m n) {σ : ℝ}
    (O : Noise N m n μ q σ) {a b : ℝ}
    (ha : ‖old-new‖^2 ≤ a) (hb : ‖average q-new‖^2 ≤ b) :
    (∫ ω, meanIncrement (update c CU CP s (O.replies ω)) ∂μ) ≤
      3*c.β^2*(globalError s.1 old+a+b)+c.β^2*σ^2/N := by
  have hz : (∫ ω, average (fun j => O.value j ω) ∂μ) = 0 := by
    simp only [Probability.average, integral_smul,
      integral_finsetSum _ (fun j _ => (O.memLp j).integrable (by norm_num)),
      O.centered, Finset.sum_const_zero, smul_zero]
  have hv := independent_average_variance hN O.memLp O.independent O.centered O.variance
  have he (ω : Ω) : average (update c CU CP s (O.replies ω)).1.M-average s.1.M =
      c.β • (average q-average s.1.M)+c.β • average (fun j => O.value j ω) := by
    change average (fun j => (1-c.β) • s.1.M j+c.β • (q j+O.value j ω))-average s.1.M = _
    rw [average_add, average_smul, average_smul, average_add]
    module
  change (∫ ω, ‖average (update c CU CP s (O.replies ω)).1.M-average s.1.M‖^2 ∂μ) ≤ _
  simp_rw [he]
  rw [affine_variance_identity _ _ (average_memLp O.memLp) hz,
    norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
  have ht := three_sq (average q-new) (new-old) (old-average s.1.M)
  rw [sub_add_sub_cancel, sub_add_sub_cancel, norm_sub_rev new old,
    norm_sub_rev old (average s.1.M)] at ht
  dsimp [globalError]
  have hab := mul_le_mul_of_nonneg_left (add_le_add ha hb)
    (by positivity : 0 ≤ 3*c.β^2)
  have hv' := mul_le_mul_of_nonneg_left hv (sq_nonneg c.β)
  simp only [div_eq_mul_inv] at hv' ⊢
  nlinarith only [mul_le_mul_of_nonneg_left ht (sq_nonneg c.β), hab, hv']

end EF21Muon.Nesterov.Averaged
