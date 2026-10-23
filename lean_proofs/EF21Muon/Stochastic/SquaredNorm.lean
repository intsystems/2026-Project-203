import EF21Muon.Geometry

noncomputable section

namespace EF21Muon.Stochastic

open scoped BigOperators
open LongTails.Algorithms.MatrixPolar

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem young_sq (x y : E) {t : ℝ} (ht : 0 < t) :
    ‖x + y‖ ^ 2 ≤ (1 + t) * ‖x‖ ^ 2 + (1 + 1 / t) * ‖y‖ ^ 2 := by
  have hi := real_inner_le_norm x y
  rw [norm_add_sq_real]
  apply (mul_le_mul_iff_right₀ ht).mp
  have he : t * (1 + 1 / t) = t + 1 := by field_simp
  nlinarith only [mul_le_mul_of_nonneg_left hi ht.le,
    sq_nonneg (t * ‖x‖ - ‖y‖), he]

theorem two_sq (x y : E) : ‖x + y‖ ^ 2 ≤ 2 * ‖x‖ ^ 2 + 2 * ‖y‖ ^ 2 := by
  simpa only [div_one, one_add_one_eq_two] using young_sq x y (t := 1) (by norm_num)

omit [InnerProductSpace ℝ E] in
theorem three_sq (x y z : E) :
    ‖x + y + z‖ ^ 2 ≤ 3 * (‖x‖ ^ 2 + ‖y‖ ^ 2 + ‖z‖ ^ 2) := by
  have h := norm_add_le (x + y) z
  have h' := norm_add_le x y
  have hsum : ‖x + y + z‖ ≤ ‖x‖ + ‖y‖ + ‖z‖ := by linarith
  have hs := mul_self_le_mul_self (norm_nonneg _) hsum
  nlinarith only [hs, sq_nonneg (‖x‖ - ‖y‖), sq_nonneg (‖x‖ - ‖z‖),
    sq_nonneg (‖y‖ - ‖z‖)]

theorem contraction_coefficients {a : ℝ} (ha : 0 < a) (_ha1 : a ≤ 1) :
    (1 - a) * (1 + a / 2) ≤ 1 - a / 2 ∧
      (1 - a) * (1 + 2 / a) ≤ 2 / a := by
  constructor
  · nlinarith only [sq_nonneg a]
  · apply (le_div_iff₀ ha).mpr
    calc
      (1 - a) * (1 + 2 / a) * a = (1 - a) * (a + 2) := by field_simp
      _ ≤ 2 := by nlinarith only [sq_nonneg a, ha]

theorem contracted_add_sq (x y : E) {a : ℝ} (ha : 0 < a) (ha1 : a ≤ 1) :
    (1 - a) * ‖x + y‖ ^ 2 ≤
      (1 - a / 2) * ‖x‖ ^ 2 + 2 / a * ‖y‖ ^ 2 := by
  have hy := mul_le_mul_of_nonneg_left (young_sq x y (t := a / 2) (by positivity))
    (sub_nonneg.mpr ha1)
  have hc := contraction_coefficients ha ha1
  have hdiv : 1 / (a / 2) = 2 / a := by ring
  rw [hdiv] at hy
  nlinarith only [hy, mul_le_mul_of_nonneg_right hc.1 (sq_nonneg ‖x‖),
    mul_le_mul_of_nonneg_right hc.2 (sq_nonneg ‖y‖)]

theorem momentum_mean_sq (A B C : E) {β : ℝ} (hβ : 0 < β) (hβ1 : β ≤ 1) :
    ‖(1 - β) • (A + B) + β • C‖ ^ 2 ≤
      (1 - β / 2) * ‖A‖ ^ 2 + 2 / β * ‖B‖ ^ 2 +
        β ^ 2 * (1 + 2 / β) * ‖C‖ ^ 2 := by
  have hy := young_sq ((1 - β) • (A + B)) (β • C) (t := β / 2) (by positivity)
  have hc := contraction_coefficients hβ hβ1
  have hdiv : 1 / (β / 2) = 2 / β := by ring
  simp only [norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, hdiv] at hy
  have h1 : (1 + β / 2) * (1 - β) ^ 2 ≤ 1 - β := by
    nlinarith only [mul_nonneg (sub_nonneg.mpr hβ1) (show 0 ≤ β + β ^ 2 by positivity)]
  have h2 := mul_le_mul_of_nonneg_right h1 (sq_nonneg ‖A + B‖)
  have h3 := contracted_add_sq A B hβ hβ1
  nlinarith only [hy, h2, h3]

/-- The spectral norm is at most the Frobenius norm, with constant one. -/
theorem spectralNorm_le_frobenius {m n : ℕ} (X : Space m n) : spectralNorm X ≤ ‖X‖ := by
  apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg X)
  intro v
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (norm_nonneg X) (norm_nonneg v))).mp
  rw [mul_pow, EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq,
    EuclideanSpace.real_norm_sq_eq, Fintype.sum_prod_type, Finset.sum_mul]
  apply Finset.sum_le_sum
  intro i _
  rw [operatorEquiv_apply_coordinate]
  exact Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun j => X (i, j)) (fun j => v j)

theorem frobenius_le_nuclear {m n : ℕ} (X : Space m n) : ‖X‖ ≤ nuclearNorm X := by
  have hi := inner_le_nuclear_mul_spectral X X
  rw [real_inner_self_eq_norm_sq, abs_of_nonneg (sq_nonneg ‖X‖)] at hi
  have hs := mul_le_mul_of_nonneg_left (spectralNorm_le_frobenius X) (nuclearNorm_nonneg X)
  by_cases hz : ‖X‖ = 0
  · simpa [hz] using nuclearNorm_nonneg X
  · have hp : 0 < ‖X‖ := lt_of_le_of_ne (norm_nonneg X) (Ne.symm hz)
    have hmul : ‖X‖ * ‖X‖ ≤ nuclearNorm X * ‖X‖ := by nlinarith only [hi, hs]
    exact (mul_le_mul_iff_left₀ hp).mp hmul

end EF21Muon.Stochastic
