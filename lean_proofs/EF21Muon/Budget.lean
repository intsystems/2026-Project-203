import EF21Muon.Tuning

noncomputable section

namespace EF21Muon

open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {m n : ℕ}

def iterationBudget (c : Controls m n) (Δ G ε : ℝ) : ℕ :=
  max 1 ⌈(8 * Δ / c.γ + 64 * geometryConstant m n ^ 2 * G ^ 2 / c.δ) / ε ^ 2⌉₊

theorem stationarity_at_budget (c : Controls m n) {f : Space m n → ℝ}
    {grad CU CP : Space m n → Space m n} {fmin Δ G ε : ℝ}
    (hg : ∀ x, HasGradientAt f (grad x) x)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ c.L * ‖x - y‖)
    (hmin : ∀ x, fmin ≤ f x) (hCU : Contracts CU c.qU) (hCP : Contracts CP c.qP)
    (x₀ : Space m n) (hΔ : f x₀ - fmin ≤ Δ) (hG : ‖grad x₀‖ ≤ G) (hε : 0 < ε) :
    let T := iterationBudget c Δ G ε
    (∑ t ∈ Finset.range T, nuclearNorm (grad (trajectory c.β c.γ grad CU CP x₀ t).x) ^ 2) / T ≤
      ε ^ 2 := by
  let T := iterationBudget c Δ G ε
  let R := 8 * Δ / c.γ + 64 * geometryConstant m n ^ 2 * G ^ 2 / c.δ
  have hT : 0 < T := lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  have hT' : (0 : ℝ) < T := by exact_mod_cast hT
  have hs := average_stationarity c hg hlip hmin hCU hCP x₀ T hT
  have h1 := div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_left hΔ (by norm_num : (0 : ℝ) ≤ 8))
    (mul_pos c.γ_pos hT').le
  have hGsq : ‖grad x₀‖ ^ 2 ≤ G ^ 2 := by
    nlinarith only [mul_self_le_mul_self (norm_nonneg _) hG]
  have h2 := div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_left hGsq (by positivity : 0 ≤ 64 * geometryConstant m n ^ 2))
    (mul_pos c.δ_pos hT').le
  have hsum : 8 * Δ / (c.γ * T) + 64 * geometryConstant m n ^ 2 * G ^ 2 / (c.δ * T) =
      R / T := by dsimp [R]; field_simp
  have hceil : R / ε ^ 2 ≤ (T : ℝ) := by
    apply (Nat.le_ceil (R / ε ^ 2)).trans
    have hceilT : ⌈R / ε ^ 2⌉₊ ≤ T := le_max_right _ _
    exact_mod_cast hceilT
  have hR : R / T ≤ ε ^ 2 := by
    apply (div_le_iff₀ hT').mpr
    have h := (div_le_iff₀ (sq_pos_of_pos hε)).mp hceil
    nlinarith only [h]
  exact (hs.trans (add_le_add h1 h2)).trans (hsum.le.trans hR)

end EF21Muon
