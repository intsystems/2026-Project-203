import EF21Muon.Distributed.Rates

/-! The step-condition algebra behind the generalized-smooth corollary.
This verifies the three exponent inequalities in the upstream proof, including
the relaxed second condition. It does not prove generalized convergence. -/

noncomputable section

namespace EF21Muon.Distributed

/-- eta / S^(3/4), written using positive square roots. -/
def generalizedRadius (S η : ℝ) : ℝ := η / (Real.sqrt S * Real.sqrt (Real.sqrt S))

theorem generalizedRadius_sq {S : ℝ} (hS : 0 ≤ S) (η : ℝ) :
    generalizedRadius S η ^ 2 = η ^ 2 / (S * Real.sqrt S) := by
  unfold generalizedRadius
  rw [div_pow, mul_pow, Real.sq_sqrt hS, Real.sq_sqrt (Real.sqrt_nonneg S)]

/-- With beta = 1/sqrt(S), these conditions bound each of the three
contributions to S*(C1+C2/beta) by 1/3. In particular the second condition
has sqrt(S), not 1/sqrt(S), on its right-hand side. -/
theorem generalized_exponent_conditions {S η ρ L₁ M q : ℝ}
    (hS : 0 < S) (hq : q < 1)
    (hglobal : 6 * η ^ 2 * L₁ ^ 2 ≤ Real.sqrt S)
    (hcompression : 24 * q * η ^ 2 * ρ * M ^ 2 ≤ (1 - q) * Real.sqrt S)
    (hlocal : 24 * η ^ 2 * ρ * M ^ 2 ≤ 1) :
    2 * S * generalizedRadius S η ^ 2 * L₁ ^ 2 ≤ 1 / 3 ∧
    S * (8 * q / (1 - q)) * generalizedRadius S η ^ 2 * ρ * M ^ 2 ≤ 1 / 3 ∧
    S * (8 / (1 / Real.sqrt S)) * generalizedRadius S η ^ 2 * ρ * M ^ 2 ≤ 1 / 3 := by
  have hs := Real.sqrt_pos.mpr hS
  have hq' : 0 < 1 - q := sub_pos.mpr hq
  rw [generalizedRadius_sq hS.le]
  constructor
  · calc
      _ = 2 * η ^ 2 * L₁ ^ 2 / Real.sqrt S := by field_simp
      _ ≤ 1 / 3 := (div_le_iff₀ hs).mpr (by linarith only [hglobal])
  constructor
  · calc
      _ = 8 * q * η ^ 2 * ρ * M ^ 2 / ((1 - q) * Real.sqrt S) := by field_simp
      _ ≤ 1 / 3 := (div_le_iff₀ (mul_pos hq' hs)).mpr (by linarith only [hcompression])
  · calc
      _ = 8 * η ^ 2 * ρ * M ^ 2 := by field_simp
      _ ≤ 1 / 3 := by linarith only [hlocal]

end EF21Muon.Distributed
