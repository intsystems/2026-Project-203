import EF21Muon.Stochastic.Constants

noncomputable section

namespace EF21Muon.Distributed

open EF21Muon.Stochastic

def globalWeight (c : Parameters) : ℝ := 6 * c.ρ ^ 2 / c.β
def localWeight (c : Parameters) : ℝ := 72 * c.ρ ^ 2 * c.β / c.αU ^ 2

/-- `ell2` is the mean of the squared local smoothness constants. -/
def injection (c : Parameters) (ell2 : ℝ) : ℝ :=
  globalWeight c * c.β ^ 2 * (1 + 2 / c.β) * c.L ^ 2 +
    localWeight c * c.β ^ 2 * (1 + 2 / c.β) * ell2 +
      c.uplinkWeight * (6 * c.β ^ 2 * ell2 / c.αU)

def broadcastWeight (c : Parameters) (ell2 : ℝ) : ℝ := injection c ell2 * (2 / c.αP - 1)

def zeta (c : Parameters) (ell2 τ : ℝ) : ℝ :=
  globalWeight c * (2 * c.L ^ 2 / c.β) + localWeight c * (2 * ell2 / c.β) +
    c.uplinkWeight * (6 * c.β ^ 2 * ell2 / c.αU) +
      (injection c ell2 + broadcastWeight c ell2) * (2 * τ / c.αP)

def noise (c : Parameters) (N : ℕ) : ℝ :=
  6 * c.ρ ^ 2 * c.β * ((N : ℝ)⁻¹ + (1 - c.αU) * c.β / c.αU +
    12 * c.β ^ 2 / c.αU ^ 2)

theorem weights_nonneg (c : Parameters) {ell2 : ℝ} (hell : 0 ≤ ell2) :
    0 ≤ globalWeight c ∧ 0 ≤ localWeight c ∧ 0 ≤ c.uplinkWeight ∧
      0 ≤ injection c ell2 ∧ 0 ≤ broadcastWeight c ell2 := by
  have hβ := c.β_pos
  have hU := c.αU_pos
  have hP := c.αP_pos
  have hA : 0 ≤ globalWeight c := by unfold globalWeight; positivity
  have hB : 0 ≤ localWeight c := by unfold localWeight; positivity
  have hC := c.weights_nonneg.2.1
  have hI : 0 ≤ injection c ell2 := by unfold injection; positivity
  have hdiv : 1 ≤ 2 / c.αP := (le_div_iff₀ hP).mpr (by linarith [c.αP_le_one])
  exact ⟨hA, hB, hC, hI, mul_nonneg hI (sub_nonneg.mpr hdiv)⟩

theorem coefficients (c : Parameters) (N : ℕ) (ell2 : ℝ) :
    3 * c.ρ ^ 2 + globalWeight c * (1 - c.β / 2) = globalWeight c ∧
    localWeight c * (1 - c.β / 2) + c.uplinkWeight * (6 * c.β ^ 2 / c.αU) = localWeight c ∧
    3 * c.ρ ^ 2 + c.uplinkWeight * (1 - c.αU / 2) = c.uplinkWeight ∧
    (injection c ell2 + broadcastWeight c ell2) * (1 - c.αP / 2) = broadcastWeight c ell2 ∧
    (globalWeight c / N + localWeight c + c.uplinkWeight * (1 - c.αU)) * c.β ^ 2 = noise c N := by
  have hβ := ne_of_gt c.β_pos
  have hU := ne_of_gt c.αU_pos
  have hP := ne_of_gt c.αP_pos
  unfold globalWeight localWeight Parameters.uplinkWeight broadcastWeight injection noise
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> field_simp <;> ring

/-- The full distributed coefficient in Theorem 19, with the downlink norm factor explicit. -/
theorem zeta_eq_paper (c : Parameters) (ell2 τ : ℝ) :
    zeta c ell2 τ = c.ρ ^ 2 * (12 * c.L ^ 2 / c.β ^ 2 +
      24 * τ * (c.β + 2) * c.L ^ 2 / c.αP ^ 2 +
      36 * (c.β ^ 2 + 4) * ell2 / c.αU ^ 2 +
      144 * τ * c.β ^ 2 * (2 * c.β + 5) * ell2 / (c.αP ^ 2 * c.αU ^ 2)) := by
  have hβ := ne_of_gt c.β_pos
  have hU := ne_of_gt c.αU_pos
  have hP := ne_of_gt c.αP_pos
  unfold zeta broadcastWeight injection globalWeight localWeight Parameters.uplinkWeight
  field_simp; ring

theorem exact_downlink_le (c : Parameters) {ell2 τ : ℝ} (hell : 0 ≤ ell2) (hτ : 0 ≤ τ) :
    zeta c ell2 0 ≤ zeta c ell2 τ := by
  rcases weights_nonneg c hell with ⟨_, _, _, hI, hD⟩
  have hP := c.αP_pos
  have h := mul_nonneg (add_nonneg hI hD) (by positivity : 0 ≤ 2 * τ / c.αP)
  unfold zeta
  simp only [mul_zero, zero_div, add_zero]
  linarith only [h]

theorem zeta_nonneg (c : Parameters) {ell2 τ : ℝ} (hell : 0 ≤ ell2) (hτ : 0 ≤ τ) :
    0 ≤ zeta c ell2 τ := by
  rcases weights_nonneg c hell with ⟨hA, hB, hC, hI, hD⟩
  have hβ := c.β_pos
  have hU := c.αU_pos
  have hP := c.αP_pos
  unfold zeta
  positivity

end EF21Muon.Distributed
