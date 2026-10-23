import EF21Muon.Stochastic.Tracking

noncomputable section

namespace EF21Muon.Stochastic

def Parameters.momentumWeight (c : Parameters) : ℝ :=
  6 * c.ρ ^ 2 / c.β + 72 * c.ρ ^ 2 * c.β / c.αU ^ 2

def Parameters.uplinkWeight (c : Parameters) : ℝ := 6 * c.ρ ^ 2 / c.αU

def Parameters.downlinkInjection (c : Parameters) : ℝ :=
  c.momentumWeight * c.β ^ 2 * (1 + 2 / c.β) * c.L ^ 2 +
    c.uplinkWeight * (6 * c.β ^ 2 * c.L ^ 2 / c.αU)

def Parameters.downlinkWeight (c : Parameters) : ℝ := c.downlinkInjection * (2 / c.αP - 1)

def Parameters.zetaWith (c : Parameters) (τ : ℝ) : ℝ :=
  c.momentumWeight * (2 * c.L ^ 2 / c.β) +
    c.uplinkWeight * (6 * c.β ^ 2 * c.L ^ 2 / c.αU) +
    (c.downlinkInjection + c.downlinkWeight) * (2 * τ / c.αP)

/-- Theorem 19 / Remark 23, with one worker and one matrix layer. -/
def Parameters.zeta (c : Parameters) : ℝ := c.zetaWith (c.ρ ^ 2)

def Parameters.noise (c : Parameters) : ℝ :=
  6 * c.ρ ^ 2 * c.β * (1 + (1 - c.αU) * c.β / c.αU + 12 * c.β ^ 2 / c.αU ^ 2)

theorem Parameters.weights_nonneg (c : Parameters) :
    0 ≤ c.momentumWeight ∧ 0 ≤ c.uplinkWeight ∧
      0 ≤ c.downlinkInjection ∧ 0 ≤ c.downlinkWeight := by
  have hβ := c.β_pos
  have hα := c.αU_pos
  have hP := c.αP_pos
  have hW : 0 ≤ c.momentumWeight := by unfold momentumWeight; positivity
  have hU : 0 ≤ c.uplinkWeight := by unfold uplinkWeight; positivity
  have hI : 0 ≤ c.downlinkInjection := by unfold downlinkInjection; positivity
  have hdiv : 1 ≤ 2 / c.αP := (le_div_iff₀ hP).mpr (by linarith [c.αP_le_one])
  exact ⟨hW, hU, hI, mul_nonneg hI (sub_nonneg.mpr hdiv)⟩

theorem Parameters.zeta_nonneg (c : Parameters) : 0 ≤ c.zeta := by
  rcases c.weights_nonneg with ⟨hW, hU, hI, hD⟩
  have hβ := c.β_pos
  have hα := c.αU_pos
  have hP := c.αP_pos
  unfold zeta zetaWith
  positivity

theorem Parameters.zeta_eq_paper (c : Parameters) :
    c.zeta = c.ρ ^ 2 * (12 * c.L ^ 2 / c.β ^ 2 +
      24 * c.ρ ^ 2 * (c.β + 2) * c.L ^ 2 / c.αP ^ 2 +
      36 * (c.β ^ 2 + 4) * c.L ^ 2 / c.αU ^ 2 +
      144 * c.ρ ^ 2 * c.β ^ 2 * (2 * c.β + 5) * c.L ^ 2 / (c.αP ^ 2 * c.αU ^ 2)) := by
  have hβ := ne_of_gt c.β_pos
  have hα := ne_of_gt c.αU_pos
  have hP := ne_of_gt c.αP_pos
  unfold zeta zetaWith downlinkWeight downlinkInjection momentumWeight uplinkWeight
  field_simp; ring

theorem Parameters.coefficients (c : Parameters) :
    3 * c.ρ ^ 2 + c.momentumWeight * (1 - c.β / 2) +
        c.uplinkWeight * (6 * c.β ^ 2 / c.αU) = c.momentumWeight ∧
      3 * c.ρ ^ 2 + c.uplinkWeight * (1 - c.αU / 2) = c.uplinkWeight ∧
      (c.downlinkInjection + c.downlinkWeight) * (1 - c.αP / 2) = c.downlinkWeight ∧
      (c.momentumWeight + c.uplinkWeight * (1 - c.αU)) * c.β ^ 2 = c.noise := by
  have hβ := ne_of_gt c.β_pos
  have hα := ne_of_gt c.αU_pos
  have hP := ne_of_gt c.αP_pos
  unfold momentumWeight uplinkWeight downlinkWeight downlinkInjection noise
  constructor
  · field_simp; ring
  constructor
  · field_simp; ring
  constructor
  · field_simp; ring
  · field_simp; ring

theorem Parameters.zetaWith_zero (c : Parameters) :
    c.zetaWith 0 = c.ρ ^ 2 * (12 * c.L ^ 2 / c.β ^ 2 +
      36 * (c.β ^ 2 + 4) * c.L ^ 2 / c.αU ^ 2) := by
  have hβ := ne_of_gt c.β_pos
  have hα := ne_of_gt c.αU_pos
  unfold zetaWith momentumWeight uplinkWeight
  field_simp; ring

theorem Parameters.zetaWith_zero_nonneg (c : Parameters) : 0 ≤ c.zetaWith 0 := by
  rw [c.zetaWith_zero]
  positivity

theorem Parameters.zetaWith_zero_le (c : Parameters) : c.zetaWith 0 ≤ c.zeta := by
  rcases c.weights_nonneg with ⟨hW, hU, hI, hD⟩
  have hP := c.αP_pos
  unfold zeta zetaWith
  have h := mul_nonneg (add_nonneg hI hD) (by positivity : 0 ≤ 2 * c.ρ ^ 2 / c.αP)
  simp only [mul_zero, zero_div, add_zero]
  nlinarith only [h]

/-- The spectral-compressor version printed in Theorem 19. -/
theorem Parameters.zetaWith_one_eq_paper (c : Parameters) :
    c.zetaWith 1 = c.ρ ^ 2 * (12 * c.L ^ 2 / c.β ^ 2 +
      24 * (c.β + 2) * c.L ^ 2 / c.αP ^ 2 +
      36 * (c.β ^ 2 + 4) * c.L ^ 2 / c.αU ^ 2 +
      144 * c.β ^ 2 * (2 * c.β + 5) * c.L ^ 2 / (c.αP ^ 2 * c.αU ^ 2)) := by
  have hβ := ne_of_gt c.β_pos
  have hα := ne_of_gt c.αU_pos
  have hP := ne_of_gt c.αP_pos
  unfold zetaWith downlinkWeight downlinkInjection momentumWeight uplinkWeight
  field_simp; ring

theorem Parameters.zetaWith_zero_le_one (c : Parameters) : c.zetaWith 0 ≤ c.zetaWith 1 := by
  rcases c.weights_nonneg with ⟨hW, hU, hI, hD⟩
  have hP := c.αP_pos
  unfold zetaWith
  have h := mul_nonneg (add_nonneg hI hD) (by positivity : 0 ≤ 2 / c.αP)
  simp only [mul_zero, zero_div, mul_one, add_zero]
  linarith only [h]

theorem Parameters.zetaWith_zero_lt_one (c : Parameters) (hρ : 0 < c.ρ) (hL : 0 < c.L) :
    c.zetaWith 0 < c.zetaWith 1 := by
  have hβ := c.β_pos
  have hU := c.αU_pos
  have hP := c.αP_pos
  have hW : 0 < c.momentumWeight := by unfold momentumWeight; positivity
  have hI : 0 < c.downlinkInjection := by
    have hC := c.weights_nonneg.2.1
    unfold downlinkInjection
    positivity
  have hD := c.weights_nonneg.2.2.2
  have h := mul_pos (add_pos_of_pos_of_nonneg hI hD) (by positivity : 0 < 2 / c.αP)
  unfold zetaWith
  simp only [mul_zero, zero_div, mul_one, add_zero]
  linarith only [h]

/-- The published square-root threshold implies the scalar descent condition. -/
theorem Parameters.step_condition_for (c : Parameters) {ζ : ℝ} (hζ : 0 ≤ ζ)
    (hγ : c.γ * (2 * c.L + 2 * Real.sqrt ζ) ≤ 1) :
    2 * c.L * c.γ + 4 * ζ * c.γ ^ 2 ≤ 1 := by
  have hs := Real.sq_sqrt hζ
  have hroot := Real.sqrt_nonneg ζ
  have hγ0 := c.γ_pos.le
  have hL := c.L_nonneg
  have hb : 2 * Real.sqrt ζ * c.γ ≤ 1 := by nlinarith only [hγ, mul_nonneg hL hγ0]
  have hprod := mul_nonneg (by positivity : 0 ≤ 2 * Real.sqrt ζ * c.γ) (sub_nonneg.mpr hb)
  nlinarith only [hγ, hprod, mul_nonneg (sq_nonneg c.γ) (sub_nonneg.mpr (le_of_eq hs)),
    mul_nonneg (sq_nonneg c.γ) (sub_nonneg.mpr (le_of_eq hs.symm))]

theorem Parameters.step_condition (c : Parameters)
    (hγ : c.γ * (2 * c.L + 2 * Real.sqrt c.zeta) ≤ 1) :
    2 * c.L * c.γ + 4 * c.zeta * c.γ ^ 2 ≤ 1 := c.step_condition_for c.zeta_nonneg hγ

end EF21Muon.Stochastic
