import EF21Muon.Divergence.Smooth
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

/-! Smooth bounded periodic potentials with prescribed opposite derivatives
at two residues. This analytic realization replaces the article's plateau ramps. -/

noncomputable section

namespace EF21Muon.Divergence

open scoped ContDiff

def wave (freq center θ x : ℝ) : ℝ := -Real.cos (freq*(x-center))/(freq*Real.sin θ)
def waveDerivative (freq center θ x : ℝ) : ℝ := Real.sin (freq*(x-center))/Real.sin θ

theorem wave_hasDeriv {freq center θ : ℝ} (hfreq : freq ≠ 0) (hθ : Real.sin θ ≠ 0) (x : ℝ) :
    HasDerivAt (wave freq center θ) (waveDerivative freq center θ x) x := by
  have h := ((((hasDerivAt_id x).sub_const center).const_mul freq).cos.neg).div_const
    (freq*Real.sin θ)
  convert h using 1
  · rfl
  · dsimp [waveDerivative]
    field_simp

theorem wave_contDiff (freq center θ : ℝ) : ContDiff ℝ ∞ (wave freq center θ) := by
  unfold wave
  fun_prop

theorem wave_periodic {freq p : ℝ} (center θ : ℝ) (h : freq*p = 2*Real.pi) :
    Function.Periodic (wave freq center θ) p := by
  intro x
  have ha : freq*(x+p-center) = freq*(x-center)+2*Real.pi := by nlinarith only [h]
  simp only [wave, ha, Real.cos_add_two_pi]

theorem waveDerivative_periodic {freq p : ℝ} (center θ : ℝ) (h : freq*p = 2*Real.pi) :
    Function.Periodic (waveDerivative freq center θ) p := by
  intro x
  have ha : freq*(x+p-center) = freq*(x-center)+2*Real.pi := by nlinarith only [h]
  simp only [waveDerivative, ha, Real.sin_add_two_pi]

theorem waveDerivative_at_plus {freq center θ x : ℝ} (hθ : Real.sin θ ≠ 0)
    (hx : freq*(x-center) = θ) : waveDerivative freq center θ x = 1 := by
  simp [waveDerivative, hx, hθ]

theorem waveDerivative_at_minus {freq center θ x : ℝ} (hθ : Real.sin θ ≠ 0)
    (hx : freq*(x-center) = -θ) : waveDerivative freq center θ x = -1 := by
  simp [waveDerivative, hx, hθ]

theorem wave_bounded (freq center θ x : ℝ) :
    |wave freq center θ x| ≤ 1 / |freq*Real.sin θ| := by
  simp only [wave, abs_div, abs_neg]
  exact div_le_div_of_nonneg_right (Real.abs_cos_le_one _) (abs_nonneg _)

theorem waveDerivative_bounded (freq center θ x : ℝ) :
    |waveDerivative freq center θ x| ≤ 1 / |Real.sin θ| := by
  simp only [waveDerivative, abs_div]
  exact div_le_div_of_nonneg_right (Real.abs_sin_le_one _) (abs_nonneg _)

theorem waveDerivative_lipschitz (freq center θ : ℝ) :
    ∃ K, LipschitzWith K (waveDerivative freq center θ) := by
  refine ⟨⟨|freq|/|Real.sin θ|, by positivity⟩, (lipschitzWith_iff_norm_sub_le).mpr ?_⟩
  intro x y
  change |waveDerivative freq center θ x - waveDerivative freq center θ y| ≤ _
  simp only [waveDerivative, ← sub_div, abs_div, Real.norm_eq_abs]
  calc
    _ ≤ |freq*(x-center)-freq*(y-center)|/|Real.sin θ| :=
      div_le_div_of_nonneg_right (Real.abs_sin_sub_sin_le _ _) (abs_nonneg _)
    _ = (|freq|/|Real.sin θ|)*|x-y| := by
      rw [show freq*(x-center)-freq*(y-center) = freq*(x-y) by ring, abs_mul]
      ring

end EF21Muon.Divergence
