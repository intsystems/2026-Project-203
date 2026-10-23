import Mathlib.Tactic

namespace EF21Muon

theorem tracking_square {δ C z z' h : ℝ} (hδ : 0 ≤ δ) (hδ1 : δ ≤ 1)
    (_hC : 0 ≤ C) (_hz : 0 ≤ z) (hz' : 0 ≤ z') (_hh : 0 ≤ h)
    (ht : z' ≤ (1 - δ) * z + C * h) :
    δ * z' ^ 2 ≤ δ * (1 - δ) * z ^ 2 + C ^ 2 * h ^ 2 := by
  have hs := mul_self_le_mul_self hz' ht
  nlinarith [mul_nonneg hδ (sub_nonneg.mpr hs),
    mul_nonneg (by linarith : 0 ≤ 1 - δ) (sq_nonneg (δ * z - C * h))]

theorem lyapunov_step {δ γ κ C a z z' h f f' : ℝ}
    (hδ : 0 < δ) (hγ : 0 ≤ γ) (_hκ : 0 ≤ κ) (hC : 0 ≤ C)
    (_ha : 0 ≤ a) (hh : 0 ≤ h)
    (hsmall : 4 * κ ^ 2 * C * γ ≤ δ)
    (hmove : h ≤ κ * γ * a)
    (htrack : δ * z' ^ 2 ≤ δ * (1 - δ) * z ^ 2 + C ^ 2 * h ^ 2)
    (hdescent : f' ≤ f - γ / 2 * a ^ 2 + γ * κ ^ 2 * z ^ 2) :
    δ ^ 2 * f' + 2 * γ * κ ^ 2 * δ * z' ^ 2 ≤
      δ ^ 2 * f + 2 * γ * κ ^ 2 * δ * z ^ 2 -
        δ ^ 2 * γ / 4 * a ^ 2 - δ ^ 2 * γ * κ ^ 2 * z ^ 2 := by
  have hs := mul_self_le_mul_self (by positivity : 0 ≤ 4 * κ ^ 2 * C * γ) hsmall
  have hm := mul_self_le_mul_self hh hmove
  have hterm : 2 * κ ^ 2 * C ^ 2 * h ^ 2 ≤ δ ^ 2 / 4 * a ^ 2 := by
    nlinarith [mul_nonneg (by positivity : 0 ≤ 2 * κ ^ 2 * C ^ 2) (sub_nonneg.mpr hm),
      mul_nonneg (sq_nonneg a) (sub_nonneg.mpr hs)]
  nlinarith [mul_nonneg (by positivity : 0 ≤ δ ^ 2) (sub_nonneg.mpr hdescent),
    mul_nonneg (by positivity : 0 ≤ 2 * γ * κ ^ 2) (sub_nonneg.mpr htrack),
    mul_nonneg hγ (sub_nonneg.mpr hterm)]

end EF21Muon
