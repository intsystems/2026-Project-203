import EF21Muon.Convergence

noncomputable section

namespace EF21Muon

open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {m n : ℕ}

def deterministicControls (m n : ℕ) (β L qU qP : ℝ)
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hL : 0 ≤ L)
    (hqU : 0 ≤ qU) (hqU1 : qU < 1) (hqP : 0 ≤ qP) (hqP1 : qP < 1) : Controls m n := by
  let δ := min β (min (1 - qU) (1 - qP)) / 2
  let B := 6 * β * L / (1 - qP)
  let C := 2 * L + qP * (B + 3 * β * L)
  let κ := geometryConstant m n
  let γ := δ / (4 * κ ^ 2 * (C + L + 1))
  have hδ : 0 < δ := by dsimp [δ]; positivity
  have hδβ : 2 * δ ≤ β := by dsimp [δ]; linarith [min_le_left β (min (1 - qU) (1 - qP))]
  have hδU : 2 * δ ≤ 1 - qU := by
    dsimp [δ]
    linarith [(min_le_right β (min (1 - qU) (1 - qP))).trans (min_le_left _ _)]
  have hδP : 2 * δ ≤ 1 - qP := by
    dsimp [δ]
    linarith [(min_le_right β (min (1 - qU) (1 - qP))).trans (min_le_right _ _)]
  have hδ1 : δ ≤ 1 := by linarith
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hκ : 0 < κ := geometryConstant_pos
  have hγ : 0 < γ := by dsimp [γ]; positivity
  have hBd : B * (1 - qP) = 6 * β * L := by
    dsimp [B]
    exact div_mul_cancel₀ _ (by linarith)
  have hPd : qP * (B + 3 * β * L) ≤ (1 - δ) * B := by
    nlinarith only [hBd, mul_nonneg hB (sub_nonneg.mpr hδP),
      mul_nonneg (by positivity : 0 ≤ 3 * β * L) (by linarith : 0 ≤ 1 - qP)]
  have hstep : 4 * κ ^ 2 * (C + L + 1) * γ = δ := by
    dsimp [γ]
    field_simp
  have hγL : 2 * L * κ ^ 2 * γ ≤ 1 := by
    nlinarith only [hstep, hδ1, mul_nonneg
      (by positivity : 0 ≤ 4 * κ ^ 2 * γ) (by linarith : 0 ≤ C + 1),
      mul_nonneg hL (by positivity : 0 ≤ κ ^ 2 * γ)]
  have hγC : 4 * κ ^ 2 * C * γ ≤ δ := by
    nlinarith only [hstep, mul_nonneg (by positivity : 0 ≤ 4 * κ ^ 2 * γ)
      (by linarith : 0 ≤ L + 1)]
  exact {
    β := β, γ := γ, L := L, qU := qU, qP := qP, B := B, δ := δ, C := C
    β_pos := hβ, β_le_one := hβ1, γ_pos := hγ, L_nonneg := hL
    qU_nonneg := hqU, qU_le_one := hqU1.le, qP_nonneg := hqP
    B_nonneg := hB, δ_pos := hδ, δ_le_one := hδ1, C_nonneg := hC
    lag_decay := hδβ, uplink_decay := by linarith
    downlink_decay := hPd, movement := le_rfl, smooth_step := hγL, tracking_step := hγC }

theorem muonUSign_stationarity {f : Space m n → ℝ} {grad : Space m n → Space m n}
    {fmin β L : ℝ} (hm : 0 < m) (hn : 0 < n)
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hL : 0 ≤ L)
    (hg : ∀ x, HasGradientAt f (grad x) x)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ L * ‖x - y‖)
    (hmin : ∀ x, fmin ≤ f x) (x₀ : Space m n) (T : ℕ) (hT : 0 < T) :
    let q := signContraction m n
    let δ := min β (min (1 - q) 1) / 2
    let κ := geometryConstant m n
    let γ := δ / (4 * κ ^ 2 * (3 * L + 1))
    (∑ t ∈ Finset.range T, nuclearNorm (grad (muonUSign β γ grad x₀ t).x) ^ 2) / T ≤
      8 * (f x₀ - fmin) / (γ * T) + 64 * κ ^ 2 * ‖grad x₀‖ ^ 2 / (δ * T) := by
  let c := deterministicControls m n β L (signContraction m n) 0 hβ hβ1 hL
    signContraction_nonneg (signContraction_lt_one hm hn) (by norm_num) (by norm_num)
  have h := average_stationarity c hg hlip hmin (signCompressor_contracts hm hn)
    identity_contracts x₀ T hT
  simpa only [c, deterministicControls, muonUSign, zero_mul, add_zero,
    sub_zero, show 2 * L + L + 1 = 3 * L + 1 by ring] using h

theorem muonSign_stationarity {f : Space m n → ℝ} {grad : Space m n → Space m n}
    {fmin β L : ℝ} (hm : 0 < m) (hn : 0 < n)
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hL : 0 ≤ L)
    (hg : ∀ x, HasGradientAt f (grad x) x)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ L * ‖x - y‖)
    (hmin : ∀ x, fmin ≤ f x) (x₀ : Space m n) (T : ℕ) (hT : 0 < T) :
    let q := signContraction m n
    let δ := min β (min (1 - q) (1 - q)) / 2
    let B := 6 * β * L / (1 - q)
    let C := 2 * L + q * (B + 3 * β * L)
    let κ := geometryConstant m n
    let γ := δ / (4 * κ ^ 2 * (C + L + 1))
    (∑ t ∈ Finset.range T, nuclearNorm (grad (muonSign β γ grad x₀ t).x) ^ 2) / T ≤
      8 * (f x₀ - fmin) / (γ * T) + 64 * κ ^ 2 * ‖grad x₀‖ ^ 2 / (δ * T) := by
  let c := deterministicControls m n β L (signContraction m n) (signContraction m n)
    hβ hβ1 hL signContraction_nonneg (signContraction_lt_one hm hn)
    signContraction_nonneg (signContraction_lt_one hm hn)
  exact average_stationarity c hg hlip hmin (signCompressor_contracts hm hn)
    (signCompressor_contracts hm hn) x₀ T hT

end EF21Muon
