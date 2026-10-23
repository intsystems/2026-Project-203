import EF21Muon.Algorithm
import Mathlib.Tactic

noncomputable section

namespace EF21Muon

open LongTails.Algorithms.MatrixPolar

variable {m n : ℕ}

theorem downlink_step {β γ q : ℝ} {grad CU CP : Space m n → Space m n}
    (hq : 0 ≤ q) (hC : Contracts CP q) (s : State m n) :
    downlinkError (step β γ grad CU CP s) ≤
      q * (downlinkError s + ‖(step β γ grad CU CP s).x - s.x‖) := by
  let v := (step β γ grad CU CP s).x
  have he : (step β γ grad CU CP s).w - v = CP (v - s.w) - (v - s.w) := by
    dsimp [step, v]
    module
  change ‖(step β γ grad CU CP s).w - v‖ ≤ _
  rw [he]
  apply (hC _).trans
  apply mul_le_mul_of_nonneg_left _ hq
  calc
    ‖v - s.w‖ = ‖s.w - v‖ := norm_sub_rev _ _
    _ ≤ ‖s.w - s.x‖ + ‖v - s.x‖ := by
      simpa only [norm_sub_rev s.x v] using norm_sub_le_norm_sub_add_norm_sub s.w s.x v

theorem lag_step {β γ L : ℝ} {grad CU CP : Space m n → Space m n}
    (hβ : 0 ≤ β) (hβ1 : β ≤ 1)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ L * ‖x - y‖) (s : State m n) :
    lag grad (step β γ grad CU CP s) ≤
      (1 - β) * lag grad s + (1 - β) * L * ‖(step β γ grad CU CP s).x - s.x‖ +
        β * L * downlinkError (step β γ grad CU CP s) := by
  let t := step β γ grad CU CP s
  have he : t.momentum - grad t.x =
      (1 - β) • (s.momentum - grad s.x) +
      (1 - β) • (grad s.x - grad t.x) + β • (grad t.w - grad t.x) := by
    dsimp [t, step]
    module
  have h := (norm_add_le ((1 - β) • (s.momentum - grad s.x) +
      (1 - β) • (grad s.x - grad t.x)) (β • (grad t.w - grad t.x))).trans
    (add_le_add (norm_add_le _ _) le_rfl)
  rw [← he] at h
  simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg hβ,
    abs_of_nonneg (sub_nonneg.mpr hβ1)] at h
  have hx := hlip s.x t.x
  rw [norm_sub_rev s.x t.x] at hx
  have hw := hlip t.w t.x
  change ‖t.momentum - grad t.x‖ ≤ (1 - β) * ‖s.momentum - grad s.x‖ +
    (1 - β) * L * ‖t.x - s.x‖ + β * L * ‖t.w - t.x‖
  nlinarith only [h, mul_le_mul_of_nonneg_left hx (sub_nonneg.mpr hβ1),
    mul_le_mul_of_nonneg_left hw hβ]

theorem momentum_change {β γ L : ℝ} {grad CU CP : Space m n → Space m n}
    (hβ : 0 ≤ β)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ L * ‖x - y‖) (s : State m n) :
    ‖(step β γ grad CU CP s).momentum - s.momentum‖ ≤
      β * (lag grad s + L * ‖(step β γ grad CU CP s).x - s.x‖ +
        L * downlinkError (step β γ grad CU CP s)) := by
  let t := step β γ grad CU CP s
  have he : t.momentum - s.momentum = β • (grad t.w - s.momentum) := by
    dsimp [t, step]
    module
  rw [he, norm_smul, Real.norm_eq_abs, abs_of_nonneg hβ]
  apply mul_le_mul_of_nonneg_left _ hβ
  calc
    ‖grad t.w - s.momentum‖ ≤ ‖grad t.w - grad t.x‖ +
        (‖grad t.x - grad s.x‖ + ‖grad s.x - s.momentum‖) :=
      (norm_sub_le_norm_sub_add_norm_sub (grad t.w) (grad t.x) s.momentum).trans
        (add_le_add le_rfl (norm_sub_le_norm_sub_add_norm_sub _ _ _))
    _ ≤ L * ‖t.w - t.x‖ + (L * ‖t.x - s.x‖ + ‖grad s.x - s.momentum‖) := by
      gcongr
      · exact hlip _ _
      · exact hlip _ _
    _ = _ := by simp only [lag, downlinkError, norm_sub_rev (grad s.x) s.momentum]; ring

theorem uplink_step {β γ q : ℝ} {grad CU CP : Space m n → Space m n}
    (hq : 0 ≤ q) (hC : Contracts CU q) (s : State m n) :
    uplinkError (step β γ grad CU CP s) ≤
      q * (uplinkError s + ‖(step β γ grad CU CP s).momentum - s.momentum‖) := by
  let t := step β γ grad CU CP s
  have he : t.estimator - t.momentum =
      CU (t.momentum - s.estimator) - (t.momentum - s.estimator) := by
    dsimp [t, step]
    module
  change ‖t.estimator - t.momentum‖ ≤ _
  rw [he]
  apply (hC _).trans
  apply mul_le_mul_of_nonneg_left _ hq
  calc
    ‖t.momentum - s.estimator‖ = ‖s.estimator - t.momentum‖ := norm_sub_rev _ _
    _ ≤ ‖s.estimator - s.momentum‖ + ‖t.momentum - s.momentum‖ := by
      simpa only [norm_sub_rev s.momentum t.momentum] using
        norm_sub_le_norm_sub_add_norm_sub s.estimator s.momentum t.momentum

set_option maxHeartbeats 800000 in
theorem tracking_step {β γ L qU qP B δ C : ℝ}
    {grad CU CP : Space m n → Space m n}
    (hβ : 0 ≤ β) (hβ1 : β ≤ 1) (hL : 0 ≤ L)
    (hqU : 0 ≤ qU) (hqU1 : qU ≤ 1) (hqP : 0 ≤ qP) (hB : 0 ≤ B)
    (hδβ : 2 * δ ≤ β) (hδU : qU ≤ 1 - δ)
    (hδP : qP * (B + 3 * β * L) ≤ (1 - δ) * B)
    (hC : 2 * L + qP * (B + 3 * β * L) ≤ C)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ L * ‖x - y‖)
    (hCU : Contracts CU qU) (hCP : Contracts CP qP) (s : State m n) :
    trackingError B grad (step β γ grad CU CP s) ≤
      (1 - δ) * trackingError B grad s + C * ‖(step β γ grad CU CP s).x - s.x‖ := by
  let t := step β γ grad CU CP s
  have ha := lag_step hβ hβ1 hlip s (γ := γ) (CU := CU) (CP := CP)
  have hb := (uplink_step hqU hCU s (β := β) (γ := γ) (grad := grad) (CP := CP)).trans
    (mul_le_mul_of_nonneg_left (add_le_add le_rfl (momentum_change hβ hlip s)) hqU)
  have hp := downlink_step hqP hCP s (β := β) (γ := γ) (grad := grad) (CU := CU)
  have hqa : qU * β ≤ β := mul_le_of_le_one_left hβ hqU1
  have hqaL : qU * β * L ≤ β * L := mul_le_mul_of_nonneg_right hqa hL
  have ha0 : 0 ≤ lag grad s := norm_nonneg _
  have hb0 : 0 ≤ uplinkError s := norm_nonneg _
  have hp0 : 0 ≤ downlinkError s := norm_nonneg _
  have hd0 := norm_nonneg (t.x - s.x)
  have hp1 : 0 ≤ downlinkError t := norm_nonneg _
  change 2 * lag grad t + uplinkError t + B * downlinkError t ≤ _
  have h1 : 2 * lag grad t + uplinkError t + B * downlinkError t ≤
      (2 - β) * lag grad s + qU * uplinkError s + 2 * L * ‖t.x - s.x‖ +
        (B + 3 * β * L) * downlinkError t := by
    nlinarith only [ha, hb, mul_nonneg (sub_nonneg.mpr hqa) ha0,
      mul_nonneg (sub_nonneg.mpr hqaL) hd0,
      mul_nonneg (sub_nonneg.mpr hqaL) hp1,
      mul_nonneg (mul_nonneg hβ hL) hd0]
  apply h1.trans
  calc
    _ ≤ (2 - β) * lag grad s + qU * uplinkError s + 2 * L * ‖t.x - s.x‖ +
        (B + 3 * β * L) * (qP * (downlinkError s + ‖t.x - s.x‖)) := by
      gcongr
    _ ≤ (1 - δ) * trackingError B grad s + C * ‖t.x - s.x‖ := by
      unfold trackingError
      nlinarith only [mul_nonneg (sub_nonneg.mpr hδU) hb0,
        mul_nonneg (sub_nonneg.mpr hδP) hp0,
        mul_nonneg (sub_nonneg.mpr hC) hd0,
        mul_nonneg (by linarith : 0 ≤ β - 2 * δ) ha0]

end EF21Muon
