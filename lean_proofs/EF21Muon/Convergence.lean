import EF21Muon.Tracking
import EF21Muon.Geometry
import EF21Muon.Scalar
import EF21Muon.Controls
import LongTails.Proofs.GdConvex

noncomputable section

namespace EF21Muon

open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {m n : ℕ}

theorem step_movement {β γ : ℝ} {grad CU CP : Space m n → Space m n}
    (hγ : 0 ≤ γ) (s : State m n) :
    ‖(step β γ grad CU CP s).x - s.x‖ ≤
      geometryConstant m n * γ * nuclearNorm s.estimator := by
  have he : (step β γ grad CU CP s).x - s.x =
      -(γ * nuclearNorm s.estimator) • polar s.estimator := by dsimp [step]; module
  rw [he, norm_smul, Real.norm_eq_abs, abs_neg,
    abs_of_nonneg (mul_nonneg hγ (nuclearNorm_nonneg _))]
  have h := mul_le_mul_of_nonneg_left (norm_polar_le_geometry s.estimator)
    (mul_nonneg hγ (nuclearNorm_nonneg s.estimator))
  nlinarith only [h]

theorem sharp_descent (c : Controls m n) {f : Space m n → ℝ}
    {grad CU CP : Space m n → Space m n}
    (hg : ∀ x, HasGradientAt f (grad x) x)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ c.L * ‖x - y‖) (s : State m n) :
    f (step c.β c.γ grad CU CP s).x ≤ f s.x - c.γ / 2 * nuclearNorm s.estimator ^ 2 +
      c.γ * geometryConstant m n ^ 2 * trackingError c.B grad s ^ 2 := by
  let t := step c.β c.γ grad CU CP s
  let a := nuclearNorm s.estimator
  let κ := geometryConstant m n
  let z := trackingError c.B grad s
  have ha : 0 ≤ a := nuclearNorm_nonneg _
  have hz : 0 ≤ z := trackingError_nonneg c.B_nonneg _ _
  have hκ : 0 ≤ κ := geometryConstant_pos.le
  have hpair := real_inner_le_norm (s.estimator - grad s.x) (polar s.estimator)
  rw [inner_sub_left, polar_pairing_eq_nuclear] at hpair
  have hnorm : ‖s.estimator - grad s.x‖ * ‖polar s.estimator‖ ≤ z * κ :=
    mul_le_mul (estimator_error_le_tracking c.B_nonneg grad s)
      (norm_polar_le_geometry _) (norm_nonneg _) hz
  have hp : a - κ * z ≤ inner ℝ (grad s.x) (polar s.estimator) := by
    dsimp [a, κ, z] at *
    nlinarith only [hpair, hnorm]
  have hm := step_movement c.γ_pos.le s (β := c.β) (grad := grad) (CU := CU) (CP := CP)
  have hms := mul_self_le_mul_self (norm_nonneg _) hm
  have hquad : c.L / 2 * ‖t.x - s.x‖ ^ 2 ≤ c.γ / 4 * a ^ 2 := by
    have h1 := mul_nonneg c.L_nonneg (sub_nonneg.mpr hms)
    have h2 := mul_nonneg (mul_nonneg c.γ_pos.le (sq_nonneg a))
      (sub_nonneg.mpr c.smooth_step)
    change _ ≤ _
    dsimp only [t, a, κ] at *
    nlinarith only [h1, h2]
  have hs := OracleTemplate.Gradient.smooth_upper_bound hg hlip s.x t.x
  have he : t.x - s.x = -(c.γ * a) • polar s.estimator := by dsimp [t, step, a]; module
  rw [he, inner_smul_right] at hs
  have hq : c.γ * a * (a - κ * z) ≤
      c.γ * a * inner ℝ (grad s.x) (polar s.estimator) :=
    mul_le_mul_of_nonneg_left hp (mul_nonneg c.γ_pos.le ha)
  have hy := mul_nonneg c.γ_pos.le (sq_nonneg (a / 2 - κ * z))
  rw [← he] at hs
  change f t.x ≤ f s.x - c.γ / 2 * a ^ 2 + c.γ * κ ^ 2 * z ^ 2
  nlinarith only [hs, hquad, hq, hy]

def potential (c : Controls m n) (f : Space m n → ℝ) (grad : Space m n → Space m n)
    (fmin : ℝ) (s : State m n) : ℝ :=
  c.δ ^ 2 * (f s.x - fmin) +
    2 * c.γ * geometryConstant m n ^ 2 * c.δ * trackingError c.B grad s ^ 2

theorem potential_nonneg (c : Controls m n) {f : Space m n → ℝ}
    (grad : Space m n → Space m n) {fmin : ℝ} (hmin : ∀ x, fmin ≤ f x)
    (s : State m n) : 0 ≤ potential c f grad fmin s := by
  have hγ := c.γ_pos
  have hδ := c.δ_pos
  have hf := sub_nonneg.mpr (hmin s.x)
  unfold potential
  positivity

set_option maxHeartbeats 800000 in
theorem potential_descent (c : Controls m n) {f : Space m n → ℝ}
    {grad CU CP : Space m n → Space m n}
    (hg : ∀ x, HasGradientAt f (grad x) x)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ c.L * ‖x - y‖)
    (hCU : Contracts CU c.qU) (hCP : Contracts CP c.qP) (fmin : ℝ) (s : State m n) :
    potential c f grad fmin (step c.β c.γ grad CU CP s) +
      c.δ ^ 2 * c.γ / 8 * nuclearNorm (grad s.x) ^ 2 ≤ potential c f grad fmin s := by
  let t := step c.β c.γ grad CU CP s
  have hγ := c.γ_pos
  have ht := tracking_step c.β_pos.le c.β_le_one c.L_nonneg c.qU_nonneg c.qU_le_one
    c.qP_nonneg c.B_nonneg c.lag_decay c.uplink_decay c.downlink_decay c.movement
    hlip hCU hCP s (γ := c.γ)
  have ht2 := tracking_square c.δ_pos.le c.δ_le_one c.C_nonneg
    (trackingError_nonneg c.B_nonneg grad s) (trackingError_nonneg c.B_nonneg grad t)
    (norm_nonneg _) ht
  have hy := lyapunov_step c.δ_pos c.γ_pos.le geometryConstant_pos.le c.C_nonneg
    (nuclearNorm_nonneg _) (norm_nonneg _) c.tracking_step (step_movement c.γ_pos.le s) ht2
    (sharp_descent c hg hlip s (CU := CU) (CP := CP))
  have hg2 := gradient_sq_le_tracking c.B_nonneg grad s
  have hg3 := mul_nonneg (by positivity : 0 ≤ c.δ ^ 2 * c.γ / 8)
    (sub_nonneg.mpr hg2)
  unfold potential
  nlinarith only [hy, hg3, mul_nonneg (by positivity : 0 ≤ c.δ ^ 2 * c.γ)
    (by positivity : 0 ≤ geometryConstant m n ^ 2 * trackingError c.B grad s ^ 2)]

theorem finite_time (c : Controls m n) {f : Space m n → ℝ}
    {grad CU CP : Space m n → Space m n} {fmin : ℝ}
    (hg : ∀ x, HasGradientAt f (grad x) x)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ c.L * ‖x - y‖)
    (hmin : ∀ x, fmin ≤ f x) (hCU : Contracts CU c.qU) (hCP : Contracts CP c.qP)
    (x₀ : Space m n) (T : ℕ) :
    c.δ ^ 2 * c.γ / 8 *
        (∑ t ∈ Finset.range T, nuclearNorm (grad (trajectory c.β c.γ grad CU CP x₀ t).x) ^ 2) ≤
      c.δ ^ 2 * (f x₀ - fmin) +
        8 * c.γ * geometryConstant m n ^ 2 * c.δ * ‖grad x₀‖ ^ 2 := by
  let S := trajectory c.β c.γ grad CU CP x₀
  have hsum : ∀ k, c.δ ^ 2 * c.γ / 8 *
      (∑ t ∈ Finset.range k, nuclearNorm (grad (S t).x) ^ 2) +
        potential c f grad fmin (S k) ≤ potential c f grad fmin (S 0) := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      have h := potential_descent c hg hlip hCU hCP fmin (S k)
      change potential c f grad fmin (S (k + 1)) + _ ≤ _ at h
      rw [Finset.sum_range_succ]
      linarith
  have h0 : potential c f grad fmin (S 0) = c.δ ^ 2 * (f x₀ - fmin) +
      8 * c.γ * geometryConstant m n ^ 2 * c.δ * ‖grad x₀‖ ^ 2 := by
    simp only [potential, S, trajectory, initial, trackingError, lag, uplinkError,
      downlinkError, zero_sub, norm_neg, sub_self, norm_zero, mul_zero, add_zero]
    ring
  have hn := potential_nonneg c grad hmin (S T)
  have hs := hsum T
  rw [h0] at hs
  change _ ≤ _
  linarith

theorem average_stationarity (c : Controls m n) {f : Space m n → ℝ}
    {grad CU CP : Space m n → Space m n} {fmin : ℝ}
    (hg : ∀ x, HasGradientAt f (grad x) x)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ c.L * ‖x - y‖)
    (hmin : ∀ x, fmin ≤ f x) (hCU : Contracts CU c.qU) (hCP : Contracts CP c.qP)
    (x₀ : Space m n) (T : ℕ) (hT : 0 < T) :
    (∑ t ∈ Finset.range T, nuclearNorm (grad (trajectory c.β c.γ grad CU CP x₀ t).x) ^ 2) / T ≤
      8 * (f x₀ - fmin) / (c.γ * T) +
        64 * geometryConstant m n ^ 2 * ‖grad x₀‖ ^ 2 / (c.δ * T) := by
  have hs := finite_time c hg hlip hmin hCU hCP x₀ T
  have hδ : c.δ ≠ 0 := ne_of_gt c.δ_pos
  have hγ : c.γ ≠ 0 := ne_of_gt c.γ_pos
  have hT' : (0 : ℝ) < T := by exact_mod_cast hT
  have hγpos := c.γ_pos
  have hδpos := c.δ_pos
  have hcoef : 0 < c.δ ^ 2 * c.γ / 8 := by positivity
  have hb : (∑ t ∈ Finset.range T,
      nuclearNorm (grad (trajectory c.β c.γ grad CU CP x₀ t).x) ^ 2) ≤
      8 * (f x₀ - fmin) / c.γ +
        64 * geometryConstant m n ^ 2 * ‖grad x₀‖ ^ 2 / c.δ := by
    apply (mul_le_mul_iff_right₀ hcoef).mp
    calc
      _ ≤ _ := hs
      _ = _ := by field_simp; ring
  apply (div_le_div_of_nonneg_right hb hT'.le).trans_eq
  field_simp

end EF21Muon
