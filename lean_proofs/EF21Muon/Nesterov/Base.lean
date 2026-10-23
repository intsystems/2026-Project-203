import EF21Muon.Distributed.Setup
import EF21Muon.Distributed.RadiusTracking

/-! The paper's Nesterov filter, with the preceding EMA buffer retained.
The round is cut as step/broadcast/query/uplink, as in the EMA formalization.
The paper-order correspondence is supplied separately. -/

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

abbrev Layer (N m n : ℕ) := LayerState N m n × (Fin N → Space m n)
abbrev States (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) := ∀ i, Layer N (m i) (n i)

variable {N m n : ℕ}

def target (c : Parameters) (s : Layer N m n) (j : Fin N) : Space m n :=
  (2-c.β) • s.1.M j - (1-c.β) • s.2 j

def increment (s : Layer N m n) : ℝ := average (fun j => ‖s.1.M j-s.2 j‖^2)

def compression (c : Parameters) (s : Layer N m n) : ℝ :=
  average (fun j => ‖target c s j-s.1.G j‖^2)

def firstIncrement (s : Layer N m n) : ℝ := average (fun j => ‖s.1.M j-s.2 j‖)

def firstCompression (c : Parameters) (s : Layer N m n) : ℝ :=
  average (fun j => ‖target c s j-s.1.G j‖)

def update (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) (reply : Fin N → Space m n) : Layer N m n :=
  let M := fun j => (1-c.β) • s.1.M j + c.β • reply j
  let V := fun j => (1-c.β) • M j + c.β • reply j
  ((nextX c s.1, nextW c CP s.1, M, fun j => s.1.G j+CU (V j-s.1.G j)), s.1.M)

def start (x : Space m n) : Layer N m n := (LayerState.start x, 0)

theorem update_M (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) (reply : Fin N → Space m n) (j : Fin N) :
    (update c CU CP s reply).1.M j = (1-c.β) • s.1.M j+c.β • reply j := rfl

theorem update_target (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) (reply : Fin N → Space m n) (j : Fin N) :
    target c (update c CU CP s reply) j =
      (1-c.β) • (update c CU CP s reply).1.M j+c.β • reply j := by
  dsimp [target, update, LayerState.M]
  module

theorem target_increment (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) (reply : Fin N → Space m n) (j : Fin N) :
    target c (update c CU CP s reply) j-target c s j =
      (2-c.β) • ((update c CU CP s reply).1.M j-s.1.M j) -
        (1-c.β) • (s.1.M j-s.2 j) := by
  dsimp [target, update, LayerState.M]
  module

theorem target_increment_sq (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) (reply : Fin N → Space m n) (j : Fin N) :
    ‖target c (update c CU CP s reply) j-target c s j‖^2 ≤
      8*‖(update c CU CP s reply).1.M j-s.1.M j‖^2 + 2*‖s.1.M j-s.2 j‖^2 := by
  rw [target_increment]
  have ht := two_sq ((2-c.β) • ((update c CU CP s reply).1.M j-s.1.M j))
    (-(1-c.β) • (s.1.M j-s.2 j))
  simp only [neg_smul, ← sub_eq_add_neg] at ht
  simp only [norm_neg, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs] at ht
  have hb : (2-c.β)^2 ≤ 4 := by nlinarith [c.β_pos, c.β_le_one]
  have ha : (1-c.β)^2 ≤ 1 := by nlinarith [c.β_pos, c.β_le_one]
  nlinarith [mul_le_mul_of_nonneg_right hb
    (sq_nonneg ‖(update c CU CP s reply).1.M j-s.1.M j‖),
    mul_le_mul_of_nonneg_right ha (sq_nonneg ‖s.1.M j-s.2 j‖)]

theorem residual_identity (c : Parameters) (CU CP : Space m n → Space m n)
    (s : Layer N m n) (reply : Fin N → Space m n) (j : Fin N) :
    ‖target c (update c CU CP s reply) j-(update c CU CP s reply).1.G j‖ =
      ‖CU (target c (update c CU CP s reply) j-s.1.G j) -
        (target c (update c CU CP s reply) j-s.1.G j)‖ := by
  rw [norm_sub_rev]
  congr 1
  rw [update_target]
  dsimp [update, LayerState.G, LayerState.M]
  module

theorem mean_error_sq (c : Parameters) (hN : 0 < N) (s : Layer N m n)
    (g : Space m n) (localG : Fin N → Space m n) (hmean : average localG = g) :
    ‖average s.1.G-g‖^2 ≤
      3*(compression c s+localError s.1 localG+increment s) := by
  have hid (j : Fin N) : s.1.G j-localG j =
      (s.1.G j-target c s j)+(s.1.M j-localG j)+
        (1-c.β) • (s.1.M j-s.2 j) := by dsimp [target]; module
  have ha : (1-c.β)^2 ≤ 1 := by nlinarith [c.β_pos, c.β_le_one]
  have hj (j : Fin N) : ‖s.1.G j-localG j‖^2 ≤
      3*(‖target c s j-s.1.G j‖^2+‖s.1.M j-localG j‖^2+‖s.1.M j-s.2 j‖^2) := by
    have h := three_sq (s.1.G j-target c s j) (s.1.M j-localG j)
      ((1-c.β) • (s.1.M j-s.2 j))
    rw [← hid j, norm_sub_rev (s.1.G j) (target c s j)] at h
    simp only [norm_smul, mul_pow, Real.norm_eq_abs, sq_abs] at h
    nlinarith [h, mul_le_mul_of_nonneg_right ha (sq_nonneg ‖s.1.M j-s.2 j‖)]
  have hav := average_sq_le hN (fun j => s.1.G j-localG j)
  rw [average_sub, hmean] at hav
  change ‖average s.1.G-g‖^2 ≤ average (fun j => ‖s.1.G j-localG j‖^2) at hav
  exact hav.trans (by
    simpa only [average_mul, average_add, compression, localError, increment]
      using average_mono hj)

theorem first_mean_error (c : Parameters) (s : Layer N m n) (g Z : Space m n) :
    ‖average s.1.G-g‖ ≤
      debiasedError s.1 g Z+firstCompression c s+firstIncrement s+‖Z‖ := by
  have hM := norm_add_le (average s.1.M-g-Z) Z
  have hid : average s.1.M-g-Z+Z = average s.1.M-g := by module
  rw [hid] at hM
  have htarget j : target c s j-s.1.M j = (1-c.β) • (s.1.M j-s.2 j) := by
    dsimp [target]; module
  have hb : (1-c.β) ≤ 1 := by linarith [c.β_pos]
  have hj (j : Fin N) : ‖s.1.G j-s.1.M j‖ ≤
      ‖target c s j-s.1.G j‖+‖s.1.M j-s.2 j‖ := by
    have hpoint := norm_add_le (s.1.G j-target c s j) (target c s j-s.1.M j)
    rw [sub_add_sub_cancel, norm_sub_rev (s.1.G j) (target c s j), htarget,
      norm_smul, Real.norm_eq_abs, abs_of_nonneg (sub_nonneg.mpr c.β_le_one)] at hpoint
    nlinarith [hpoint, mul_le_mul_of_nonneg_right hb (norm_nonneg (s.1.M j-s.2 j))]
  have hG := norm_average_le (fun j => s.1.G j-s.1.M j)
  rw [average_sub] at hG
  have hbound := hG.trans (by simpa only [average_add] using average_mono hj)
  have ht := norm_add_le (average s.1.G-average s.1.M) (average s.1.M-g)
  rw [sub_add_sub_cancel] at ht
  dsimp [debiasedError, firstCompression, firstIncrement]
  linarith only [ht, hM, hbound]

end EF21Muon.Nesterov
