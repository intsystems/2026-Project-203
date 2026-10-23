import EF21Muon.Stochastic.MatrixNorms
import EF21Muon.Compression.ScaledSign
import Mathlib.Tactic.GCongr

/-! Machinery for experimental fixed-radius steps and measured EF21-P traces.
These statements keep direction/tracking errors explicit. They do not assert
that stochastic minibatch gradients equal the objective gradient. -/

noncomputable section

namespace EF21Muon.Experiments

open scoped BigOperators
open LongTails.Algorithms.MatrixPolar

/-- Approximate polar directions are allowed: r bounds the operator norm and
delta bounds the loss in pairing with the estimator. -/
theorem radius_direction_bound {m n : ℕ} (g h D : Space m n) {r δ : ℝ}
    (hr : spectralNorm D ≤ r) (hpair : nuclearNorm h-δ ≤ inner ℝ h D) :
    nuclearNorm g-(1+r)*nuclearNorm (g-h)-δ ≤ inner ℝ g D := by
  have ht := nuclearNorm_add_le h (g-h)
  rw [add_sub_cancel] at ht
  have hi := (le_abs_self (inner ℝ (h-g) D)).trans (inner_le_nuclear_mul_spectral _ _)
  have hh := mul_le_mul_of_nonneg_left hr (nuclearNorm_nonneg (h-g))
  have hrev : nuclearNorm (h-g) = nuclearNorm (g-h) := by
    rw [← neg_sub g h, nuclearNorm_neg]
  rw [inner_sub_left] at hi
  rw [hrev] at hi hh
  nlinarith only [ht, hi, hh, hpair]

theorem radius_descent {m n : ℕ} (g h D : Space m n) {r δ η F F' L : ℝ}
    (hη : 0 ≤ η) (hr : spectralNorm D ≤ r)
    (hpair : nuclearNorm h-δ ≤ inner ℝ h D)
    (hsmooth : F' ≤ F-η*inner ℝ g D+L/2*η^2) :
    F' ≤ F-η*nuclearNorm g+η*((1+r)*nuclearNorm (g-h)+δ)+L/2*η^2 := by
  have h := mul_le_mul_of_nonneg_left (radius_direction_bound g h D hr hpair) hη
  nlinarith only [h, hsmooth]

theorem query_gradient_transfer {m n : ℕ} (gX gW : Space m n) {L gap : ℝ}
    (h : nuclearNorm (gW-gX) ≤ L*gap) :
    nuclearNorm gW ≤ nuclearNorm gX+L*gap := by
  have ht := nuclearNorm_add_le gX (gW-gX)
  rw [add_sub_cancel] at ht
  linarith only [ht, h]

/-- Telescoping descent for any nonnegative schedule, including zero steps.
F may also be a sequence of expectations when the one-step inequality has
separately been justified. No probabilistic premise is hidden here. -/
theorem scheduled_descent_sum (F η S e : ℕ → ℝ) (fmin L : ℝ) (T : ℕ)
    (hlower : fmin ≤ F T)
    (hstep : ∀ t < T, F (t+1) ≤ F t-η t*S t+η t*e t+L/2*(η t)^2) :
    (∑ t ∈ Finset.range T, η t*S t) ≤
      F 0-fmin+(∑ t ∈ Finset.range T, η t*e t)+L/2*(∑ t ∈ Finset.range T, (η t)^2) := by
  have hs : ∀ k ≤ T, F k+(∑ t ∈ Finset.range k, η t*S t) ≤
      F 0+(∑ t ∈ Finset.range k, η t*e t)+L/2*(∑ t ∈ Finset.range k, (η t)^2) := by
    intro k hk
    induction k with
    | zero => simp
    | succ k ih =>
      have h := ih (by omega)
      have h' := hstep k (by omega)
      simp only [Finset.sum_range_succ]
      linarith
  linarith only [hs T (le_refl T), hlower]

theorem scheduled_stationarity (F η S e : ℕ → ℝ) (fmin L : ℝ) (T : ℕ)
    (hmass : 0 < ∑ t ∈ Finset.range T, η t) (hlower : fmin ≤ F T)
    (hstep : ∀ t < T, F (t+1) ≤ F t-η t*S t+η t*e t+L/2*(η t)^2) :
    (∑ t ∈ Finset.range T, η t*S t)/(∑ t ∈ Finset.range T, η t) ≤
      (F 0-fmin+(∑ t ∈ Finset.range T, η t*e t)+
        L/2*(∑ t ∈ Finset.range T, (η t)^2))/(∑ t ∈ Finset.range T, η t) := by
  exact div_le_div_of_nonneg_right (scheduled_descent_sum F η S e fmin L T hlower hstep) hmass.le

/-- A bounded direction/tracking defect remains in the certificate even if
the schedule's quadratic penalty tends to zero. This is not a lower bound. -/
theorem scheduled_residual (F η S e : ℕ → ℝ) (fmin L ε : ℝ) (T : ℕ)
    (hη : ∀ t < T, 0 ≤ η t) (hmass : 0 < ∑ t ∈ Finset.range T, η t)
    (hlower : fmin ≤ F T) (he : ∀ t < T, e t ≤ ε)
    (hstep : ∀ t < T, F (t+1) ≤ F t-η t*S t+η t*e t+L/2*(η t)^2) :
    (∑ t ∈ Finset.range T, η t*S t)/(∑ t ∈ Finset.range T, η t) ≤
      (F 0-fmin)/(∑ t ∈ Finset.range T, η t)+ε+
        (L/2*(∑ t ∈ Finset.range T, (η t)^2))/(∑ t ∈ Finset.range T, η t) := by
  have he' : (∑ t ∈ Finset.range T, η t*e t) ≤ (∑ t ∈ Finset.range T, η t)*ε := by
    rw [Finset.sum_mul]
    exact Finset.sum_le_sum (fun t ht => mul_le_mul_of_nonneg_left
      (he t (Finset.mem_range.mp ht)) (hη t (Finset.mem_range.mp ht)))
  have h := div_le_div_of_nonneg_right
    (scheduled_descent_sum F η S e fmin L T hlower hstep) hmass.le
  have h' := div_le_div_of_nonneg_right he' hmass.le
  field_simp at h h' ⊢
  nlinarith only [h, h']

def trackingEnvelope (q movement : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | t+1 => q t*(trackingEnvelope q movement initial t+movement t)

/-- The multiplier is needed only at the residual actually compressed. -/
theorem tracking_step {E : Type*} [NormedAddCommGroup E]
    (oldGap movement : E) (C : E → E) {q : ℝ} (hq : 0 ≤ q)
    (hc : ‖oldGap+movement-C (oldGap+movement)‖ ≤ q*‖oldGap+movement‖) :
    ‖oldGap+movement-C (oldGap+movement)‖ ≤ q*(‖oldGap‖+‖movement‖) :=
  hc.trans (mul_le_mul_of_nonneg_left (norm_add_le oldGap movement) hq)

theorem tracking_envelope (gap q movement : ℕ → ℝ) (T : ℕ)
    (hq : ∀ t < T, 0 ≤ q t)
    (hstep : ∀ t < T, gap (t+1) ≤ q t*(gap t+movement t)) :
    gap T ≤ trackingEnvelope q movement (gap 0) T := by
  have hs : ∀ k ≤ T, gap k ≤ trackingEnvelope q movement (gap 0) k := by
    intro k hk
    induction k with
    | zero => exact le_refl _
    | succ k ih =>
      exact (hstep k (by omega)).trans (mul_le_mul_of_nonneg_left
        (add_le_add (ih (by omega)) (le_refl _)) (hq k (by omega)))
  exact hs T (le_refl T)

end EF21Muon.Experiments
