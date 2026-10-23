import EF21Muon.Experiments.Schedules
import EF21Muon.Probability.FirstMoment

/-! Closed fixed-momentum error budgets. The scalar closure takes primitive
tracking recurrences, not a stationarity or estimator-error bound. The vector
identities and stochastic filter lemmas below justify the corresponding steps
for the actual EMA / Nesterov filter. See research/experiments/FIXED_MOMENTUM.md
for the full assumptions and the worker-averaged derivation. -/

noncomputable section
namespace EF21Muon.Experiments
open scoped BigOperators
open MeasureTheory EF21Muon.Probability
open LongTails.Algorithms.MatrixPolar

/-- A multiplicative oracle-quality guarantee avoids an artificial constant
error floor for a direction which retains a fixed fraction of optimal pairing. -/
theorem radius_direction_gain {m n : ℕ} (g h D : Space m n) {r δ gain : ℝ}
    (hgain : 0 ≤ gain) (hr : spectralNorm D ≤ r)
    (hpair : gain*nuclearNorm h-δ ≤ inner ℝ h D) :
    gain*nuclearNorm g-(gain+r)*nuclearNorm (g-h)-δ ≤ inner ℝ g D := by
  have ht := nuclearNorm_add_le h (g-h)
  rw [add_sub_cancel] at ht
  have ht' := mul_le_mul_of_nonneg_left ht hgain
  have hi := (le_abs_self (inner ℝ (h-g) D)).trans (inner_le_nuclear_mul_spectral _ _)
  have hh := mul_le_mul_of_nonneg_left hr (nuclearNorm_nonneg (h-g))
  have hrev : nuclearNorm (h-g) = nuclearNorm (g-h) := by
    rw [← neg_sub g h, nuclearNorm_neg]
  rw [inner_sub_left, hrev] at hi
  rw [hrev] at hh
  nlinarith only [ht', hi, hh, hpair]

def filterGain (β θ : ℝ) : ℝ :=
  (β+(1-β)*θ)^2+(1-θ)^2*(1-β)^2*β/(2-β)

def initialImpulse (β θ G : ℝ) (t : ℕ) : ℝ :=
  (1-θ)*β*(1-β)^t*G + if t = 0 then θ*G else 0

theorem geometric_mass {q : ℝ} (hq : 0 ≤ q) (T : ℕ) :
    (1-q)*(∑ t ∈ Finset.range T, q^t) ≤ 1 := by
  have h : (1-q)*(∑ t ∈ Finset.range T, q^t)+q^T = 1 := by
    induction T with
    | zero => simp
    | succ T ih =>
      rw [Finset.sum_range_succ, pow_succ]
      nlinarith only [ih]
  linarith only [h, pow_nonneg hq T]

theorem initialImpulse_nonneg {β θ G : ℝ} (hβ : 0 ≤ β) (hβ1 : β ≤ 1)
    (hθ : 0 ≤ θ) (hθ1 : θ ≤ 1) (hG : 0 ≤ G) (t : ℕ) :
    0 ≤ initialImpulse β θ G t := by
  unfold initialImpulse
  positivity

theorem initialImpulse_mass {β θ G : ℝ} (_hβ : 0 < β) (hβ1 : β ≤ 1)
    (hθ : 0 ≤ θ) (hθ1 : θ ≤ 1) (hG : 0 ≤ G) (T : ℕ) :
    (∑ t ∈ Finset.range T, initialImpulse β θ G t) ≤ G := by
  have hg := geometric_mass (sub_nonneg.mpr hβ1) T
  have hi : (∑ t ∈ Finset.range T, if t = 0 then θ*G else 0) ≤ θ*G := by
    by_cases ht : T = 0
    · simp [ht, mul_nonneg hθ hG]
    · simp [Finset.sum_ite_eq', Finset.mem_range.mpr (Nat.pos_of_ne_zero ht)]
  have hm := mul_le_mul_of_nonneg_left hg (mul_nonneg (sub_nonneg.mpr hθ1) hG)
  simp only [initialImpulse, Finset.sum_add_distrib] at *
  have he : (∑ t ∈ Finset.range T, (1-θ)*β*(1-β)^t*G) =
      (1-θ)*G*β*(∑ t ∈ Finset.range T, (1-β)^t) := by
    simp only [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro t _; ring
  rw [he]
  nlinarith only [hm, hi]

theorem trackingEnvelope_nonneg (q movement : ℕ → ℝ) {initial : ℝ}
    (hq : ∀ t, 0 ≤ q t) (hm : ∀ t, 0 ≤ movement t) (hi : 0 ≤ initial) (t : ℕ) :
    0 ≤ trackingEnvelope q movement initial t := by
  induction t with
  | zero => exact hi
  | succ t ih => exact mul_nonneg (hq t) (add_nonneg ih (hm t))

theorem trackingEnvelope_total {q : ℝ} (hq : 0 ≤ q) (hq1 : q < 1)
    (movement : ℕ → ℝ) (hm : ∀ t, 0 ≤ movement t) (T : ℕ) :
    (∑ t ∈ Finset.range T, trackingEnvelope (fun _ => q) movement 0 (t+1)) ≤
      q/(1-q)*(∑ t ∈ Finset.range T, movement t) := by
  let b := trackingEnvelope (fun _ => q) movement 0
  have hb t : 0 ≤ b t := trackingEnvelope_nonneg _ _ (fun _ => hq) hm (le_refl _) t
  have hs : ∀ k, (1-q)*(∑ t ∈ Finset.range k, b (t+1))+q*b k =
      q*(∑ t ∈ Finset.range k, movement t) := by
    intro k
    induction k with
    | zero => simp [b, trackingEnvelope]
    | succ k ih =>
      simp only [Finset.sum_range_succ]
      have he : b (k+1) = q*(b k+movement k) := rfl
      nlinarith only [ih, he]
  rw [div_mul_eq_mul_div]
  apply (le_div_iff₀ (sub_pos.mpr hq1)).mpr
  have h := hs T
  have hp := mul_nonneg hq (hb T)
  dsimp [b] at h hp
  nlinarith only [h, hp]

/-- A constant forcing term and an initial impulse are kept separate. -/
theorem compression_envelope {u : ℕ → ℝ} {q J ε U : ℝ}
    (hq : 0 ≤ q) (hU : 0 ≤ U) (hbalance : (1-q)*U = q*J+ε)
    (impulse : ℕ → ℝ) (hzero : u 0 = 0)
    (hstep : ∀ t, u (t+1) ≤ q*(u t+J+impulse t)+ε) (t : ℕ) :
    u t ≤ U+trackingEnvelope (fun _ => q) impulse 0 t := by
  induction t with
  | zero => simpa [hzero, trackingEnvelope] using hU
  | succ t ih =>
    have hm := mul_le_mul_of_nonneg_left ih hq
    simp only [trackingEnvelope]
    nlinarith only [hstep t, hm, hbalance]

/-- The expected Frobenius estimator error, weighted by any nonnegative
schedule with a known peak. No monotonicity of the schedule is needed. -/
theorem fixed_momentum_error_sum (η u e : ℕ → ℝ)
    {β θ q G H J ε U A peak : ℝ}
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hθ : 0 ≤ θ) (hθ1 : θ ≤ 1)
    (hq : 0 ≤ q) (hq1 : q < 1) (hG : 0 ≤ G) (hH : 0 ≤ H)
    (hpeak : 0 ≤ peak) (hη : ∀ t, 0 ≤ η t) (hηpeak : ∀ t, η t ≤ peak)
    (hU : 0 ≤ U) (hbalance : (1-q)*U = q*J+ε) (hu0 : u 0 = 0)
    (hu : ∀ t, u (t+1) ≤ q*(u t+J+initialImpulse β θ H t)+ε)
    (he : ∀ t, e t ≤ u (t+1)+(1-θ)*(1-β)^(t+1)*G+A) (T : ℕ) :
    (∑ t ∈ Finset.range T, η t*e t) ≤
      peak*((1-θ)*(1-β)*G/β+q/(1-q)*H)+
      (∑ t ∈ Finset.range T, η t)*(A+U) := by
  let b := trackingEnvelope (fun _ => q) (initialImpulse β θ H) 0
  have hi t := initialImpulse_nonneg hβ.le hβ1 hθ hθ1 hH t
  have hb t : 0 ≤ b t := trackingEnvelope_nonneg _ _ (fun _ => hq) hi (le_refl _) t
  have hu' t := compression_envelope hq hU hbalance _ hu0 hu t
  have hs := trackingEnvelope_total hq hq1 _ hi T
  have ht := mul_le_mul_of_nonneg_left (initialImpulse_mass hβ hβ1 hθ hθ1 hH T)
    (div_nonneg hq (sub_pos.mpr hq1).le)
  have hbs : (∑ t ∈ Finset.range T, b (t+1)) ≤ q/(1-q)*H := hs.trans ht
  have hweightb : (∑ t ∈ Finset.range T, η t*b (t+1)) ≤ peak*(q/(1-q)*H) := by
    calc
      _ ≤ ∑ t ∈ Finset.range T, peak*b (t+1) := Finset.sum_le_sum
        (fun t _ => mul_le_mul_of_nonneg_right (hηpeak t) (hb _))
      _ = peak*(∑ t ∈ Finset.range T, b (t+1)) := (Finset.mul_sum _ _ _).symm
      _ ≤ _ := mul_le_mul_of_nonneg_left hbs hpeak
  have hgeom := geometric_mass (sub_nonneg.mpr hβ1) T
  have hgs : (∑ t ∈ Finset.range T, (1-β)^t) ≤ 1/β := by
    apply (le_div_iff₀ hβ).mpr
    nlinarith only [hgeom]
  have hp : 0 ≤ (1-θ)*(1-β)*G := by positivity
  have hwg : (∑ t ∈ Finset.range T, η t*((1-θ)*(1-β)^(t+1)*G)) ≤
      peak*((1-θ)*(1-β)*G/β) := by
    calc
      _ ≤ ∑ t ∈ Finset.range T, peak*((1-θ)*(1-β)^(t+1)*G) :=
        Finset.sum_le_sum (fun t _ => mul_le_mul_of_nonneg_right (hηpeak t) (by positivity))
      _ = peak*((1-θ)*(1-β)*G)*(∑ t ∈ Finset.range T, (1-β)^t) := by
        simp only [Finset.mul_sum, pow_succ]
        apply Finset.sum_congr rfl
        intro t _; ring
      _ ≤ peak*((1-θ)*(1-β)*G)*(1/β) :=
        mul_le_mul_of_nonneg_left hgs (mul_nonneg hpeak hp)
      _ = _ := by ring
  have he' t : e t ≤ U+b (t+1)+(1-θ)*(1-β)^(t+1)*G+A := by
    have h := hu' (t+1)
    change u (t+1) ≤ U+b (t+1) at h
    linarith only [he t, h]
  have hs' := Finset.sum_le_sum (fun t (_ : t ∈ Finset.range T) =>
    mul_le_mul_of_nonneg_left (he' t) (hη t))
  have heq : (∑ t ∈ Finset.range T, η t*(U+b (t+1)+(1-θ)*(1-β)^(t+1)*G+A)) =
      (∑ t ∈ Finset.range T, η t*b (t+1))+
      (∑ t ∈ Finset.range T, η t*((1-θ)*(1-β)^(t+1)*G))+
      (∑ t ∈ Finset.range T, η t)*(A+U) := by
    simp only [Finset.sum_mul, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro t _; ring
  rw [heq] at hs'
  nlinarith only [hs', hweightb, hwg]

/-- Compose the primitive tracking recurrences with the true objective drift.
This closes the error term instead of assuming a stationarity estimate. -/
theorem fixed_momentum_stationarity (F η S e u δ : ℕ → ℝ)
    {β θ q G H J ε U A peak K fmin C : ℝ}
    (hβ : 0 < β) (hβ1 : β ≤ 1) (hθ : 0 ≤ θ) (hθ1 : θ ≤ 1)
    (hq : 0 ≤ q) (hq1 : q < 1) (hG : 0 ≤ G) (hH : 0 ≤ H)
    (hpeak : 0 ≤ peak) (hη : ∀ t, 0 ≤ η t) (hηpeak : ∀ t, η t ≤ peak)
    (hU : 0 ≤ U) (hbalance : (1-q)*U = q*J+ε) (hu0 : u 0 = 0)
    (hu : ∀ t, u (t+1) ≤ q*(u t+J+initialImpulse β θ H t)+ε)
    (he : ∀ t, e t ≤ u (t+1)+(1-θ)*(1-β)^(t+1)*G+A)
    (hK : 0 ≤ K) (T : ℕ) (hmass : 0 < ∑ t ∈ Finset.range T, η t)
    (hlower : fmin ≤ F T)
    (hstep : ∀ t < T, F (t+1) ≤ F t-η t*S t+η t*(K*e t+δ t)+C/2*(η t)^2) :
    (∑ t ∈ Finset.range T, η t*S t)/(∑ t ∈ Finset.range T, η t) ≤
      (F 0-fmin+K*(peak*((1-θ)*(1-β)*G/β+q/(1-q)*H)+
        (∑ t ∈ Finset.range T, η t)*(A+U))+
        (∑ t ∈ Finset.range T, η t*δ t)+C/2*(∑ t ∈ Finset.range T, (η t)^2))/
      (∑ t ∈ Finset.range T, η t) := by
  have he' := fixed_momentum_error_sum η u e hβ hβ1 hθ hθ1 hq hq1 hG hH
    hpeak hη hηpeak hU hbalance hu0 hu he T
  have hs := scheduled_descent_sum F η S (fun t => K*e t+δ t) fmin C T hlower hstep
  have heq : (∑ t ∈ Finset.range T, η t*(K*e t+δ t)) =
      K*(∑ t ∈ Finset.range T, η t*e t)+(∑ t ∈ Finset.range T, η t*δ t) := by
    simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro t _; ring
  rw [heq] at hs
  apply div_le_div_of_nonneg_right _ hmass.le
  have hm := mul_le_mul_of_nonneg_left he' hK
  linarith only [hs, hm]

theorem momentum_bias_cap (a : ℕ → ℝ) {β G A : ℝ}
    (_hβ : 0 < β) (hβ1 : β ≤ 1) (hA : 0 ≤ A) (ha0 : a 0 ≤ G)
    (ha : ∀ t, a (t+1) ≤ (1-β)*a t+β*A) (t : ℕ) :
    a t ≤ (1-β)^t*G+A := by
  induction t with
  | zero => simp only [pow_zero, one_mul]; linarith only [ha0, hA]
  | succ t ih =>
    have hh := mul_le_mul_of_nonneg_left ih (sub_nonneg.mpr hβ1)
    rw [pow_succ]
    nlinarith only [ha t, hh]

section Vector
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem radius_bias_identity (β : ℝ) (M Z gx gw gx' ξ : E) :
    ((1-β) • M+β • (gw+ξ))-gx'-((1-β) • Z+β • ξ) =
      (1-β) • (M-gx-Z)+β • (gw-gx)+(gx-gx') := by module

theorem filtered_target_identity (β θ : ℝ) (M Z gx gw ξ : E) :
    (1-θ) • ((1-β) • M+β • (gw+ξ))+θ • (gw+ξ)-gx =
      ((1-θ)*(1-β)) • (M-gx-Z)+(β+(1-β)*θ) • (gw-gx)+
      (((1-θ)*(1-β)) • Z+(β+(1-β)*θ) • ξ) := by module

theorem target_increment_identity (β θ : ℝ) (M reply previous : E) :
    ((1-θ) • ((1-β) • M+β • reply)+θ • reply)-
      ((1-θ) • M+θ • previous) =
      ((1-θ)*β) • (reply-M)+θ • (reply-previous) := by module

theorem radius_bias_bound {β : ℝ} (hβ : 0 ≤ β) (hβ1 : β ≤ 1)
    (M Z gx gw gx' ξ : E) :
    ‖((1-β) • M+β • (gw+ξ))-gx'-((1-β) • Z+β • ξ)‖ ≤
      (1-β)*‖M-gx-Z‖+β*‖gw-gx‖+‖gx-gx'‖ := by
  rw [radius_bias_identity]
  have h1 := norm_add_le ((1-β) • (M-gx-Z)+β • (gw-gx)) (gx-gx')
  have h2 := norm_add_le ((1-β) • (M-gx-Z)) (β • (gw-gx))
  simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg hβ,
    abs_of_nonneg (sub_nonneg.mpr hβ1)] at h2
  linarith only [h1, h2]

theorem target_increment_bound {β θ : ℝ} (hβ : 0 ≤ β)
    (hθ : 0 ≤ θ) (hθ1 : θ ≤ 1) (M reply previous : E) :
    ‖((1-θ) • ((1-β) • M+β • reply)+θ • reply)-
      ((1-θ) • M+θ • previous)‖ ≤
      (1-θ)*β*‖reply-M‖+θ*‖reply-previous‖ := by
  rw [target_increment_identity]
  have h := norm_add_le (((1-θ)*β) • (reply-M)) (θ • (reply-previous))
  simpa only [norm_smul, Real.norm_eq_abs, abs_of_nonneg hθ,
    abs_of_nonneg (mul_nonneg (sub_nonneg.mpr hθ1) hβ)] using h
end Vector

section Noise
variable {Ω E : Type*} [MeasurableSpace Ω]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

omit [CompleteSpace E] [IsProbabilityMeasure μ] in
/-- Orthogonality to past noise, rather than independence of changing queries,
is the stochastic premise. It follows from martingale-difference noise. -/
theorem orthogonal_variance (a b : ℝ) {Z ξ : Ω → E}
    (hZ : MemLp Z 2 μ) (hξ : MemLp ξ 2 μ)
    (hcross : (∫ ω, inner ℝ (Z ω) (ξ ω) ∂μ) = 0) :
    (∫ ω, ‖a • Z ω+b • ξ ω‖^2 ∂μ) =
      a^2*(∫ ω, ‖Z ω‖^2 ∂μ)+b^2*(∫ ω, ‖ξ ω‖^2 ∂μ) := by
  have hi := inner_integrable hZ hξ
  have hsZ := square_integrable hZ
  have hsξ := square_integrable hξ
  have he (ω : Ω) : ‖a • Z ω+b • ξ ω‖^2 =
      a^2*‖Z ω‖^2+2*a*b*inner ℝ (Z ω) (ξ ω)+b^2*‖ξ ω‖^2 := by
    rw [norm_add_sq_real, inner_smul_left, inner_smul_right,
      norm_smul, norm_smul, mul_pow, mul_pow, Real.norm_eq_abs, Real.norm_eq_abs,
      sq_abs, sq_abs, conj_trivial]
    ring
  simp_rw [he]
  have hi' : Integrable (fun ω => a^2*‖Z ω‖^2+2*a*b*inner ℝ (Z ω) (ξ ω)) μ :=
    (hsZ.const_mul _).add (hi.const_mul _)
  rw [integral_add hi' (hsξ.const_mul _),
    integral_add (hsZ.const_mul _) (hi.const_mul _), integral_const_mul,
    integral_const_mul, integral_const_mul, hcross]
  ring

theorem noise_variance_cap {β σ : ℝ} (hβ : 0 < β) (hβ1 : β ≤ 1)
    (v : ℕ → ℝ) (hv0 : v 0 = 0)
    (hv : ∀ t, v (t+1) ≤ (1-β)^2*v t+β^2*σ^2) (t : ℕ) :
    v t ≤ σ^2*β/(2-β) := by
  have hd : 0 < 2-β := by linarith
  have hfix : (1-β)^2*(σ^2*β/(2-β))+β^2*σ^2 = σ^2*β/(2-β) := by
    field_simp; ring
  induction t with
  | zero => rw [hv0]; positivity
  | succ t ih =>
    exact (hv t).trans (by nlinarith only [mul_le_mul_of_nonneg_left ih (sq_nonneg (1-β)), hfix])

omit [CompleteSpace E] in
theorem filtered_noise_bound {β θ σ : ℝ} (hβ : 0 < β) (hβ1 : β ≤ 1)
    (hσ : 0 ≤ σ) {Z ξ : Ω → E} (hZ : MemLp Z 2 μ) (hξ : MemLp ξ 2 μ)
    (hcross : (∫ ω, inner ℝ (Z ω) (ξ ω) ∂μ) = 0)
    (hvZ : (∫ ω, ‖Z ω‖^2 ∂μ) ≤ σ^2*β/(2-β))
    (hvξ : (∫ ω, ‖ξ ω‖^2 ∂μ) ≤ σ^2) :
    (∫ ω, ‖((1-θ)*(1-β)) • Z ω+(β+(1-β)*θ) • ξ ω‖ ∂μ) ≤
      σ*Real.sqrt (filterGain β θ) := by
  have hd : 0 < 2-β := by linarith
  have hg : 0 ≤ filterGain β θ := by unfold filterGain; positivity
  have hv := orthogonal_variance ((1-θ)*(1-β)) (β+(1-β)*θ) hZ hξ hcross
  have h1 := mul_le_mul_of_nonneg_left hvZ (sq_nonneg ((1-θ)*(1-β)))
  have h2 := mul_le_mul_of_nonneg_left hvξ (sq_nonneg (β+(1-β)*θ))
  apply integral_norm_le_of_sq ((hZ.const_smul _).add (hξ.const_smul _)) (mul_nonneg hσ (Real.sqrt_nonneg _))
  simp only [Pi.add_apply, Pi.smul_apply]
  rw [hv]
  simp only [mul_pow, Real.sq_sqrt hg]
  convert add_le_add h1 h2 using 1 <;> (try unfold filterGain) <;> ring

theorem filterGain_ema {β : ℝ} (hβ : β ≠ 2) :
    filterGain β 0 = β/(2-β) := by
  unfold filterGain
  have hd : 2-β ≠ 0 := sub_ne_zero.mpr (Ne.symm hβ)
  field_simp
  ring

theorem filterGain_nesterov (β : ℝ) :
    filterGain β β = β^2*(2-β)^2+β*(1-β)^4/(2-β) := by
  unfold filterGain
  ring
end Noise
end EF21Muon.Experiments
