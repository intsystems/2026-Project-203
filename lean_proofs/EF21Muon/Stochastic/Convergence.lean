import EF21Muon.Stochastic.Constants

noncomputable section

namespace EF21Muon.Stochastic

open LongTails.Algorithms.MatrixPolar
open LongTails.Proofs.MatrixSmooth

variable {m n : ℕ} {Ω : Type*} [Fintype Ω]

theorem sharp_descent (c : Parameters) {f : Space m n → ℝ}
    (hg : ∀ x, HasGradientAt f (gradient f x) x)
    (hlip : ∀ x y, nuclearNorm (gradient f x - gradient f y) ≤ c.L * spectralNorm (x - y))
    (hdual : ∀ z : Space m n, nuclearNorm z ≤ c.ρ * ‖z‖) (s : State m n) :
    f (serverPoint c.γ s) ≤ f s.x + 3 * c.ρ ^ 2 * c.γ *
      (momentumSq (gradient f) s + uplinkSq s) - c.γ / 4 * nuclearNorm (gradient f s.x) ^ 2 -
        c.γ / 4 * nuclearNorm s.estimator ^ 2 +
        c.L / 2 * c.γ ^ 2 * nuclearNorm s.estimator ^ 2 := by
  let a := nuclearNorm s.estimator
  let g := nuclearNorm (gradient f s.x)
  let e := nuclearNorm (s.estimator - gradient f s.x)
  have hd := polar_descent hg hlip c.L_nonneg
    (mul_nonneg c.γ_pos.le (nuclearNorm_nonneg s.estimator)) s.x s.estimator
  have hi := (le_abs_self (inner ℝ (s.estimator - gradient f s.x) (polar s.estimator))).trans
    (inner_le_nuclear_mul_spectral _ _)
  have hi' := mul_le_mul_of_nonneg_left (polar_mem s.estimator)
    (nuclearNorm_nonneg (s.estimator - gradient f s.x))
  rw [inner_sub_left, polar_pairing_eq_nuclear] at hi
  have hp : a - e ≤ inner ℝ (gradient f s.x) (polar s.estimator) := by
    dsimp [a, e]; linarith only [hi, hi']
  have htri := nuclearNorm_add_le s.estimator (gradient f s.x - s.estimator)
  rw [add_sub_cancel] at htri
  have hrev : nuclearNorm (gradient f s.x - s.estimator) = e := by
    dsimp [e]; rw [← neg_sub s.estimator (gradient f s.x), nuclearNorm_neg]
  rw [hrev] at htri
  have hg2 := mul_self_le_mul_self (nuclearNorm_nonneg (gradient f s.x)) htri
  have hnorm := two_sq (s.estimator - s.momentum) (s.momentum - gradient f s.x)
  rw [sub_add_sub_cancel, norm_sub_rev s.estimator s.momentum] at hnorm
  have he := hdual (s.estimator - gradient f s.x)
  have he2 := mul_self_le_mul_self (nuclearNorm_nonneg _) he
  have he3 := mul_le_mul_of_nonneg_left hnorm (sq_nonneg c.ρ)
  have heb : e ^ 2 ≤ 2 * c.ρ ^ 2 * (momentumSq (gradient f) s + uplinkSq s) := by
    dsimp [e, momentumSq, uplinkSq]; nlinarith only [he2, he3]
  have hgamm := mul_le_mul_of_nonneg_left heb c.γ_pos.le
  have hpair := mul_le_mul_of_nonneg_left hp
    (mul_nonneg c.γ_pos.le (nuclearNorm_nonneg s.estimator))
  have hs := mul_nonneg c.γ_pos.le (sq_nonneg (a / 2 - e))
  have hs2 := mul_nonneg c.γ_pos.le (sq_nonneg (a - e))
  have hg2' := mul_le_mul_of_nonneg_left hg2 c.γ_pos.le
  change f (serverPoint c.γ s) ≤ _ at hd
  change f (serverPoint c.γ s) ≤ f s.x + _ - c.γ / 4 * g ^ 2 -
    c.γ / 4 * a ^ 2 + c.L / 2 * c.γ ^ 2 * a ^ 2
  dsimp only [a, g, e] at *
  nlinarith only [hd, hpair, hs, hs2, hg2', hgamm]

def potential (c : Parameters) (f : Space m n → ℝ) (fmin : ℝ) (s : State m n) : ℝ :=
  f s.x - fmin + c.γ * c.momentumWeight * momentumSq (gradient f) s +
    c.γ * c.uplinkWeight * uplinkSq s + c.γ * c.downlinkWeight * downlinkSq s

theorem potential_nonneg (c : Parameters) {f : Space m n → ℝ} {fmin : ℝ}
    (hmin : ∀ x, fmin ≤ f x) (s : State m n) : 0 ≤ potential c f fmin s := by
  rcases c.weights_nonneg with ⟨hW, hU, hI, hD⟩
  have hγ := c.γ_pos
  have hf := sub_nonneg.mpr (hmin s.x)
  unfold potential momentumSq uplinkSq downlinkSq
  positivity

set_option maxHeartbeats 800000 in
theorem potential_descent (c : Parameters) (τ : ℝ) {p : Sampling Ω} {f : Space m n → ℝ} {σ : ℝ}
    (O : Oracle p (gradient f) σ)
    (hg : ∀ x, HasGradientAt f (gradient f x) x)
    (hlip : ∀ x y, nuclearNorm (gradient f x - gradient f y) ≤ c.L * spectralNorm (x - y))
    (hdual : ∀ z : Space m n, nuclearNorm z ≤ c.ρ * ‖z‖)
    {CU CP : Space m n → Space m n}
    (hCU : SquaredContracts CU c.αU) (hdown : BroadcastBound c CP τ)
    (hsmall : 2 * c.L * c.γ + 4 * c.zetaWith τ * c.γ ^ 2 ≤ 1)
    (fmin : ℝ) (s : State m n) :
    p.expect (fun ω => potential c f fmin (advance c O.reply CU CP s ω)) +
      c.γ / 4 * nuclearNorm (gradient f s.x) ^ 2 ≤
        potential c f fmin s + c.γ * c.noise * σ ^ 2 := by
  rcases c.weights_nonneg with ⟨hW, hU, hI, hD⟩
  have hγ := c.γ_pos
  have hd := sharp_descent c hg hlip hdual s
  have hm := mul_le_mul_of_nonneg_left (momentum_rec c O hlip CU CP s)
    (mul_nonneg hγ.le hW)
  have hu := mul_le_mul_of_nonneg_left (uplink_rec c O hlip hCU CP s)
    (mul_nonneg hγ.le hU)
  have hp := mul_le_mul_of_nonneg_left (hdown s)
    (mul_nonneg hγ.le (add_nonneg hI hD))
  rcases c.coefficients with ⟨hE, hH, hP, hN⟩
  have hE' := congrArg (fun r : ℝ => c.γ * r * momentumSq (gradient f) s) hE
  have hH' := congrArg (fun r : ℝ => c.γ * r * uplinkSq s) hH
  have hP' := congrArg (fun r : ℝ => c.γ * r * downlinkSq s) hP
  have hN' := congrArg (fun r : ℝ => c.γ * r * σ ^ 2) hN
  have hZ : c.γ ^ 3 * c.zetaWith τ * nuclearNorm s.estimator ^ 2 =
      c.γ ^ 3 * (c.momentumWeight * (2 * c.L ^ 2 / c.β) +
        c.uplinkWeight * (6 * c.β ^ 2 * c.L ^ 2 / c.αU) +
        (c.downlinkInjection + c.downlinkWeight) * (2 * τ / c.αP)) *
          nuclearNorm s.estimator ^ 2 := rfl
  have hI' : c.γ * c.downlinkInjection *
      ‖queryPoint c.γ CP s - serverPoint c.γ s‖ ^ 2 =
      c.γ * (c.momentumWeight * c.β ^ 2 * (1 + 2 / c.β) * c.L ^ 2 +
        c.uplinkWeight * (6 * c.β ^ 2 * c.L ^ 2 / c.αU)) *
          ‖queryPoint c.γ CP s - serverPoint c.γ s‖ ^ 2 := rfl
  have hs := mul_le_mul_of_nonneg_left hsmall
    (by positivity : 0 ≤ c.γ / 4 * nuclearNorm s.estimator ^ 2)
  have hxp : ∀ ω, (advance c O.reply CU CP s ω).x = serverPoint c.γ s := fun _ => rfl
  have hwp : ∀ ω, downlinkSq (advance c O.reply CU CP s ω) =
      ‖queryPoint c.γ CP s - serverPoint c.γ s‖ ^ 2 := fun _ => rfl
  simp only [potential, hxp, hwp, Sampling.expect_add, Sampling.expect_sub,
    Sampling.expect_const, Sampling.expect_mul]
  simp only [div_eq_mul_inv] at hd hm hu hp hE' hH' hP' hN' hZ hI' hs ⊢
  nlinarith only [hd, hm, hu, hp, hE', hH', hP', hN', hZ, hI', hs]

/-- Finite-time stochastic stationarity for the actual EF21 sampling tree. -/
theorem finite_time (c : Parameters) (τ : ℝ) {p : Sampling Ω} {f : Space m n → ℝ} {σ fmin : ℝ}
    (O : Oracle p (gradient f) σ)
    (hg : ∀ x, HasGradientAt f (gradient f x) x)
    (hlip : ∀ x y, nuclearNorm (gradient f x - gradient f y) ≤ c.L * spectralNorm (x - y))
    (hmin : ∀ x, fmin ≤ f x)
    (hdual : ∀ z : Space m n, nuclearNorm z ≤ c.ρ * ‖z‖)
    {CU CP : Space m n → Space m n}
    (hCU : SquaredContracts CU c.αU) (hdown : BroadcastBound c CP τ)
    (hsmall : 2 * c.L * c.γ + 4 * c.zetaWith τ * c.γ ^ 2 ≤ 1)
    (T : ℕ) (s : State m n) :
    expectedTotal p (advance c O.reply CU CP)
        (fun s => c.γ / 4 * nuclearNorm (gradient f s.x) ^ 2) T s ≤
      potential c f fmin s + (T : ℝ) * c.γ * c.noise * σ ^ 2 := by
  have h := expectedTotal_le p (advance c O.reply CU CP)
    (fun s => c.γ / 4 * nuclearNorm (gradient f s.x) ^ 2)
    (potential c f fmin) (c.γ * c.noise * σ ^ 2)
    (potential_nonneg c hmin)
    (potential_descent c τ O hg hlip hdual hCU hdown hsmall fmin) T s
  convert h using 1; ring

theorem initial_potential (c : Parameters) (f : Space m n → ℝ) (fmin : ℝ) (x₀ : Space m n) :
    potential c f fmin (initial x₀) = f x₀ - fmin +
      c.γ * c.momentumWeight * ‖gradient f x₀‖ ^ 2 := by
  simp [potential, initial, momentumSq, uplinkSq, downlinkSq]

/-- Paper-form bound: 4 Psi_0/(gamma T) plus the explicit noise floor. -/
theorem average_stationarity (c : Parameters) (τ : ℝ) {p : Sampling Ω} {f : Space m n → ℝ} {σ fmin : ℝ}
    (O : Oracle p (gradient f) σ)
    (hg : ∀ x, HasGradientAt f (gradient f x) x)
    (hlip : ∀ x y, nuclearNorm (gradient f x - gradient f y) ≤ c.L * spectralNorm (x - y))
    (hmin : ∀ x, fmin ≤ f x)
    (hdual : ∀ z : Space m n, nuclearNorm z ≤ c.ρ * ‖z‖)
    {CU CP : Space m n → Space m n}
    (hCU : SquaredContracts CU c.αU) (hdown : BroadcastBound c CP τ)
    (hsmall : 2 * c.L * c.γ + 4 * c.zetaWith τ * c.γ ^ 2 ≤ 1)
    (x₀ : Space m n) (T : ℕ) (hT : 0 < T) :
    expectedTotal p (advance c O.reply CU CP)
        (fun s => nuclearNorm (gradient f s.x) ^ 2) T (initial x₀) / T ≤
      4 * (f x₀ - fmin) / (c.γ * T) +
        4 * c.momentumWeight * ‖gradient f x₀‖ ^ 2 / T + 4 * c.noise * σ ^ 2 := by
  have h := finite_time c τ O hg hlip hmin hdual hCU hdown hsmall T (initial x₀)
  rw [expectedTotal_mul, initial_potential] at h
  have hT' : (0 : ℝ) < T := by exact_mod_cast hT
  apply (div_le_iff₀ hT').mpr
  have hγ := c.γ_pos
  have hbound :
      (4 * (f x₀ - fmin) / (c.γ * T) +
        4 * c.momentumWeight * ‖gradient f x₀‖ ^ 2 / T + 4 * c.noise * σ ^ 2) * T =
      4 / c.γ * (f x₀ - fmin + c.γ * c.momentumWeight * ‖gradient f x₀‖ ^ 2 +
        T * c.γ * c.noise * σ ^ 2) := by
    field_simp
  rw [hbound]
  have hh := mul_le_mul_of_nonneg_left h (by positivity : 0 ≤ 4 / c.γ)
  have hcancel : 4 / c.γ * (c.γ / 4) = 1 := by field_simp
  nlinarith only [hh, congrArg (fun r : ℝ => r *
    expectedTotal p (advance c O.reply CU CP) (fun s => nuclearNorm (gradient f s.x) ^ 2)
      T (initial x₀)) hcancel]

end EF21Muon.Stochastic
