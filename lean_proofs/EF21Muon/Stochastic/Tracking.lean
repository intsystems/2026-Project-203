import EF21Muon.Protocol
import EF21Muon.Stochastic.FiniteSampling
import EF21Muon.Stochastic.SquaredNorm
import LongTails.Proofs.MatrixSmooth

noncomputable section

namespace EF21Muon.Stochastic

open LongTails.Algorithms.MatrixPolar
open LongTails.Proofs.MatrixSmooth

variable {m n : ℕ} {Ω : Type*} [Fintype Ω]

structure Parameters where
  β : ℝ
  γ : ℝ
  L : ℝ
  αU : ℝ
  αP : ℝ
  ρ : ℝ
  β_pos : 0 < β
  β_le_one : β ≤ 1
  γ_pos : 0 < γ
  L_nonneg : 0 ≤ L
  αU_pos : 0 < αU
  αU_le_one : αU ≤ 1
  αP_pos : 0 < αP
  αP_le_one : αP ≤ 1
  ρ_nonneg : 0 ≤ ρ

def SquaredContracts (C : Space m n → Space m n) (α : ℝ) : Prop :=
  ∀ z, ‖C z - z‖ ^ 2 ≤ (1 - α) * ‖z‖ ^ 2

def serverPoint (γ : ℝ) (s : State m n) : Space m n :=
  s.x - (γ * nuclearNorm s.estimator) • polar s.estimator

def advance (c : Parameters) (oracle : Space m n → Ω → Space m n)
    (CU CP : Space m n → Space m n) (s : State m n) (ω : Ω) : State m n :=
  replyStep c.β c.γ CU CP s (oracle (queryPoint c.γ CP s) ω)

def momentumSq (grad : Space m n → Space m n) (s : State m n) : ℝ :=
  ‖s.momentum - grad s.x‖ ^ 2

def uplinkSq (s : State m n) : ℝ := ‖s.momentum - s.estimator‖ ^ 2

def downlinkSq (s : State m n) : ℝ := ‖s.w - s.x‖ ^ 2

theorem movement_spectral (c : Parameters) (s : State m n) :
    spectralNorm (serverPoint c.γ s - s.x) ≤ c.γ * nuclearNorm s.estimator := by
  have he : serverPoint c.γ s - s.x = -(c.γ * nuclearNorm s.estimator) • polar s.estimator := by
    dsimp [serverPoint]; module
  rw [he, spectralNorm_smul, abs_neg, abs_of_nonneg (mul_nonneg c.γ_pos.le (nuclearNorm_nonneg _))]
  simpa using mul_le_mul_of_nonneg_left (polar_mem s.estimator)
    (mul_nonneg c.γ_pos.le (nuclearNorm_nonneg _))

theorem movement_frobenius (c : Parameters)
    (hpolar : ∀ z : Space m n, ‖polar z‖ ≤ c.ρ) (s : State m n) :
    ‖serverPoint c.γ s - s.x‖ ≤ c.ρ * c.γ * nuclearNorm s.estimator := by
  have he : serverPoint c.γ s - s.x = -(c.γ * nuclearNorm s.estimator) • polar s.estimator := by
    dsimp [serverPoint]; module
  rw [he, norm_smul, Real.norm_eq_abs, abs_neg,
    abs_of_nonneg (mul_nonneg c.γ_pos.le (nuclearNorm_nonneg _))]
  nlinarith only [mul_le_mul_of_nonneg_left (hpolar s.estimator)
    (mul_nonneg c.γ_pos.le (nuclearNorm_nonneg s.estimator))]

theorem gradient_movement_sq (c : Parameters) {grad : Space m n → Space m n}
    (hlip : ∀ x y, nuclearNorm (grad x - grad y) ≤ c.L * spectralNorm (x - y))
    (s : State m n) :
    ‖grad (serverPoint c.γ s) - grad s.x‖ ^ 2 ≤
      c.L ^ 2 * c.γ ^ 2 * nuclearNorm s.estimator ^ 2 := by
  have h := (frobenius_le_nuclear _).trans ((hlip _ _).trans
    (mul_le_mul_of_nonneg_left (movement_spectral c s) c.L_nonneg))
  have hs := mul_self_le_mul_self (norm_nonneg _) h
  nlinarith only [hs]

theorem gradient_broadcast_sq (c : Parameters) {grad : Space m n → Space m n}
    (hlip : ∀ x y, nuclearNorm (grad x - grad y) ≤ c.L * spectralNorm (x - y))
    (CP : Space m n → Space m n) (s : State m n) :
    ‖grad (queryPoint c.γ CP s) - grad (serverPoint c.γ s)‖ ^ 2 ≤
      c.L ^ 2 * ‖queryPoint c.γ CP s - serverPoint c.γ s‖ ^ 2 := by
  have h := (frobenius_le_nuclear (grad (queryPoint c.γ CP s) - grad (serverPoint c.γ s))).trans
    ((hlip (queryPoint c.γ CP s) (serverPoint c.γ s)).trans
      (mul_le_mul_of_nonneg_left
        (spectralNorm_le_frobenius (queryPoint c.γ CP s - serverPoint c.γ s)) c.L_nonneg))
  have hs := mul_self_le_mul_self (norm_nonneg _) h
  nlinarith only [hs]

theorem downlink_rec (c : Parameters) {CP : Space m n → Space m n}
    (hCP : SquaredContracts CP c.αP)
    (hpolar : ∀ z : Space m n, ‖polar z‖ ≤ c.ρ) (s : State m n) :
    ‖queryPoint c.γ CP s - serverPoint c.γ s‖ ^ 2 ≤
      (1 - c.αP / 2) * downlinkSq s +
        2 * c.ρ ^ 2 / c.αP * c.γ ^ 2 * nuclearNorm s.estimator ^ 2 := by
  have hα := c.αP_pos
  have he : queryPoint c.γ CP s - serverPoint c.γ s =
      CP (serverPoint c.γ s - s.w) - (serverPoint c.γ s - s.w) := by
    dsimp [queryPoint, serverPoint]; module
  rw [he]
  have hc := hCP (serverPoint c.γ s - s.w)
  rw [norm_sub_rev (serverPoint c.γ s) s.w] at hc
  have hsum : s.w - serverPoint c.γ s = (s.w - s.x) + (s.x - serverPoint c.γ s) := by module
  rw [hsum] at hc
  have hy := contracted_add_sq (s.w - s.x) (s.x - serverPoint c.γ s) c.αP_pos c.αP_le_one
  rw [norm_sub_rev s.x] at hy
  have hm := mul_self_le_mul_self (norm_nonneg _) (movement_frobenius c hpolar s)
  have hm' := mul_le_mul_of_nonneg_left hm (by positivity : 0 ≤ 2 / c.αP)
  unfold downlinkSq
  simp only [div_eq_mul_inv] at hy hm' ⊢
  nlinarith only [hc, hy, hm']

/-- A certified bound on the actual broadcast update, used by the generic Lyapunov lemma. -/
def BroadcastBound (c : Parameters) (CP : Space m n → Space m n) (τ : ℝ) : Prop :=
  ∀ s, ‖queryPoint c.γ CP s - serverPoint c.γ s‖ ^ 2 ≤
    (1 - c.αP / 2) * downlinkSq s + 2 * τ / c.αP * c.γ ^ 2 * nuclearNorm s.estimator ^ 2

theorem identity_broadcast_bound (c : Parameters) :
    BroadcastBound (m := m) (n := n) c id 0 := by
  intro s
  have he : queryPoint c.γ id s = serverPoint c.γ s := by
    dsimp [queryPoint, serverPoint]; module
  rw [he, sub_self, norm_zero]
  have hp : 0 ≤ 1 - c.αP / 2 := by linarith [c.αP_le_one]
  simpa only [downlinkSq, zero_pow (by norm_num : (2 : ℕ) ≠ 0), mul_zero, zero_div, zero_mul, add_zero]
    using mul_nonneg hp (sq_nonneg ‖s.w - s.x‖)

/-- The paper's squared momentum recurrence, including the beta-squared noise term. -/
theorem momentum_rec (c : Parameters) {p : Sampling Ω} {grad : Space m n → Space m n}
    {σ : ℝ} (O : Oracle p grad σ)
    (hlip : ∀ x y, nuclearNorm (grad x - grad y) ≤ c.L * spectralNorm (x - y))
    (CU CP : Space m n → Space m n) (s : State m n) :
    p.expect (fun ω => momentumSq grad (advance c O.reply CU CP s ω)) ≤
      (1 - c.β / 2) * momentumSq grad s +
        2 * c.L ^ 2 / c.β * c.γ ^ 2 * nuclearNorm s.estimator ^ 2 +
        c.β ^ 2 * (1 + 2 / c.β) * c.L ^ 2 *
          ‖queryPoint c.γ CP s - serverPoint c.γ s‖ ^ 2 + c.β ^ 2 * σ ^ 2 := by
  have hβ := c.β_pos
  let A := s.momentum - grad s.x
  let B := grad s.x - grad (serverPoint c.γ s)
  let C := grad (queryPoint c.γ CP s) - grad (serverPoint c.γ s)
  let mean := (1 - c.β) • (A + B) + c.β • C
  have he (ω : Ω) : (advance c O.reply CU CP s ω).momentum -
      grad (advance c O.reply CU CP s ω).x =
        mean + c.β • (O.reply (queryPoint c.γ CP s) ω - grad (queryPoint c.γ CP s)) := by
    dsimp [advance, replyStep, serverPoint, mean, A, B, C]; module
  have hn := O.affine_variance (queryPoint c.γ CP s) mean c.β
  simp only [momentumSq, he]
  have hy := momentum_mean_sq A B C c.β_pos c.β_le_one
  have hB := gradient_movement_sq c hlip s
  rw [norm_sub_rev] at hB
  have hC := gradient_broadcast_sq c hlip CP s
  have hB' := mul_le_mul_of_nonneg_left hB (by positivity : 0 ≤ 2 / c.β)
  have hC' := mul_le_mul_of_nonneg_left hC
    (by have := c.β_pos; positivity : 0 ≤ c.β ^ 2 * (1 + 2 / c.β))
  change _ ≤ ‖mean‖ ^ 2 + _ at hn
  change ‖mean‖ ^ 2 ≤ _ at hy
  dsimp [A, B, C] at hy
  simp only [div_eq_mul_inv] at hB' hC' hy ⊢
  nlinarith only [hn, hy, hB', hC']

/-- The uplink recurrence uses unbiasedness before applying compressor contraction. -/
theorem uplink_rec (c : Parameters) {p : Sampling Ω} {grad : Space m n → Space m n}
    {σ : ℝ} (O : Oracle p grad σ)
    (hlip : ∀ x y, nuclearNorm (grad x - grad y) ≤ c.L * spectralNorm (x - y))
    {CU : Space m n → Space m n} (hCU : SquaredContracts CU c.αU)
    (CP : Space m n → Space m n) (s : State m n) :
    p.expect (fun ω => uplinkSq (advance c O.reply CU CP s ω)) ≤
      (1 - c.αU / 2) * uplinkSq s + 6 * c.β ^ 2 / c.αU * momentumSq grad s +
        6 * c.β ^ 2 * c.L ^ 2 / c.αU * c.γ ^ 2 * nuclearNorm s.estimator ^ 2 +
        6 * c.β ^ 2 * c.L ^ 2 / c.αU *
          ‖queryPoint c.γ CP s - serverPoint c.γ s‖ ^ 2 +
        (1 - c.αU) * c.β ^ 2 * σ ^ 2 := by
  have hα := c.αU_pos
  let mean := (1 - c.β) • s.momentum + c.β • grad (queryPoint c.γ CP s) - s.estimator
  have he (ω : Ω) :
      (advance c O.reply CU CP s ω).estimator - (advance c O.reply CU CP s ω).momentum =
      CU ((advance c O.reply CU CP s ω).momentum - s.estimator) -
        ((advance c O.reply CU CP s ω).momentum - s.estimator) := by
    dsimp [advance, replyStep]; module
  have hnorm (ω : Ω) : uplinkSq (advance c O.reply CU CP s ω) =
      ‖CU ((advance c O.reply CU CP s ω).momentum - s.estimator) -
        ((advance c O.reply CU CP s ω).momentum - s.estimator)‖ ^ 2 := by
    rw [uplinkSq, norm_sub_rev, he]
  have hcontract : p.expect (fun ω => uplinkSq (advance c O.reply CU CP s ω)) ≤
      p.expect (fun ω => (1 - c.αU) *
        ‖(advance c O.reply CU CP s ω).momentum - s.estimator‖ ^ 2) := by
    apply p.expect_mono
    intro ω
    rw [hnorm]
    exact hCU _
  have hm (ω : Ω) : (advance c O.reply CU CP s ω).momentum - s.estimator =
      mean + c.β • (O.reply (queryPoint c.γ CP s) ω - grad (queryPoint c.γ CP s)) := by
    dsimp [advance, replyStep, mean]; module
  simp only [Sampling.expect_mul, hm] at hcontract
  have hn := mul_le_mul_of_nonneg_left
    (O.affine_variance (queryPoint c.γ CP s) mean c.β) (sub_nonneg.mpr c.αU_le_one)
  have hmean : mean = (s.momentum - s.estimator) +
      c.β • (grad (queryPoint c.γ CP s) - s.momentum) := by dsimp [mean]; module
  have hy := contracted_add_sq (s.momentum - s.estimator)
    (c.β • (grad (queryPoint c.γ CP s) - s.momentum)) c.αU_pos c.αU_le_one
  rw [← hmean, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs] at hy
  have hthree := three_sq (grad (queryPoint c.γ CP s) - grad (serverPoint c.γ s))
    (grad (serverPoint c.γ s) - grad s.x) (grad s.x - s.momentum)
  have hcancel : grad (queryPoint c.γ CP s) - grad (serverPoint c.γ s) +
      (grad (serverPoint c.γ s) - grad s.x) + (grad s.x - s.momentum) =
      grad (queryPoint c.γ CP s) - s.momentum := by module
  rw [hcancel, norm_sub_rev (grad s.x) s.momentum] at hthree
  have hb := gradient_broadcast_sq c hlip CP s
  have hmov := gradient_movement_sq c hlip s
  have hs : ‖grad (queryPoint c.γ CP s) - s.momentum‖ ^ 2 ≤
      3 * (c.L ^ 2 * ‖queryPoint c.γ CP s - serverPoint c.γ s‖ ^ 2 +
        c.L ^ 2 * c.γ ^ 2 * nuclearNorm s.estimator ^ 2 + momentumSq grad s) := by
    dsimp [momentumSq]; linarith only [hthree, hb, hmov]
  have hs' := mul_le_mul_of_nonneg_left hs (by positivity : 0 ≤ 2 / c.αU * c.β ^ 2)
  change _ ≤ (1 - c.αU / 2) * uplinkSq s + _ at hy
  simp only [div_eq_mul_inv] at hy hs' ⊢
  nlinarith only [hcontract, hn, hy, hs']

end EF21Muon.Stochastic
