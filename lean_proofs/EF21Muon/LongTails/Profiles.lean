import EF21Muon.LongTails.Polynomial
import EF21Muon.Nesterov.SmoothPaper

noncomputable section

namespace EF21Muon.LongTails

open MeasureTheory
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def emaProfile (P : Problem N m n μ c) {g : ℝ} (hg : 0 < g) (x : Model m n) : Profile where
  a := (P.f x - P.fmin) / g + ∑ i, 6 * (c i).ρ ^ 2 * ‖P.grad x i‖ ^ 2
  h := ∑ i, 72 * (c i).ρ ^ 2 / (c i).αU ^ 2 *
    average (fun j => ‖P.localGrad j x i‖ ^ 2)
  s₁ := ∑ i, 6 * (c i).ρ ^ 2 / N * P.σ i ^ 2
  s₂ := ∑ i, 6 * (c i).ρ ^ 2 * (1 - (c i).αU) / (c i).αU * P.σ i ^ 2
  s₃ := ∑ i, 72 * (c i).ρ ^ 2 / (c i).αU ^ 2 * P.σ i ^ 2
  a_nonneg := add_nonneg (div_nonneg (sub_nonneg.mpr (P.lower_bound x)) hg.le)
    (Finset.sum_nonneg (fun i _ => by positivity))
  h_nonneg := Finset.sum_nonneg (fun i _ => mul_nonneg (by positivity)
    (average_nonneg (fun _ => sq_nonneg _)))
  s₁_nonneg := Finset.sum_nonneg (fun i _ => by positivity)
  s₂_nonneg := Finset.sum_nonneg (fun i _ =>
    mul_nonneg (div_nonneg (mul_nonneg (by positivity)
      (sub_nonneg.mpr (c i).αU_le_one)) (c i).αU_pos.le) (sq_nonneg _))
  s₃_nonneg := Finset.sum_nonneg (fun i _ => by positivity)

theorem ema_initial_profile (P : Problem N m n μ c) {g β : ℝ}
    (hg : 0 < g) (hβpos : 0 < β) (hβ : ∀ i, (c i).β = β)
    (hγ : ∀ i, (c i).γ = g * β) (x : Model m n) :
    P.potential (Distributed.start x) =
      g * ((emaProfile P hg x).a + (emaProfile P hg x).h * β ^ 2) := by
  rw [P.initial_potential]
  have hterm i : (c i).γ *
      (globalWeight (c i) * ‖P.grad x i‖ ^ 2 +
        localWeight (c i) * average (fun j => ‖P.localGrad j x i‖ ^ 2)) =
      g * (6 * (c i).ρ ^ 2 * ‖P.grad x i‖ ^ 2) +
        g * β ^ 2 * (72 * (c i).ρ ^ 2 / (c i).αU ^ 2 *
          average (fun j => ‖P.localGrad j x i‖ ^ 2)) := by
    rw [hγ i]
    unfold globalWeight localWeight
    rw [hβ i]
    field_simp <;> ring
  simp_rw [hterm]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  dsimp [emaProfile]
  field_simp
  ring

theorem ema_noise_profile (P : Problem N m n μ c) {g β : ℝ}
    (hg : 0 < g) (hβ : ∀ i, (c i).β = β) (hγ : ∀ i, (c i).γ = g * β)
    (x : Model m n) :
    P.noiseTotal = g * β * ((emaProfile P hg x).s₁ * β +
      (emaProfile P hg x).s₂ * β ^ 2 + (emaProfile P hg x).s₃ * β ^ 3) := by
  dsimp [Problem.noiseTotal, emaProfile]
  simp only [Finset.mul_sum, Finset.sum_mul, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  rw [hγ i]
  unfold Distributed.noise
  rw [hβ i]
  simp only [div_eq_mul_inv]
  ring

theorem ema_polynomial_bound (P : Problem N m n μ c) (A : Scheme N m n c)
    {g β : ℝ} (hg : 0 < g) (hβpos : 0 < β) (hβ : ∀ i, (c i).β = β)
    (hγ : ∀ i, (c i).γ = g * β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Distributed.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) (T : ℕ) (hT : 0 < T) :
    P.stationarity A (g * β) (Distributed.start x) T ≤ (emaProfile P hg x).bound T β := by
  have hγ' i : (c i).γ = g * (c i).β := by rw [hβ i]; exact hγ i
  have hstep i := (c i).step_condition_for
    (Distributed.zeta_nonneg _ (P.ell2_nonneg i) (A.τ_nonneg i))
    (Distributed.rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i) hg.le (hγ' i) (hsmall i))
  have h := P.average_stationarity A hstep (mul_pos hg hβpos) (Distributed.start x) T hT
  rw [ema_initial_profile P hg hβpos hβ hγ x, ema_noise_profile P hg hβ hγ x] at h
  have he : 4 * (g * ((emaProfile P hg x).a + (emaProfile P hg x).h * β ^ 2)) /
      (g * β * T) + 4 * (g * β * ((emaProfile P hg x).s₁ * β +
        (emaProfile P hg x).s₂ * β ^ 2 + (emaProfile P hg x).s₃ * β ^ 3)) / (g * β) =
      (emaProfile P hg x).bound T β := by unfold Profile.bound; field_simp; ring
  exact h.trans_eq he

def nesterovProfile (P : Problem N m n μ c) {g : ℝ} (hg : 0 < g) (x : Model m n) : Profile where
  a := (P.f x - P.fmin) / g + ∑ i, 2 * Nesterov.dWeight (c i) *
    average (fun j => ‖P.localGrad j x i‖ ^ 2)
  h := ∑ i, 6 * Nesterov.rWeight (c i) * average (fun j => ‖P.localGrad j x i‖ ^ 2)
  s₁ := ∑ i, 2 * Nesterov.dWeight (c i) * P.σ i ^ 2
  s₂ := ∑ i, Nesterov.rWeight (c i) * P.σ i ^ 2
  s₃ := ∑ i, 6 * Nesterov.rWeight (c i) * P.σ i ^ 2
  a_nonneg := add_nonneg (div_nonneg (sub_nonneg.mpr (P.lower_bound x)) hg.le)
    (Finset.sum_nonneg (fun i _ => mul_nonneg (by unfold Nesterov.dWeight; positivity)
      (average_nonneg (fun _ => sq_nonneg _))))
  h_nonneg := Finset.sum_nonneg (fun i _ => mul_nonneg
    (mul_nonneg (by norm_num) (Nesterov.weights_nonneg (c i) (le_refl (0 : ℝ))).2.2.2.1)
    (average_nonneg (fun _ => sq_nonneg _)))
  s₁_nonneg := Finset.sum_nonneg (fun i _ => by unfold Nesterov.dWeight; positivity)
  s₂_nonneg := Finset.sum_nonneg (fun i _ => mul_nonneg
    (Nesterov.weights_nonneg (c i) (le_refl (0 : ℝ))).2.2.2.1 (sq_nonneg _))
  s₃_nonneg := Finset.sum_nonneg (fun i _ => mul_nonneg
    (mul_nonneg (by norm_num) (Nesterov.weights_nonneg (c i) (le_refl (0 : ℝ))).2.2.2.1)
    (sq_nonneg _))

theorem nesterov_initial_profile (P : Problem N m n μ c) {g β : ℝ}
    (hg : 0 < g) (hβpos : 0 < β) (hβ : ∀ i, (c i).β = β)
    (hγ : ∀ i, (c i).γ = g * β) (x : Model m n) :
    Nesterov.smoothPotential P (Nesterov.initial x) =
      g * ((nesterovProfile P hg x).a + (nesterovProfile P hg x).h * β ^ 2) := by
  have he : Nesterov.smoothPotential P (Nesterov.initial x) = P.f x - P.fmin +
      ∑ i, (c i).γ * Nesterov.fWeight (c i) * average (fun j => ‖P.localGrad j x i‖ ^ 2) := by
    have hx : Nesterov.points (Nesterov.initial x : Nesterov.States N m n) = x := rfl
    simp only [Nesterov.smoothPotential, hx]
    congr 1
    apply Finset.sum_congr rfl
    intro i _
    simp [Nesterov.initial, Nesterov.start, LayerState.start, Nesterov.energy, localError,
      Nesterov.compression, Nesterov.increment, broadcastError, Nesterov.target,
      LayerState.M, LayerState.G, LayerState.x, LayerState.w, Probability.average, mul_assoc]
  rw [he]
  have hterm i : (c i).γ * Nesterov.fWeight (c i) *
      average (fun j => ‖P.localGrad j x i‖ ^ 2) =
      g * (2 * Nesterov.dWeight (c i) * average (fun j => ‖P.localGrad j x i‖ ^ 2)) +
        g * β ^ 2 * (6 * Nesterov.rWeight (c i) * average (fun j => ‖P.localGrad j x i‖ ^ 2)) := by
    rw [hγ i]
    unfold Nesterov.fWeight
    rw [hβ i]
    field_simp <;> ring
  simp_rw [hterm]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  dsimp [nesterovProfile]
  field_simp
  ring

theorem nesterov_noise_profile (P : Problem N m n μ c) {g β : ℝ}
    (hg : 0 < g) (hβpos : 0 < β) (hβ : ∀ i, (c i).β = β)
    (hγ : ∀ i, (c i).γ = g * β) (x : Model m n) :
    Nesterov.smoothNoise P = g * β * ((nesterovProfile P hg x).s₁ * β +
      (nesterovProfile P hg x).s₂ * β ^ 2 + (nesterovProfile P hg x).s₃ * β ^ 3) := by
  dsimp [Nesterov.smoothNoise, nesterovProfile]
  simp only [Finset.mul_sum, Finset.sum_mul, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  rw [hγ i]
  unfold Nesterov.noise Nesterov.fWeight
  rw [hβ i]
  field_simp
  ring

theorem nesterov_polynomial_bound (P : Problem N m n μ c) (A : Scheme N m n c)
    {g β : ℝ} (hg : 0 < g) (hβpos : 0 < β) (hβ : ∀ i, (c i).β = β)
    (hγ : ∀ i, (c i).γ = g * β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Nesterov.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) (T : ℕ) (hT : 0 < T) :
    Nesterov.smoothStationarity P A (g * β) (Nesterov.initial x) T ≤
      (nesterovProfile P hg x).bound T β := by
  have hγ' i : (c i).γ = g * (c i).β := by rw [hβ i]; exact hγ i
  have hstep i := Nesterov.rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i)
    hg.le (hγ' i) (hsmall i)
  have h := Nesterov.smooth_average P A hstep (mul_pos hg hβpos) (Nesterov.initial x) T hT
  rw [nesterov_initial_profile P hg hβpos hβ hγ x,
    nesterov_noise_profile P hg hβpos hβ hγ x] at h
  have he : 4 * (g * ((nesterovProfile P hg x).a + (nesterovProfile P hg x).h * β ^ 2)) /
      (g * β * T) + 4 * (g * β * ((nesterovProfile P hg x).s₁ * β +
        (nesterovProfile P hg x).s₂ * β ^ 2 + (nesterovProfile P hg x).s₃ * β ^ 3)) / (g * β) =
      (nesterovProfile P hg x).bound T β := by unfold Profile.bound; field_simp; ring
  exact h.trans_eq he

def Profile.doubleNoise (P : Profile) : Profile where
  a := P.a
  h := P.h
  s₁ := 2*P.s₁
  s₂ := 2*P.s₂
  s₃ := 2*P.s₃
  a_nonneg := P.a_nonneg
  h_nonneg := P.h_nonneg
  s₁_nonneg := by linarith [P.s₁_nonneg]
  s₂_nonneg := by linarith [P.s₂_nonneg]
  s₃_nonneg := by linarith [P.s₃_nonneg]

theorem nesterov_paper_polynomial_bound (P : Problem N m n μ c) (A : Scheme N m n c)
    (hCP : ∀ i, A.CP i 0 = 0)
    {g β : ℝ} (hg : 0 < g) (hβpos : 0 < β) (hβ : ∀ i, (c i).β = β)
    (hγ : ∀ i, (c i).γ = g * β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Nesterov.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) (T : ℕ) (hT : 0 < T) :
    Nesterov.paperSmoothAverage P A (g * β) x T ≤
      (nesterovProfile P hg x).doubleNoise.bound T β := by
  have hγ' i : (c i).γ = g * (c i).β := by rw [hβ i]; exact hγ i
  have hstep i := Nesterov.rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i)
    hg.le (hγ' i) (hsmall i)
  have h := (Nesterov.smooth_paper_average P A hCP hstep (mul_pos hg hβpos) x T hT).2
  rw [nesterov_initial_profile P hg hβpos hβ hγ x,
    nesterov_noise_profile P hg hβpos hβ hγ x] at h
  have he : 4 * (g * ((nesterovProfile P hg x).a + (nesterovProfile P hg x).h * β ^ 2)) /
      (g * β * T) + 8 * (g * β * ((nesterovProfile P hg x).s₁ * β +
        (nesterovProfile P hg x).s₂ * β ^ 2 + (nesterovProfile P hg x).s₃ * β ^ 3)) / (g * β) =
      (nesterovProfile P hg x).doubleNoise.bound T β := by
    unfold Profile.bound Profile.doubleNoise
    field_simp
    ring
  exact h.trans_eq he

end EF21Muon.LongTails
