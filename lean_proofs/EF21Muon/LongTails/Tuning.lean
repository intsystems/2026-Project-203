import EF21Muon.Distributed.Rates
import EF21Muon.Nesterov.SmoothRates

noncomputable section

namespace EF21Muon.LongTails

open MeasureTheory
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open scoped BigOperators

def accuracyBeta (S ε : ℝ) : ℝ := ε ^ 2 / (ε ^ 2 + 8 * S)

theorem accuracyBeta_pos {S ε : ℝ} (hS : 0 ≤ S) (hε : 0 < ε) :
    0 < accuracyBeta S ε := by
  unfold accuracyBeta
  exact div_pos (sq_pos_of_pos hε) (by positivity)

theorem accuracyBeta_le_one {S ε : ℝ} (hS : 0 ≤ S) (hε : 0 < ε) :
    accuracyBeta S ε ≤ 1 := by
  unfold accuracyBeta
  exact (div_le_one (by positivity : 0 < ε ^ 2 + 8 * S)).mpr (by linarith)

theorem accuracyBeta_noise {S ε : ℝ} (hS : 0 ≤ S) (hε : 0 < ε) :
    4 * accuracyBeta S ε * S ≤ ε ^ 2 / 2 := by
  have hd : 0 < ε ^ 2 + 8 * S := by positivity
  have he : accuracyBeta S ε * (ε ^ 2 + 8 * S) = ε ^ 2 :=
    div_mul_cancel₀ _ hd.ne'
  have hn := mul_nonneg (accuracyBeta_pos hS hε).le (sq_nonneg ε)
  nlinarith only [he, hn]

theorem accuracyBeta_zero {ε : ℝ} (hε : 0 < ε) : accuracyBeta 0 ε = 1 := by
  simp [accuracyBeta, ne_of_gt hε]

def accuracyBudget (B g S ε : ℝ) : ℕ :=
  max 1 ⌈8 * B / (g * ε ^ 2) + 64 * B * S / (g * ε ^ 4)⌉₊

theorem accuracyBudget_formula (B g S ε : ℝ) :
    accuracyBudget B g S ε =
      max 1 ⌈8 * B / (g * ε ^ 2) + 64 * B * S / (g * ε ^ 4)⌉₊ := rfl

theorem accuracyBudget_eq {B g S ε : ℝ} (hg : 0 < g) (hS : 0 ≤ S) (hε : 0 < ε) :
    accuracyBudget B g S ε = stochasticBudget B (g * accuracyBeta S ε) ε := by
  unfold accuracyBudget stochasticBudget accuracyBeta
  congr 2
  have hd : ε ^ 2 + 8 * S ≠ 0 := ne_of_gt (by positivity : 0 < ε ^ 2 + 8 * S)
  field_simp
  ring

theorem accuracyBudget_pos (B g S ε : ℝ) : 0 < accuracyBudget B g S ε :=
  lt_of_lt_of_le (by norm_num) (le_max_left _ _)

theorem accuracyBudget_zero (B g ε : ℝ) :
    accuracyBudget B g 0 ε = max 1 ⌈8 * B / (g * ε ^ 2)⌉₊ := by
  simp [accuracyBudget]

theorem bound_at_accuracyBudget {value V B noise g S ε : ℝ}
    (hg : 0 < g) (hS : 0 ≤ S) (hε : 0 < ε) (hV : V ≤ B)
    (hn : noise ≤ g * accuracyBeta S ε * (accuracyBeta S ε * S))
    (h : value ≤ 4 * V / (g * accuracyBeta S ε * accuracyBudget B g S ε) +
      4 * noise / (g * accuracyBeta S ε)) : value ≤ ε ^ 2 := by
  have hγ := mul_pos hg (accuracyBeta_pos hS hε)
  have ht := budget_transient (Ψ := B) hγ hε
  rw [← accuracyBudget_eq hg hS hε] at ht
  have hi := div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_left hV (by norm_num : (0 : ℝ) ≤ 4))
    (mul_nonneg hγ.le (Nat.cast_nonneg (accuracyBudget B g S ε)))
  have hn' := div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_left hn (by norm_num : (0 : ℝ) ≤ 4)) hγ.le
  have he : 4 * (g * accuracyBeta S ε * (accuracyBeta S ε * S)) /
      (g * accuracyBeta S ε) = 4 * accuracyBeta S ε * S := by
    field_simp
  rw [he] at hn'
  linarith only [h, hi, ht, hn', accuracyBeta_noise hS hε]

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

theorem ema_accuracy (P : Problem N m n μ c) (A : Scheme N m n c)
    {g ε : ℝ} (hg : 0 < g) (hε : 0 < ε)
    (hβ : ∀ i, (c i).β = accuracyBeta P.noiseScaleTotal ε)
    (hγ : ∀ i, (c i).γ = g * (c i).β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Distributed.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) :
    P.stationarity A (g * accuracyBeta P.noiseScaleTotal ε) (Distributed.start x)
      (accuracyBudget (P.rateInitial g x) g P.noiseScaleTotal ε) ≤ ε ^ 2 := by
  let γ := g * accuracyBeta P.noiseScaleTotal ε
  have hγpos : 0 < γ := mul_pos hg (accuracyBeta_pos P.noiseScaleTotal_nonneg hε)
  have hsteps i : (c i).γ = γ := by rw [hγ i, hβ i]
  have hstep i := (c i).step_condition_for
    (Distributed.zeta_nonneg _ (P.ell2_nonneg i) (A.τ_nonneg i))
    (Distributed.rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i) hg.le (hγ i) (hsmall i))
  have hn : P.noiseTotal ≤ γ * (accuracyBeta P.noiseScaleTotal ε * P.noiseScaleTotal) := by
    simp only [Problem.noiseTotal, hsteps, mul_assoc, ← Finset.mul_sum]
    apply mul_le_mul_of_nonneg_left _ hγpos.le
    change _ ≤ accuracyBeta P.noiseScaleTotal ε *
      ∑ i, Distributed.noiseScale (c i) N * P.σ i ^ 2
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun i _ => by
      simpa only [hβ i, mul_assoc] using
        mul_le_mul_of_nonneg_right (Distributed.noise_le (c i) N) (sq_nonneg (P.σ i))
  exact bound_at_accuracyBudget hg P.noiseScaleTotal_nonneg hε
    (P.initial_potential_le_rateInitial hg.le hγ x) hn
    (P.average_stationarity A hstep hγpos (Distributed.start x) _ (accuracyBudget_pos _ _ _ _))

omit [IsProbabilityMeasure μ] in
theorem nesterovScale_nonneg (P : Problem N m n μ c) : 0 ≤ Nesterov.noiseTotalScale P := by
  apply Finset.sum_nonneg
  intro i _
  rcases Nesterov.weights_nonneg (c i) (le_refl (0 : ℝ)) with ⟨hd, _, _, hr, _⟩
  exact mul_nonneg (by dsimp [Nesterov.noiseScale]; positivity) (sq_nonneg (P.σ i))

theorem nesterov_accuracy (P : Problem N m n μ c) (A : Scheme N m n c)
    {g ε : ℝ} (hg : 0 < g) (hε : 0 < ε)
    (hβ : ∀ i, (c i).β = accuracyBeta (Nesterov.noiseTotalScale P) ε)
    (hγ : ∀ i, (c i).γ = g * (c i).β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Nesterov.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) :
    Nesterov.smoothStationarity P A (g * accuracyBeta (Nesterov.noiseTotalScale P) ε)
      (Nesterov.initial x)
      (accuracyBudget (Nesterov.rateInitial P g x) g (Nesterov.noiseTotalScale P) ε) ≤ ε ^ 2 := by
  let γ := g * accuracyBeta (Nesterov.noiseTotalScale P) ε
  have hS := nesterovScale_nonneg P
  have hγpos : 0 < γ := mul_pos hg (accuracyBeta_pos hS hε)
  have hsteps i : (c i).γ = γ := by rw [hγ i, hβ i]
  have hstep i := Nesterov.rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i)
    hg.le (hγ i) (hsmall i)
  have hn : Nesterov.smoothNoise P ≤
      γ * (accuracyBeta (Nesterov.noiseTotalScale P) ε * Nesterov.noiseTotalScale P) := by
    simp only [Nesterov.smoothNoise, hsteps, mul_assoc, ← Finset.mul_sum]
    apply mul_le_mul_of_nonneg_left _ hγpos.le
    change _ ≤ accuracyBeta (Nesterov.noiseTotalScale P) ε *
      ∑ i, Nesterov.noiseScale (c i) * P.σ i ^ 2
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun i _ => by
      simpa only [hβ i, mul_assoc] using
        mul_le_mul_of_nonneg_right (Nesterov.noise_bound (c i)) (sq_nonneg (P.σ i))
  exact bound_at_accuracyBudget hg hS hε
    (Nesterov.initial_potential_bound P hg.le hγ x) hn
    (Nesterov.smooth_average P A hstep hγpos (Nesterov.initial x) _ (accuracyBudget_pos _ _ _ _))

end EF21Muon.LongTails
