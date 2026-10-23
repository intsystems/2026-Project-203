import EF21Muon.LongTails.Profiles

noncomputable section

namespace EF21Muon.LongTails

open MeasureTheory
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open scoped BigOperators

theorem Profile.collapsedBudget_eq_accuracy (P : Profile) {g : ℝ} (hg : 0 < g) (ε : ℝ) :
    P.collapsedBudget ε = accuracyBudget (g*(P.a+P.h)) g (P.s₁+P.s₂+P.s₃) ε := by
  unfold collapsedBudget accuracyBudget
  congr 2
  field_simp <;> ring

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

theorem ema_profile_initial_total (P : Problem N m n μ c) {g : ℝ}
    (hg : 0 < g) (x : Model m n) :
    g*((emaProfile P hg x).a+(emaProfile P hg x).h) = P.rateInitial g x := by
  dsimp [emaProfile, Problem.rateInitial]
  rw [Finset.sum_add_distrib]
  field_simp
  ring

theorem ema_profile_noise_total (P : Problem N m n μ c) {g : ℝ}
    (hg : 0 < g) (x : Model m n) :
    (emaProfile P hg x).s₁+(emaProfile P hg x).s₂+(emaProfile P hg x).s₃ =
      P.noiseScaleTotal := by
  dsimp [emaProfile, Problem.noiseScaleTotal, Distributed.noiseScale]
  simp only [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  simp only [div_eq_mul_inv]
  ring

theorem nesterov_profile_initial_total (P : Problem N m n μ c) {g : ℝ}
    (hg : 0 < g) (x : Model m n) :
    g*((nesterovProfile P hg x).a+(nesterovProfile P hg x).h) = Nesterov.rateInitial P g x := by
  dsimp [nesterovProfile, Nesterov.rateInitial]
  simp only [add_mul, Finset.sum_add_distrib]
  field_simp
  ring

theorem nesterov_profile_noise_total (P : Problem N m n μ c) {g : ℝ}
    (hg : 0 < g) (x : Model m n) :
    (nesterovProfile P hg x).s₁+(nesterovProfile P hg x).s₂+(nesterovProfile P hg x).s₃ =
      Nesterov.noiseTotalScale P := by
  dsimp [nesterovProfile, Nesterov.noiseTotalScale, Nesterov.noiseScale]
  simp only [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem ema_optimalBudget_le_previous (P : Problem N m n μ c) {g ε : ℝ}
    (hg : 0 < g) (hε : 0 < ε) (x : Model m n) (ha : 0 < (emaProfile P hg x).a) :
    (emaProfile P hg x).optimalBudget ε ≤ accuracyBudget (P.rateInitial g x) g P.noiseScaleTotal ε := by
  have h := (emaProfile P hg x).optimalBudget_le_collapsed ha hε
  rw [(emaProfile P hg x).collapsedBudget_eq_accuracy hg ε,
    ema_profile_initial_total P hg x, ema_profile_noise_total P hg x] at h
  exact h

theorem nesterov_optimalBudget_le_previous (P : Problem N m n μ c) {g ε : ℝ}
    (hg : 0 < g) (hε : 0 < ε) (x : Model m n) (ha : 0 < (nesterovProfile P hg x).a) :
    (nesterovProfile P hg x).optimalBudget ε ≤
      accuracyBudget (Nesterov.rateInitial P g x) g (Nesterov.noiseTotalScale P) ε := by
  have h := (nesterovProfile P hg x).optimalBudget_le_collapsed ha hε
  rw [(nesterovProfile P hg x).collapsedBudget_eq_accuracy hg ε,
    nesterov_profile_initial_total P hg x, nesterov_profile_noise_total P hg x] at h
  exact h

end EF21Muon.LongTails
