import EF21Muon.RandomTies.Refresh
import EF21Muon.Probability.FiniteMixture

/-! Full random-tie trajectories. Each round samples independent fair bits
for all worker/layer/entry ties and downlink ties, independently of the fresh
gradient sample. The inherited drift is integrated over both sources of noise.
No hypothesis asserting convergence of the randomized trajectory is used. -/

noncomputable section
namespace EF21Muon.RandomTies
open MeasureTheory
open LongTails.Algorithms.MatrixPolar
open EF21Muon.Distributed EF21Muon.Stochastic EF21Muon.Probability

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def emaNext (A : Coins N m n → Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n) (s : States N m n)
    (z : Coins N m n × Ω) : States N m n :=
  let r := Distributed.next (A z.1) reply s z.2
  fun i => refresh (upBits z.1 i) (s i).G (r i).M (r i)

theorem measurable_emaNext (A : Coins N m n → Scheme N m n c)
    {reply : Model m n → Ω → Fin N → Model m n}
    (hr : Measurable (Function.uncurry reply)) :
    Measurable (Function.uncurry (emaNext A reply)) := by
  apply measurable_finite_choice
  intro b
  apply Measurable.of_eval
  intro i
  have hs := (measurable_pi_apply i).comp (Distributed.measurable_next (A b) hr)
  exact measurable_refresh _ ((measurable_pi_apply i).comp measurable_fst).snd.snd.snd
    hs.snd.snd.fst hs

theorem ema_potential_eq (P : Problem N m n μ c)
    (A : Coins N m n → Scheme N m n c)
    (hCU : ∀ b i, (A b).CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (s : States N m n) (b : Coins N m n) (ω : Ω) :
    P.potential (emaNext A P.reply s (b,ω)) =
      P.potential (Distributed.next (A b) P.reply s ω) := by
  unfold Problem.potential
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  exact refresh_energy (hm i) (hn i) (c i) _ _ _ _
    (by intro j; change _ = _+signCompressor _; simp only [Distributed.next, update, LayerState.G, LayerState.M, hCU]) _ _

/-- EMA, arbitrary worker count and heterogeneous blocks, with random ties on
both links. The coefficient of the variance term and the step restriction agree
with the deterministic-tie theorem. -/
theorem ema_finite_time (P : Problem N m n μ c)
    (A : Coins N m n → Scheme N m n c)
    (hCU : ∀ b i, (A b).CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (hsmall : ∀ b i, 2*(c i).L*(c i).γ+
      4*zeta (c i) (P.ell2 i) ((A b).τ i)*(c i).γ^2 ≤ 1)
    (T : ℕ) (s : States N m n) :
    Integrable (fun ξ => ((run (emaNext A P.reply) T s ξ).map P.cost).sum)
      (pathMeasure ((coinLaw N m n).prod μ) T) ∧
    (∫ ξ, ((run (emaNext A P.reply) T s ξ).map P.cost).sum
      ∂pathMeasure ((coinLaw N m n).prod μ) T) ≤ P.potential s+(T:ℝ)*P.noiseTotal := by
  have hb : 0 ≤ P.noiseTotal := by
    apply Finset.sum_nonneg
    intro i _
    have hγ := (c i).γ_pos
    have hβ := (c i).β_pos
    have hα := (c i).αU_pos
    have hα1 := sub_nonneg.mpr (c i).αU_le_one
    unfold noise
    positivity
  simpa only [one_pow, one_mul] using finite_choice_convergence (coinLaw N m n) μ
    (a := 1) (by norm_num) hb (measurable_emaNext A P.measurable_reply)
    P.cost_measurable P.potential_measurable P.cost_nonneg P.potential_nonneg
    (ema_potential_eq P A hCU hm hn) (fun b => P.potential_next_integrable (A b))
    (fun b => by simpa only [one_mul] using P.potential_descent (A b) (hsmall b)) T s

def nestNext (A : Coins N m n → Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n) (s : Nesterov.States N m n)
    (z : Coins N m n × Ω) : Nesterov.States N m n :=
  let r := Nesterov.next (A z.1) reply s z.2
  fun i => nestRefresh (c i) (upBits z.1 i) (s i).1.G (r i)

theorem measurable_nestNext (A : Coins N m n → Scheme N m n c)
    {reply : Model m n → Ω → Fin N → Model m n}
    (hr : Measurable (Function.uncurry reply)) :
    Measurable (Function.uncurry (nestNext A reply)) := by
  apply measurable_finite_choice
  intro b
  apply Measurable.of_eval
  intro i
  have hs := (measurable_pi_apply i).comp (Nesterov.measurable_next (A b) hr)
  exact measurable_nestRefresh _ _
    ((measurable_pi_apply i).comp measurable_fst).fst.snd.snd.snd hs

theorem nest_potential_eq (P : Problem N m n μ c)
    (A : Coins N m n → Scheme N m n c)
    (hCU : ∀ b i, (A b).CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (s : Nesterov.States N m n) (b : Coins N m n) (ω : Ω) :
    Nesterov.Averaged.smoothPotential P (nestNext A P.reply s (b,ω)) =
      Nesterov.Averaged.smoothPotential P (Nesterov.next (A b) P.reply s ω) := by
  unfold Nesterov.Averaged.smoothPotential
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  apply nestRefresh_energy (hm i) (hn i)
  intro j
  change _ = _+signCompressor (Nesterov.target _ (Nesterov.update _ _ _ _ _) _-_)
  rw [Nesterov.update_target]
  simp only [Nesterov.next, Nesterov.update, LayerState.G, LayerState.M, hCU]

/-- Nesterov with independent worker ties and randomized downlink. In particular,
the leading averaged-worker noise contribution retains its factor `1/N`. -/
theorem nest_finite_time (P : Problem N m n μ c)
    (A : Coins N m n → Scheme N m n c)
    (hCU : ∀ b i, (A b).CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (hsmall : ∀ b i, 2*(c i).L*(c i).γ+
      4*Nesterov.zeta (c i) (P.ell2 i) ((A b).τ i)*(c i).γ^2 ≤ 1)
    (T : ℕ) (s : Nesterov.States N m n) :
    Integrable (fun ξ => ((run (nestNext A P.reply) T s ξ).map (Nesterov.smoothCost P)).sum)
      (pathMeasure ((coinLaw N m n).prod μ) T) ∧
    (∫ ξ, ((run (nestNext A P.reply) T s ξ).map (Nesterov.smoothCost P)).sum
      ∂pathMeasure ((coinLaw N m n).prod μ) T) ≤
        Nesterov.Averaged.smoothPotential P s+(T:ℝ)*Nesterov.Averaged.smoothNoise P := by
  simpa only [one_pow, one_mul] using finite_choice_convergence (coinLaw N m n) μ
    (a := 1) (by norm_num) (Nesterov.Averaged.smoothNoise_nonneg P)
    (measurable_nestNext A P.measurable_reply) (Nesterov.smoothCost_measurable P)
    (Nesterov.Averaged.smoothPotential_measurable P) (Nesterov.smoothCost_nonneg P)
    (Nesterov.Averaged.smoothPotential_nonneg P) (nest_potential_eq P A hCU hm hn)
    (fun b => Nesterov.Averaged.smoothPotential_next_integrable P (A b))
    (fun b => by simpa only [one_mul] using
      Nesterov.Averaged.smoothPotential_descent P (A b) (hsmall b)) T s

end EF21Muon.RandomTies
