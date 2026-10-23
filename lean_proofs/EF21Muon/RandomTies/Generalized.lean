import EF21Muon.RandomTies.Convergence

/-! Generalized-smooth constant-radius convergence with fresh independent
uplink ties. As in the paper, this result uses the exact downlink. -/

noncomputable section
namespace EF21Muon.RandomTies
open MeasureTheory
open LongTails.Algorithms.MatrixPolar
open EF21Muon.Distributed EF21Muon.Stochastic EF21Muon.Probability

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def emaRadiusNext (P : GeneralizedProblem N m n μ) (A : Scheme N m n c)
    (s : RadiusStates N m n) (z : Coins N m n × Ω) : RadiusStates N m n :=
  let r := Distributed.radiusNext P A s z.2
  (fun i => refresh (upBits z.1 i) (s.1 i).G (r.1 i).M (r.1 i), r.2)

theorem measurable_emaRadiusNext (P : GeneralizedProblem N m n μ) (A : Scheme N m n c) :
    Measurable (Function.uncurry (emaRadiusNext P A)) := by
  apply measurable_finite_choice
  intro b
  have hr := Distributed.measurable_radiusNext P A
  apply Measurable.prodMk _ hr.snd
  apply Measurable.of_eval
  intro i
  have hs := (measurable_pi_apply i).comp hr.fst
  exact measurable_refresh _
    ((measurable_pi_apply i).comp measurable_fst.fst).snd.snd.snd hs.snd.snd.fst hs

theorem ema_radius_potential_eq (P : GeneralizedProblem N m n μ) (A : Scheme N m n c)
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (s : RadiusStates N m n) (b : Coins N m n) (ω : Ω) :
    P.radiusPotential c (emaRadiusNext P A s (b,ω)) =
      P.radiusPotential c (Distributed.radiusNext P A s ω) := by
  unfold GeneralizedProblem.radiusPotential
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  apply refresh_radiusEnergy (hm i) (hn i)
  intro j
  simp only [Distributed.radiusNext, radiusUpdate, LayerState.G, LayerState.M, hCU]

theorem ema_generalized_finite_time (P : GeneralizedProblem N m n μ) (A : Scheme N m n c)
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (T : ℕ) (s : RadiusStates N m n) :
    Integrable (fun ξ => ((run (emaRadiusNext P A) T s ξ).map (P.radiusCost c)).sum)
      (pathMeasure ((coinLaw N m n).prod μ) T) ∧
    (∫ ξ, ((run (emaRadiusNext P A) T s ξ).map (P.radiusCost c)).sum
      ∂pathMeasure ((coinLaw N m n).prod μ) T) ≤
        (1+P.radiusKappa c)^T*(P.radiusPotential c s+(T:ℝ)*P.radiusBias c) :=
  finite_choice_convergence (coinLaw N m n) μ
    (by linarith [(P.radius_constants_nonneg c).1]) (P.radius_constants_nonneg c).2
    (measurable_emaRadiusNext P A) (P.measurable_radiusCost c) (P.measurable_radiusPotential c)
    (P.radiusCost_nonneg c) (P.radiusPotential_nonneg c) (ema_radius_potential_eq P A hCU hm hn)
    (fun _ => P.radiusPotential_integrable A) (fun _ => P.radius_drift A) T s

def nestRadiusNext (P : GeneralizedProblem N m n μ) (A : Scheme N m n c)
    (s : Nesterov.RadiusStates N m n) (z : Coins N m n × Ω) : Nesterov.RadiusStates N m n :=
  let r := Nesterov.radiusNext P A s z.2
  (fun i => nestRefresh (c i) (upBits z.1 i) (s.1 i).1.G (r.1 i), r.2)

theorem measurable_nestRadiusNext (P : GeneralizedProblem N m n μ) (A : Scheme N m n c) :
    Measurable (Function.uncurry (nestRadiusNext P A)) := by
  apply measurable_finite_choice
  intro b
  have hr := Nesterov.measurable_radiusNext P A
  apply Measurable.prodMk _ hr.snd
  apply Measurable.of_eval
  intro i
  exact measurable_nestRefresh _ _
    ((measurable_pi_apply i).comp measurable_fst.fst).fst.snd.snd.snd
    ((measurable_pi_apply i).comp hr.fst)

theorem nest_radius_potential_eq (P : GeneralizedProblem N m n μ) (A : Scheme N m n c)
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (s : Nesterov.RadiusStates N m n) (b : Coins N m n) (ω : Ω) :
    Nesterov.generalizedPotential P c (nestRadiusNext P A s (b,ω)) =
      Nesterov.generalizedPotential P c (Nesterov.radiusNext P A s ω) := by
  unfold Nesterov.generalizedPotential
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  apply nestRefresh_radiusEnergy (hm i) (hn i)
  intro j
  change _ = _+signCompressor (Nesterov.target _ (Nesterov.radiusUpdate _ _ _ _) _-_)
  have he : Nesterov.target (c i) (Nesterov.radiusUpdate (c i) (A.CU i) (s.1 i)
      (fun j => P.reply (Nesterov.radiusPoints c s.1) ω j i)) j =
      (1-(c i).β) • ((1-(c i).β) • (s.1 i).1.M j+
        (c i).β • P.reply (Nesterov.radiusPoints c s.1) ω j i)+
          (c i).β • P.reply (Nesterov.radiusPoints c s.1) ω j i := by
    dsimp [Nesterov.target, Nesterov.radiusUpdate, LayerState.M]
    module
  rw [he]
  simp only [Nesterov.radiusNext, Nesterov.radiusUpdate, LayerState.G, hCU]

theorem nest_generalized_finite_time (P : GeneralizedProblem N m n μ) (A : Scheme N m n c)
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (T : ℕ) (s : Nesterov.RadiusStates N m n) :
    Integrable (fun ξ => ((run (nestRadiusNext P A) T s ξ).map (Nesterov.generalizedCost P c)).sum)
      (pathMeasure ((coinLaw N m n).prod μ) T) ∧
    (∫ ξ, ((run (nestRadiusNext P A) T s ξ).map (Nesterov.generalizedCost P c)).sum
      ∂pathMeasure ((coinLaw N m n).prod μ) T) ≤
        (1+Nesterov.radiusKappa P c)^T*
          (Nesterov.generalizedPotential P c s+(T:ℝ)*Nesterov.radiusBias P c) :=
  finite_choice_convergence (coinLaw N m n) μ
    (by linarith [(Nesterov.radius_constants_nonneg P c).1])
    (Nesterov.radius_constants_nonneg P c).2 (measurable_nestRadiusNext P A)
    (Nesterov.generalizedCost_measurable P c) (Nesterov.generalizedPotential_measurable P c)
    (Nesterov.generalizedCost_nonneg P c) (Nesterov.generalizedPotential_nonneg P c)
    (nest_radius_potential_eq P A hCU hm hn) (fun _ => Nesterov.generalizedPotential_integrable P A)
    (fun _ => Nesterov.radius_drift P A) T s

end EF21Muon.RandomTies
