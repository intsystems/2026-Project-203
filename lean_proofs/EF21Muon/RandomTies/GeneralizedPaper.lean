import EF21Muon.RandomTies.GeneralizedRates

/-! Literal query/uplink/server order for generalized smoothness and random ties. -/
noncomputable section
namespace EF21Muon.RandomTies
set_option maxHeartbeats 800000
set_option backward.isDefEq.respectTransparency false
open MeasureTheory hiding average
open EF21Muon.Distributed EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators
variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def emaRadiusPaperNext (P : GeneralizedProblem N m n μ)
    (s : States N m n) (z : Coins N m n × Ω) : States N m n :=
  radiusServer c (emaWorker (c := c) P.reply s z)

theorem emaRadiusPaperNext_project (P : GeneralizedProblem N m n μ)
    (A : Scheme N m n c) (z : RadiusStates N m n) (ω : Coins N m n × Ω) :
    emaRadiusPaperNext (c := c) P (radiusProject c z) ω =
      radiusProject c (emaRadiusNext P A z ω) := rfl

theorem emaRadiusPaper_pathState (P : GeneralizedProblem N m n μ)
    (A : Scheme N m n c) (k : ℕ) (x : Model m n)
    (ξ : Samples (Coins N m n × Ω) k) (t : Fin k) :
    pathState (emaRadiusPaperNext (c := c) P) k (start x) ξ t =
      radiusProject c (pathState (emaRadiusNext P A) k (radiusStart x) ξ t) := by
  conv_lhs => rw [← radiusProject_start c x]
  exact pathState_map _ _ _ (emaRadiusPaperNext_project P A) _ _ _ _

def emaPaperRadiusMinimum (P : GeneralizedProblem N m n μ) (c : Fin p → Parameters)
    (w : Fin p → ℝ) (x : Model m n) (T : ℕ) : ℝ :=
  expectedMinimum ((coinLaw N m n).prod μ) (emaRadiusPaperNext (c := c) P)
    (P.paperNuclear w) (start x) T

theorem ema_generalized_paper_iterate_integrable
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (x : Model m n) (t : Fin (T+1)) :
    Integrable (fun ξ => P.paperNuclear (fun i => η i/ηbar)
      (pathState (emaRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (start x) ξ t)) (pathMeasure ((coinLaw N m n).prod μ) (T+1)) := by
  let c := generalizedHorizonControls base η hη T
  let X := fun ξ => pathState (emaRadiusNext P A) (T+1) (radiusStart x) ξ t
  have hpair : Measurable (fun ξ : Samples (Coins N m n × Ω) (T+1) => (radiusStart (N := N) x, ξ)) :=
    measurable_const.prodMk measurable_id
  have hX : Measurable X := (pathState_measurable (measurable_emaRadiusNext P A) _ t).comp hpair
  have hw i := div_nonneg (hη i).le hbar.le
  have hi := path_cost_integrable (measurable_emaRadiusNext P A) (P.measurable_weightedNuclear _)
    (P.weightedNuclear_nonneg hw) (T+1) (radiusStart x)
    (ema_generalized_weighted_sum_integrable P base η hη hbar T (T+1) A hCU hm hn x) t
  have hproject := P.project_weighted_integrable c hX hw hi
  have he (ξ : Samples (Coins N m n × Ω) (T+1)) :
      pathState (emaRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (start x) ξ t = radiusProject c (X ξ) := by
    exact emaRadiusPaper_pathState P A _ _ _ _
  simpa only [he] using hproject

theorem ema_generalized_paper_expectation_eq
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (x : Model m n) (t : Fin (T+1)) :
    (∫ ξ, P.paperNuclear (fun i => η i/ηbar)
      (pathState (emaRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (start x) ξ t) ∂pathMeasure ((coinLaw N m n).prod μ) (T+1)) =
      ∑ i, (η i/ηbar)*(∫ ξ, nuclearNorm (P.grad
        (points (pathState (emaRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (start x) ξ t)) i) ∂pathMeasure ((coinLaw N m n).prod μ) (T+1)) := by
  let c := generalizedHorizonControls base η hη T
  have hpair : Measurable (fun ξ : Samples (Coins N m n × Ω) (T+1) => (radiusStart (N := N) x, ξ)) :=
    measurable_const.prodMk measurable_id
  have hpath := (pathState_measurable (measurable_emaRadiusNext P A) (T+1) t).comp hpair
  have hpaper : Measurable (fun ξ => pathState (emaRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (start x) ξ t) := by
    have h := (measurable_radiusProject c).comp hpath
    simpa only [Function.comp_def, Function.uncurry_def, emaRadiusPaper_pathState P A, c] using h
  have hX : Measurable (fun ξ =>
      (pathState (emaRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (start x) ξ t, (0 : Model m n))) :=
    hpaper.prodMk measurable_const
  exact P.integral_weightedNuclear hX (fun i => div_nonneg (hη i).le hbar.le)
    (ema_generalized_paper_iterate_integrable P base η hη hbar T A hCU hm hn x t)

theorem ema_paperRadiusMinimum_le (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (w : Fin p → ℝ)
    (x : Model m n) (T : ℕ) :
    emaPaperRadiusMinimum P c w x T ≤ emaRadiusMinimum P A w x T := by
  have hmap (ξ : Samples (Coins N m n × Ω) (T+1)) (t : Fin (T+1)) :
      pathState (emaRadiusPaperNext (c := c) P) (T+1) (start x) ξ t =
        radiusProject c (pathState (emaRadiusNext P A) (T+1) (radiusStart x) ξ t) := by
    exact emaRadiusPaper_pathState P A _ _ _ _
  have hcost (z : RadiusStates N m n) (ω : Coins N m n × Ω) :
      P.weightedNuclear w (emaRadiusNext P A z ω) = P.paperNuclear w (radiusProject c z) := rfl
  have he (ξ : Samples (Coins N m n × Ω) (T+1)) (t : Fin T) :
      P.paperNuclear w (pathState (emaRadiusPaperNext (c := c) P) (T+1) (start x) ξ t.castSucc) =
        P.weightedNuclear w (pathState (emaRadiusNext P A) (T+1) (radiusStart x) ξ t.succ) := by
    rw [hmap]
    exact (pathState_adjacent_cost _ _ _ hcost T (radiusStart x) ξ t).symm
  unfold emaRadiusMinimum expectedMinimum
  apply Finset.le_inf'
  intro t _
  refine Fin.cases ?_ (fun k => ?_) t
  · have hl : emaPaperRadiusMinimum P c w x T ≤
        ∫ ξ, P.paperNuclear w (pathState (emaRadiusPaperNext (c := c) P) (T+1) (start x) ξ 0)
          ∂pathMeasure ((coinLaw N m n).prod μ) (T+1) := Finset.inf'_le _ (Finset.mem_univ 0)
    exact hl
  · have hl : emaPaperRadiusMinimum P c w x T ≤
        ∫ ξ, P.paperNuclear w (pathState (emaRadiusPaperNext (c := c) P) (T+1) (start x) ξ k.castSucc)
          ∂pathMeasure ((coinLaw N m n).prod μ) (T+1) := Finset.inf'_le _ (Finset.mem_univ k.castSucc)
    simpa only [he] using hl

/-- The printed generalized-smooth first-moment rate, with the paper's randomized
ties and round order. The constant is independent of the horizon. -/
theorem ema_generalized_paper_minimum_rate
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (x : Model m n) :
    emaPaperRadiusMinimum P (generalizedHorizonControls base η hη T)
      (fun i => η i/ηbar) x T ≤
        (2*Real.exp (P.generalizedK base η)*
          (P.generalizedInitial base η x+P.generalizedB base η)/ηbar)/generalizedRoot T :=
  (ema_paperRadiusMinimum_le P A _ x T).trans
    (ema_generalized_minimum_rate P base η hη hbar T A hCU hm hn x)

section NesterovPaper
open EF21Muon.Nesterov

def nestRadiusPaperNext (P : GeneralizedProblem N m n μ)
    (s : Nesterov.States N m n) (z : Coins N m n × Ω) : Nesterov.States N m n :=
  Nesterov.radiusServer c (nestWorker (c := c) P.reply s z)

theorem nestRadiusPaperNext_project (P : GeneralizedProblem N m n μ)
    (A : Scheme N m n c) (z : Nesterov.RadiusStates N m n) (ω : Coins N m n × Ω) :
    nestRadiusPaperNext (c := c) P (Nesterov.radiusProject c z) ω =
      Nesterov.radiusProject c (nestRadiusNext P A z ω) := rfl

theorem nestRadiusPaper_pathState (P : GeneralizedProblem N m n μ)
    (A : Scheme N m n c) (k : ℕ) (x : Model m n)
    (ξ : Samples (Coins N m n × Ω) k) (t : Fin k) :
    pathState (nestRadiusPaperNext (c := c) P) k (Nesterov.initial x) ξ t =
      Nesterov.radiusProject c (pathState (nestRadiusNext P A) k (Nesterov.radiusStart x) ξ t) := by
  conv_lhs => rw [← Nesterov.radiusProject_start c x]
  exact pathState_map _ _ _ (nestRadiusPaperNext_project P A) _ _ _ _

def nestPaperRadiusMinimum (P : GeneralizedProblem N m n μ) (c : Fin p → Parameters)
    (w : Fin p → ℝ) (x : Model m n) (T : ℕ) : ℝ :=
  expectedMinimum ((coinLaw N m n).prod μ) (nestRadiusPaperNext (c := c) P)
    (Nesterov.paperNuclear P w) (Nesterov.initial x) T

theorem nest_generalized_paper_iterate_integrable
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (x : Model m n) (t : Fin (T+1)) :
    Integrable (fun ξ => paperNuclear P (fun i => η i/ηbar)
      (pathState (nestRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (Nesterov.initial x) ξ t)) (pathMeasure ((coinLaw N m n).prod μ) (T+1)) := by
  let c := generalizedHorizonControls base η hη T
  let X := fun ξ => pathState (nestRadiusNext P A) (T+1) (Nesterov.radiusStart x) ξ t
  have hpair : Measurable (fun ξ : Samples (Coins N m n × Ω) (T+1) => (Nesterov.radiusStart (N := N) x, ξ)) :=
    measurable_const.prodMk measurable_id
  have hX : Measurable X := (pathState_measurable (measurable_nestRadiusNext P A) _ t).comp hpair
  have hw i := div_nonneg (hη i).le hbar.le
  have hi := path_cost_integrable (measurable_nestRadiusNext P A) (measurable_weightedNuclear P _)
    (weightedNuclear_nonneg P hw) (T+1) (Nesterov.radiusStart x)
    (nest_generalized_weighted_sum_integrable P base η hη hbar T (T+1) A hCU hm hn x) t
  have hproject := project_weighted_integrable P c hX hw hi
  have he (ξ : Samples (Coins N m n × Ω) (T+1)) :
      pathState (nestRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (Nesterov.initial x) ξ t = Nesterov.radiusProject c (X ξ) := by
    exact nestRadiusPaper_pathState P A _ _ _ _
  simpa only [he] using hproject

theorem nest_generalized_paper_expectation_eq
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (x : Model m n) (t : Fin (T+1)) :
    (∫ ξ, paperNuclear P (fun i => η i/ηbar)
      (pathState (nestRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (Nesterov.initial x) ξ t) ∂pathMeasure ((coinLaw N m n).prod μ) (T+1)) =
      ∑ i, (η i/ηbar)*(∫ ξ, nuclearNorm (P.grad
        (points (pathState (nestRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (Nesterov.initial x) ξ t)) i) ∂pathMeasure ((coinLaw N m n).prod μ) (T+1)) := by
  let c := generalizedHorizonControls base η hη T
  have hpair : Measurable (fun ξ : Samples (Coins N m n × Ω) (T+1) => (Nesterov.radiusStart (N := N) x, ξ)) :=
    measurable_const.prodMk measurable_id
  have hpath := (pathState_measurable (measurable_nestRadiusNext P A) (T+1) t).comp hpair
  have hpaper : Measurable (fun ξ => pathState (nestRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (Nesterov.initial x) ξ t) := by
    have h := (Nesterov.measurable_radiusProject c).comp hpath
    simpa only [Function.comp_def, Function.uncurry_def, nestRadiusPaper_pathState P A, c] using h
  have hX : Measurable (fun ξ =>
      (pathState (nestRadiusPaperNext (c := generalizedHorizonControls base η hη T) P) (T+1) (Nesterov.initial x) ξ t, (0 : Model m n))) :=
    hpaper.prodMk measurable_const
  exact integral_weightedNuclear P hX (fun i => div_nonneg (hη i).le hbar.le)
    (nest_generalized_paper_iterate_integrable P base η hη hbar T A hCU hm hn x t)

theorem nest_paperRadiusMinimum_le (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (w : Fin p → ℝ)
    (x : Model m n) (T : ℕ) :
    nestPaperRadiusMinimum P c w x T ≤ nestRadiusMinimum P A w x T := by
  have hmap (ξ : Samples (Coins N m n × Ω) (T+1)) (t : Fin (T+1)) :
       pathState (nestRadiusPaperNext (c := c) P) (T+1) (Nesterov.initial x) ξ t =
         Nesterov.radiusProject c (pathState (nestRadiusNext P A) (T+1) (Nesterov.radiusStart x) ξ t) := by
    exact nestRadiusPaper_pathState P A _ _ _ _
  have hcost (z : Nesterov.RadiusStates N m n) (ω : Coins N m n × Ω) :
      weightedNuclear P w (nestRadiusNext P A z ω) = paperNuclear P w (Nesterov.radiusProject c z) := rfl
  have he (ξ : Samples (Coins N m n × Ω) (T+1)) (t : Fin T) :
      paperNuclear P w (pathState (nestRadiusPaperNext (c := c) P) (T+1) (Nesterov.initial x) ξ t.castSucc) =
        weightedNuclear P w (pathState (nestRadiusNext P A) (T+1) (Nesterov.radiusStart x) ξ t.succ) := by
    rw [hmap]
    exact (pathState_adjacent_cost _ _ _ hcost T (Nesterov.radiusStart x) ξ t).symm
  unfold nestRadiusMinimum expectedMinimum
  apply Finset.le_inf'
  intro t _
  refine Fin.cases ?_ (fun k => ?_) t
  · have hl : nestPaperRadiusMinimum P c w x T ≤
        ∫ ξ, paperNuclear P w (pathState (nestRadiusPaperNext (c := c) P) (T+1) (Nesterov.initial x) ξ 0)
          ∂pathMeasure ((coinLaw N m n).prod μ) (T+1) := Finset.inf'_le _ (Finset.mem_univ 0)
    exact hl
  · have hl : nestPaperRadiusMinimum P c w x T ≤
        ∫ ξ, paperNuclear P w (pathState (nestRadiusPaperNext (c := c) P) (T+1) (Nesterov.initial x) ξ k.castSucc)
          ∂pathMeasure ((coinLaw N m n).prod μ) (T+1) := Finset.inf'_le _ (Finset.mem_univ k.castSucc)
    simpa only [he] using hl


theorem nest_generalized_paper_minimum_rate
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (x : Model m n) :
    nestPaperRadiusMinimum P (generalizedHorizonControls base η hη T)
      (fun i => η i/ηbar) x T ≤
        (2*Real.exp (Nesterov.generalizedK P base η)*
          (Nesterov.generalizedInitial P base η x+Nesterov.generalizedB P base η)/ηbar)/generalizedRoot T :=
  (nest_paperRadiusMinimum_le P A _ x T).trans
    (nest_generalized_minimum_rate P base η hη hbar T A hCU hm hn x)

end NesterovPaper
end EF21Muon.RandomTies
