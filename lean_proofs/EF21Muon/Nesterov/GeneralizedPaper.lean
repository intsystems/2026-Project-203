import EF21Muon.Nesterov.GeneralizedRates
import EF21Muon.Nesterov.Correspondence
import EF21Muon.Distributed.GeneralizedPaper

/-! The generalized rate for the paper's gradient/uplink -> step/downlink
round order. This transfers the minimum of expectations through the index
shift on the same full sample path, before taking expectations. -/

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

def radiusServer (c : Fin p → Parameters) (s : States N m n) : States N m n :=
  fun i => ((radiusPoint (c i) (s i).1, radiusPoint (c i) (s i).1, (s i).1.M, (s i).1.G), (s i).2)

def radiusWorker (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters}
    (A : Scheme N m n c) (s : States N m n) (ω : Ω) : States N m n :=
  fun i =>
    let reply := fun j => P.reply (fun k => (s k).1.w) ω j i
    let M := fun j => (1-(c i).β) • (s i).1.M j+(c i).β • reply j
    let V := fun j => (1-(c i).β) • M j+(c i).β • reply j
    (((s i).1.x, (s i).1.w, M, fun j => (s i).1.G j+A.CU i (V j-(s i).1.G j)),(s i).1.M)

def radiusPaperNext (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters}
    (A : Scheme N m n c) (s : States N m n) (ω : Ω) : States N m n :=
  radiusServer c (radiusWorker P A s ω)

def radiusProject (c : Fin p → Parameters) (z : RadiusStates N m n) : States N m n :=
  radiusServer c z.1

theorem radiusProject_start (c : Fin p → Parameters) (x : Model m n) :
    radiusProject (N := N) c (radiusStart x) = initial x := by
  funext i
  simp [radiusProject, radiusServer, radiusStart, initial, start, LayerState.start, radiusPoint,
    LayerState.x, LayerState.M, LayerState.G, average, polar_zero]

theorem radiusPaperNext_project (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters}
    (A : Scheme N m n c) (z : RadiusStates N m n) (ω : Ω) :
    radiusPaperNext P A (radiusProject c z) ω = radiusProject c (radiusNext P A z ω) := rfl

theorem radiusPaper_pathState (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters}
    (A : Scheme N m n c) (k : ℕ) (x : Model m n) (ξ : Samples Ω k) (t : Fin k) :
    pathState (radiusPaperNext P A) k (initial x) ξ t =
      radiusProject c (pathState (radiusNext P A) k (radiusStart x) ξ t) := by
  rw [← radiusProject_start c x]
  exact pathState_map _ _ _ (radiusPaperNext_project P A) _ _ _ _

theorem measurable_radiusProject (c : Fin p → Parameters) :
    Measurable (radiusProject c : RadiusStates N m n → States N m n) := by
  apply Measurable.of_eval
  intro i
  have hx : Measurable (fun z : RadiusStates N m n => radiusPoints c z.1 i) :=
    (measurable_pi_apply i).comp ((measurable_radiusPoints c).comp measurable_fst)
  have hM : Measurable (fun z : RadiusStates N m n => (z.1 i).1.M) :=
    ((measurable_pi_apply i).comp measurable_fst).fst.snd.snd.fst
  have hG : Measurable (fun z : RadiusStates N m n => (z.1 i).1.G) :=
    ((measurable_pi_apply i).comp measurable_fst).fst.snd.snd.snd
  exact (hx.prodMk (hx.prodMk (hM.prodMk hG))).prodMk
    (((measurable_pi_apply i).comp measurable_fst).snd)

def paperNuclear (P : GeneralizedProblem N m n μ)
    (w : Fin p → ℝ) (s : States N m n) : ℝ := ∑ i, w i*nuclearNorm (P.grad (points s) i)

def paperRadiusMinimum (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (w : Fin p → ℝ)
    (x : Model m n) (T : ℕ) : ℝ :=
  expectedMinimum μ (radiusPaperNext P A) (paperNuclear P w) (initial x) T

theorem project_weighted_bound (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) {w : Fin p → ℝ} (hw : ∀ i, 0 ≤ w i) (z : RadiusStates N m n) :
    paperNuclear P w (radiusProject c z) ≤
      (1+∑ i, (c i).γ*P.L₁ i)*weightedNuclear P w z + ∑ i, w i*(c i).γ*P.L₀ i := by
  have hterm (i : Fin p) : w i*nuclearNorm (P.grad (radiusPoints c z.1) i) ≤
      (1+∑ k, (c k).γ*P.L₁ k)*(w i*nuclearNorm (P.grad (points z.1) i)) + w i*(c i).γ*P.L₀ i := by
    have hsum := Finset.single_le_sum (fun k (_ : k ∈ Finset.univ) =>
      mul_nonneg (c k).γ_pos.le (P.L₁_nonneg k)) (Finset.mem_univ i)
    have h1 := mul_le_mul_of_nonneg_left (P.radius_query_gradient_bound c (core z.1) i) (hw i)
    have h2 := mul_le_mul_of_nonneg_right (add_le_add (le_rfl : (1:ℝ) ≤ 1) hsum)
      (mul_nonneg (hw i) (nuclearNorm_nonneg (P.grad (points z.1) i)))
    dsimp only [points, radiusPoints] at h1 h2 ⊢
    nlinarith only [h1, h2]
  have hs := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hterm i)
  unfold paperNuclear weightedNuclear
  rw [show points (radiusProject c z) = radiusPoints c z.1 from rfl]
  simpa only [Finset.sum_add_distrib, ← Finset.mul_sum] using hs

theorem project_weighted_integrable (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) {Ξ : Type*} [MeasurableSpace Ξ] {ν : Measure Ξ} [IsFiniteMeasure ν]
    {X : Ξ → RadiusStates N m n} (hX : Measurable X)
    {w : Fin p → ℝ} (hw : ∀ i, 0 ≤ w i)
    (hi : Integrable (fun ξ => weightedNuclear P w (X ξ)) ν) :
    Integrable (fun ξ => paperNuclear P w (radiusProject c (X ξ))) ν := by
  have hquery := (measurable_radiusPoints c).comp hX.fst
  have hm : Measurable (fun ξ => paperNuclear P w (radiusProject c (X ξ))) := by
    apply Finset.measurable_sum
    intro i _
    exact (continuous_nuclearNorm.measurable.comp ((measurable_pi_apply i).comp
      (P.gradient_continuous.measurable.comp hquery))).const_mul _
  have hupper : Integrable (fun ξ =>
      (1+∑ i, (c i).γ*P.L₁ i)*weightedNuclear P w (X ξ)+∑ i, w i*(c i).γ*P.L₀ i) ν :=
    (hi.const_mul _).add (integrable_const _)
  apply hupper.mono' hm.aestronglyMeasurable
  apply Filter.Eventually.of_forall
  intro ξ
  rw [Real.norm_of_nonneg (show 0 ≤ paperNuclear P w (radiusProject c (X ξ)) from
    Finset.sum_nonneg fun i _ => mul_nonneg (hw i) (nuclearNorm_nonneg _))]
  exact project_weighted_bound P c hw (X ξ)

theorem generalized_paper_iterate_integrable [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (x : Model m n) (t : Fin (T+1)) :
    Integrable (fun ξ => paperNuclear P (fun i => η i/ηbar)
      (pathState (radiusPaperNext P A) (T+1) (initial x) ξ t)) (pathMeasure μ (T+1)) := by
  let c := generalizedHorizonControls base η hη T
  let X := fun ξ => pathState (radiusNext P A) (T+1) (radiusStart x) ξ t
  have hpair : Measurable (fun ξ : Samples Ω (T+1) => (radiusStart (N := N) x, ξ)) :=
    measurable_const.prodMk measurable_id
  have hX : Measurable X := (pathState_measurable (measurable_radiusNext P A) _ t).comp hpair
  have hw i := div_nonneg (hη i).le hbar.le
  have hi := path_cost_integrable (measurable_radiusNext P A) (measurable_weightedNuclear P _)
    (weightedNuclear_nonneg P hw) (T+1) (radiusStart x)
    (generalized_weighted_sum_integrable P base η hη hbar T (T+1) A x) t
  have hproject := project_weighted_integrable P c hX hw hi
  have he (ξ : Samples Ω (T+1)) :
      pathState (radiusPaperNext P A) (T+1) (initial x) ξ t = radiusProject c (X ξ) := by
    exact radiusPaper_pathState P A _ _ _ _
  simpa only [he] using hproject

theorem generalized_paper_expectation_eq [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (x : Model m n) (t : Fin (T+1)) :
    (∫ ξ, paperNuclear P (fun i => η i/ηbar)
      (pathState (radiusPaperNext P A) (T+1) (initial x) ξ t) ∂pathMeasure μ (T+1)) =
      ∑ i, (η i/ηbar)*(∫ ξ, nuclearNorm (P.grad
        (points (pathState (radiusPaperNext P A) (T+1) (initial x) ξ t)) i) ∂pathMeasure μ (T+1)) := by
  let c := generalizedHorizonControls base η hη T
  have hpair : Measurable (fun ξ : Samples Ω (T+1) => (radiusStart (N := N) x, ξ)) :=
    measurable_const.prodMk measurable_id
  have hpath := (pathState_measurable (measurable_radiusNext P A) (T+1) t).comp hpair
  have hpaper : Measurable (fun ξ => pathState (radiusPaperNext P A) (T+1) (initial x) ξ t) := by
    have h := (measurable_radiusProject c).comp hpath
    simpa only [Function.comp_def, Function.uncurry_def, radiusPaper_pathState P A, c] using h
  have hX : Measurable (fun ξ =>
      (pathState (radiusPaperNext P A) (T+1) (initial x) ξ t, (0 : Model m n))) :=
    hpaper.prodMk measurable_const
  exact integral_weightedNuclear P hX (fun i => div_nonneg (hη i).le hbar.le)
    (generalized_paper_iterate_integrable P base η hη hbar T A x t)

theorem paperRadiusMinimum_le (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (w : Fin p → ℝ)
    (x : Model m n) (T : ℕ) :
    paperRadiusMinimum P A w x T ≤ radiusMinimum P A w x T := by
  have hmap (ξ : Samples Ω (T+1)) (t : Fin (T+1)) :
      pathState (radiusPaperNext P A) (T+1) (initial x) ξ t =
        radiusProject c (pathState (radiusNext P A) (T+1) (radiusStart x) ξ t) := by
    exact radiusPaper_pathState P A _ _ _ _
  have hcost (z : RadiusStates N m n) (ω : Ω) :
      weightedNuclear P w (radiusNext P A z ω) = paperNuclear P w (radiusProject c z) := rfl
  have he (ξ : Samples Ω (T+1)) (t : Fin T) :
      paperNuclear P w (pathState (radiusPaperNext P A) (T+1) (initial x) ξ t.castSucc) =
        weightedNuclear P w (pathState (radiusNext P A) (T+1) (radiusStart x) ξ t.succ) := by
    rw [hmap]
    exact (pathState_adjacent_cost _ _ _ hcost T (radiusStart x) ξ t).symm
  unfold radiusMinimum expectedMinimum
  apply Finset.le_inf'
  intro t _
  refine Fin.cases ?_ (fun k => ?_) t
  · have hl : paperRadiusMinimum P A w x T ≤
        ∫ ξ, paperNuclear P w (pathState (radiusPaperNext P A) (T+1) (initial x) ξ 0)
          ∂pathMeasure μ (T+1) := Finset.inf'_le _ (Finset.mem_univ 0)
    exact hl
  · have hl : paperRadiusMinimum P A w x T ≤
        ∫ ξ, paperNuclear P w (pathState (radiusPaperNext P A) (T+1) (initial x) ξ k.castSucc)
          ∂pathMeasure μ (T+1) := Finset.inf'_le _ (Finset.mem_univ k.castSucc)
    simpa only [he] using hl

theorem generalized_paper_minimum_rate [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T)) (x : Model m n) :
    paperRadiusMinimum P A (fun i => η i/ηbar) x T ≤
      (2*Real.exp (generalizedK P base η)*
        (generalizedInitial P base η x+generalizedB P base η)/ηbar)/generalizedRoot T :=
  (paperRadiusMinimum_le P A _ x T).trans (generalized_minimum_rate P base η hη hbar T A x)

/-- The generalized Nesterov theorem, for the paper's round order and iterates 0,...,T. -/
theorem matrix_generalized_paper_rate [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (hp : 0 < p)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (x : Model m n) :
    (fun T => paperRadiusMinimum P (generalizedMatrixScheme hm hn η hη T)
      (fun i => η i/average η) x T)
      =O[Filter.atTop] (fun T : ℕ => (T:ℝ)^(-(1:ℝ)/4)) := by
  apply Asymptotics.IsBigO.trans _ (matrix_generalized_horizon_rate_rpow P hp hm hn η hη x)
  apply Asymptotics.IsBigO.of_bound 1
  apply Filter.Eventually.of_forall
  intro T
  have hw i : 0 ≤ η i/average η := div_nonneg (hη i).le (layer_average_pos hp hη).le
  have hpaper : 0 ≤ paperRadiusMinimum P (generalizedMatrixScheme hm hn η hη T)
      (fun i => η i/average η) x T :=
    expectedMinimum_nonneg μ _ _ (fun s => Finset.sum_nonneg fun i _ =>
      mul_nonneg (hw i) (nuclearNorm_nonneg _)) _ _
  have hframework : 0 ≤ radiusMinimum P (generalizedMatrixScheme hm hn η hη T)
      (fun i => η i/average η) x T :=
    expectedMinimum_nonneg μ _ _ (weightedNuclear_nonneg P hw) _ _
  rw [Real.norm_eq_abs, abs_of_nonneg hpaper, Real.norm_eq_abs, abs_of_nonneg hframework, one_mul]
  exact paperRadiusMinimum_le P _ _ _ _

theorem matrix_generalized_paper_unweighted_rate [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (hp : 0 < p)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) {η : ℝ} (hη : 0 < η) (x : Model m n) :
    (fun T => paperRadiusMinimum P (generalizedMatrixScheme hm hn (fun _ => η) (fun _ => hη) T)
      (fun _ => 1) x T)
      =O[Filter.atTop] (fun T : ℕ => (T:ℝ)^(-(1:ℝ)/4)) := by
  have h := matrix_generalized_paper_rate P hp hm hn (fun _ => η) (fun _ => hη) x
  have hw : (fun _ : Fin p => η/average (fun _ : Fin p => η)) = (fun _ => 1) := by
    funext i
    rw [average_const hp, div_self hη.ne']
  rwa [hw] at h

end EF21Muon.Nesterov
