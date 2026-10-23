import EF21Muon.Probability.GeometricDrift

/-! Individual iterate expectations and their finite minimum on actual paths. -/

noncomputable section

namespace EF21Muon.Probability

open MeasureTheory
open EF21Muon.Stochastic (Samples run)
open scoped BigOperators

variable {Ω S : Type*} [MeasurableSpace Ω] [MeasurableSpace S]

def pathState (next : S → Ω → S) : (T : ℕ) → S → Samples Ω T → Fin T → S
  | 0, _, _, t => Fin.elim0 t
  | T+1, s, ξ, t => Fin.cases s (fun k => pathState next T (next s ξ.1) ξ.2 k) t

theorem pathState_measurable {next : S → Ω → S}
    (hn : Measurable (Function.uncurry next)) (T : ℕ) (t : Fin T) :
    Measurable (fun z : S × Samples Ω T => pathState next T z.1 z.2 t) := by
  induction T with
  | zero => exact Fin.elim0 t
  | succ T ih =>
    refine Fin.cases ?_ (fun k => ?_) t
    · exact measurable_fst
    · exact (ih k).comp
        ((hn.comp (measurable_fst.prodMk measurable_snd.fst)).prodMk measurable_snd.snd)

omit [MeasurableSpace Ω] [MeasurableSpace S] in
theorem sum_run_eq_states (next : S → Ω → S) (cost : S → ℝ)
    (T : ℕ) (s : S) (ξ : Samples Ω T) :
    ((run next T s ξ).map cost).sum = ∑ t, cost (pathState next T s ξ t) := by
  induction T generalizing s with
  | zero => simp [run]
  | succ T ih => simp [run, Fin.sum_univ_succ, pathState, ih]

omit [MeasurableSpace Ω] [MeasurableSpace S] in
theorem pathState_map {R : Type*} (next : S → Ω → S) (nextR : R → Ω → R)
    (project : S → R) (hnext : ∀ s ω, nextR (project s) ω = project (next s ω))
    (T : ℕ) (s : S) (ξ : Samples Ω T) (t : Fin T) :
    pathState nextR T (project s) ξ t = project (pathState next T s ξ t) := by
  induction T generalizing s with
  | zero => exact Fin.elim0 t
  | succ T ih =>
    refine Fin.cases rfl (fun k => ?_) t
    change pathState nextR T (nextR (project s) ξ.1) ξ.2 k = _
    rw [hnext]
    exact ih _ _ _

omit [MeasurableSpace Ω] [MeasurableSpace S] in
theorem pathState_adjacent_cost (next : S → Ω → S) (F H : S → ℝ)
    (hcost : ∀ s ω, F (next s ω) = H s) (T : ℕ) (s : S)
    (ξ : Samples Ω (T+1)) (t : Fin T) :
    F (pathState next (T+1) s ξ t.succ) = H (pathState next (T+1) s ξ t.castSucc) := by
  induction T generalizing s with
  | zero => exact Fin.elim0 t
  | succ T ih =>
    refine Fin.cases ?_ (fun k => ?_) t
    · exact hcost s ξ.1
    · change F (pathState next (T+1) (next s ξ.1) ξ.2 k.succ) =
        H (pathState next (T+1) (next s ξ.1) ξ.2 k.castSucc)
      exact ih _ _ _

theorem path_cost_integrable {μ : Measure Ω} {next : S → Ω → S} {cost : S → ℝ}
    (hn : Measurable (Function.uncurry next)) (hc : Measurable cost)
    (hc0 : ∀ s, 0 ≤ cost s) (T : ℕ) (s : S)
    (hi : Integrable (fun ξ => ((run next T s ξ).map cost).sum) (pathMeasure μ T))
    (t : Fin T) : Integrable (fun ξ => cost (pathState next T s ξ t)) (pathMeasure μ T) := by
  have hp : Measurable (fun ξ : Samples Ω T => (s, ξ)) := measurable_const.prodMk measurable_id
  have hm := hc.comp ((pathState_measurable hn T t).comp hp)
  apply hi.mono' hm.aestronglyMeasurable
  apply Filter.Eventually.of_forall
  intro ξ
  change ‖cost (pathState next T s ξ t)‖ ≤ _
  rw [Real.norm_of_nonneg (hc0 _), sum_run_eq_states]
  exact Finset.single_le_sum (fun k (_ : k ∈ Finset.univ) => hc0 _) (Finset.mem_univ t)

def expectedMinimum (μ : Measure Ω) (next : S → Ω → S) (cost : S → ℝ)
    (s : S) (T : ℕ) : ℝ :=
  (Finset.univ : Finset (Fin (T+1))).inf' ⟨0, Finset.mem_univ _⟩
    (fun t => ∫ ξ, cost (pathState next (T+1) s ξ t) ∂pathMeasure μ (T+1))

omit [MeasurableSpace S] in
theorem expectedMinimum_nonneg (μ : Measure Ω) (next : S → Ω → S) (cost : S → ℝ)
    (hc0 : ∀ s, 0 ≤ cost s) (s : S) (T : ℕ) : 0 ≤ expectedMinimum μ next cost s T := by
  apply Finset.le_inf'
  intro t _
  exact integral_nonneg fun _ => hc0 _

theorem expectedMinimum_le_average (μ : Measure Ω) {next : S → Ω → S} {cost : S → ℝ}
    (hn : Measurable (Function.uncurry next)) (hc : Measurable cost)
    (hc0 : ∀ s, 0 ≤ cost s) (s : S) (T : ℕ)
    (hi : Integrable (fun ξ => ((run next (T+1) s ξ).map cost).sum) (pathMeasure μ (T+1))) :
    expectedMinimum μ next cost s T ≤
      (∫ ξ, ((run next (T+1) s ξ).map cost).sum ∂pathMeasure μ (T+1))/(T+1:ℕ) := by
  have hint t := path_cost_integrable hn hc hc0 (T+1) s hi t
  have he : (∫ ξ, ((run next (T+1) s ξ).map cost).sum ∂pathMeasure μ (T+1)) =
      ∑ t, ∫ ξ, cost (pathState next (T+1) s ξ t) ∂pathMeasure μ (T+1) := by
    simp_rw [sum_run_eq_states]
    exact integral_finsetSum _ (fun t _ => hint t)
  have hmin : ∀ t : Fin (T+1), expectedMinimum μ next cost s T ≤
      ∫ ξ, cost (pathState next (T+1) s ξ t) ∂pathMeasure μ (T+1) := by
    intro t
    exact Finset.inf'_le _ (Finset.mem_univ t)
  have hs := Finset.sum_le_sum (fun t (_ : t ∈ Finset.univ) => hmin t)
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hs
  rw [← he] at hs
  apply (le_div_iff₀ (show (0:ℝ) < (T+1:ℕ) by positivity)).mpr
  simpa only [mul_comm] using hs

end EF21Muon.Probability
