import EF21Muon.Probability.Paths

/-! Finite-horizon stochastic drift with a multiplicative objective term.
Integrability is proved along with the bound, for arbitrary probability laws. -/

noncomputable section

namespace EF21Muon.Probability

open MeasureTheory
open EF21Muon.Stochastic (Samples run)

variable {Ω S : Type*} [MeasurableSpace Ω] [MeasurableSpace S]

theorem totalPotential_geometric_bound (μ : Measure Ω) [IsProbabilityMeasure μ]
    {next : S → Ω → S} {cost V : S → ℝ} {a b : ℝ} (ha : 1 ≤ a) (hb : 0 ≤ b)
    (hn : Measurable (Function.uncurry next)) (hc : Measurable cost) (hVm : Measurable V)
    (hc0 : ∀ s, 0 ≤ cost s) (hV0 : ∀ s, 0 ≤ V s)
    (hint : ∀ s, Integrable (fun ω => V (next s ω)) μ)
    (hdrift : ∀ s, (∫ ω, V (next s ω) ∂μ) + cost s ≤ a*V s+b)
    (T : ℕ) (s : S) :
    Integrable (totalPotential next cost V T s) (pathMeasure μ T) ∧
      (∫ ξ, totalPotential next cost V T s ξ ∂pathMeasure μ T) ≤
        a^T*(V s+(T:ℝ)*b) := by
  have ha0 : 0 ≤ a := le_trans (by norm_num) ha
  induction T generalizing s with
  | zero =>
    refine ⟨integrable_const _, ?_⟩
    simp only [totalPotential, Nat.cast_zero, zero_mul, add_zero, pow_zero,
      one_mul, integral_const, probReal_univ, one_smul, le_refl]
  | succ T ih =>
    let F : Ω × Samples Ω T → ℝ := fun z => totalPotential next cost V T (next s z.1) z.2
    have hF : Measurable F := (totalPotential_measurable hn hc hVm T).comp
      ((hn.comp (measurable_const.prodMk measurable_fst)).prodMk measurable_snd)
    have hiM : Integrable (fun ω => a^T*(V (next s ω)+(T:ℝ)*b)) μ :=
      ((hint s).add (integrable_const _)).const_mul _
    have hiF : Integrable F (μ.prod (pathMeasure μ T)) := kernel_integrable_nonneg hF
      (fun _ => totalPotential_nonneg hc0 hV0 _ _ _) (fun ω => (ih (next s ω)).1)
      (fun ω => (ih (next s ω)).2) hiM
    have hi : Integrable (totalPotential next cost V (T+1) s) (pathMeasure μ (T+1)) :=
      (integrable_const (cost s)).add hiF
    refine ⟨hi, ?_⟩
    change (∫ ξ, cost s+F ξ ∂μ.prod (pathMeasure μ T)) ≤ _
    rw [integral_add (integrable_const _) hiF, integral_const, integral_prod F hiF]
    have hbound := integral_mono hiF.integral_prod_left hiM (fun ω => (ih (next s ω)).2)
    rw [integral_const_mul, integral_add (hint s) (integrable_const _)] at hbound
    simp only [probReal_univ, smul_eq_mul, one_mul, integral_const] at *
    have hpow : 1 ≤ a^T := one_le_pow₀ ha
    have hcost := mul_le_mul_of_nonneg_right hpow (hc0 s)
    have hd := mul_le_mul_of_nonneg_left (hdrift s) (pow_nonneg ha0 T)
    have hb' : 0 ≤ ((T:ℝ)+1)*b := mul_nonneg (by positivity) hb
    have hlast := mul_le_mul_of_nonneg_left ha hb'
    have hlast' := mul_le_mul_of_nonneg_left hlast (pow_nonneg ha0 T)
    rw [pow_succ]
    push_cast
    nlinarith only [hbound, hcost, hd, hlast']

/-- The accumulated nonnegative stationarity cost inherits the same bound. -/
theorem accumulated_cost_geometric_bound (μ : Measure Ω) [IsProbabilityMeasure μ]
    {next : S → Ω → S} {cost V : S → ℝ} {a b : ℝ} (ha : 1 ≤ a) (hb : 0 ≤ b)
    (hn : Measurable (Function.uncurry next)) (hc : Measurable cost) (hVm : Measurable V)
    (hc0 : ∀ s, 0 ≤ cost s) (hV0 : ∀ s, 0 ≤ V s)
    (hint : ∀ s, Integrable (fun ω => V (next s ω)) μ)
    (hdrift : ∀ s, (∫ ω, V (next s ω) ∂μ) + cost s ≤ a*V s+b)
    (T : ℕ) (s : S) :
    Integrable (fun ξ => ((run next T s ξ).map cost).sum) (pathMeasure μ T) ∧
      (∫ ξ, ((run next T s ξ).map cost).sum ∂pathMeasure μ T) ≤
        a^T*(V s+(T:ℝ)*b) := by
  have h := totalPotential_geometric_bound μ ha hb hn hc hVm hc0 hV0 hint hdrift T s
  have ht : ∀ k, Measurable (Function.uncurry (terminal next k)) := by
    intro k
    induction k with
    | zero => exact measurable_fst
    | succ k ih =>
      exact ih.comp
        ((hn.comp (measurable_fst.prodMk measurable_snd.fst)).prodMk measurable_snd.snd)
  have hpair : Measurable (fun ξ : Samples Ω T => (s, ξ)) :=
    measurable_const.prodMk measurable_id
  have hm : Measurable (fun ξ => ((run next T s ξ).map cost).sum) := by
    have hr := (totalPotential_measurable hn hc hVm T).comp hpair
    have hterminal := hVm.comp ((ht T).comp hpair)
    simpa only [Function.comp_def, Function.uncurry_def, Pi.sub_def, totalPotential_eq,
      add_sub_cancel_right] using hr.sub hterminal
  have hnonneg (ξ : Samples Ω T) : 0 ≤ ((run next T s ξ).map cost).sum := by
    apply List.sum_nonneg
    intro y hy
    obtain ⟨t, _, rfl⟩ := List.mem_map.mp hy
    exact hc0 t
  have hle (ξ : Samples Ω T) : ((run next T s ξ).map cost).sum ≤
      totalPotential next cost V T s ξ := by
    rw [totalPotential_eq]
    exact le_add_of_nonneg_right (hV0 _)
  have hi := h.1.mono' hm.aestronglyMeasurable (Filter.Eventually.of_forall
    fun ξ => (Real.norm_of_nonneg (hnonneg ξ)).trans_le (hle ξ))
  exact ⟨hi, (integral_mono hi h.1 hle).trans h.2⟩

end EF21Muon.Probability
