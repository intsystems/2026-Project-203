import EF21Muon.Probability.GeometricDrift

/-! Lifting finite families of conditional drifts to fresh random choices.
The transition itself may change, provided its new potential agrees pointwise
with the corresponding member of the proved family. -/

noncomputable section
namespace EF21Muon.Probability
open MeasureTheory
open EF21Muon.Stochastic

variable {B Ω S : Type*} [Fintype B] [MeasurableSpace B] [MeasurableSingletonClass B]
    [MeasurableSpace Ω] [MeasurableSpace S]

theorem measurable_finite_choice {f : S → B × Ω → S}
    (hf : ∀ b, Measurable (fun z : S × Ω => f z.1 (b,z.2))) :
    Measurable (Function.uncurry f) := by
  have h : Measurable (fun z : B × (S × Ω) => f z.2.1 (z.1,z.2.2)) :=
    measurable_from_prod_countable_right hf
  exact h.comp (measurable_snd.fst.prodMk (measurable_fst.prodMk measurable_snd.snd))

theorem measurable_delayed_choice {f : S × B → B × Ω → S × B}
    (hf : ∀ old fresh, Measurable (fun z : S × Ω => f (z.1,old) (fresh,z.2))) :
    Measurable (Function.uncurry f) := by
  have h : Measurable (fun z : (B × B) × (S × Ω) =>
      f (z.2.1,z.1.1) (z.1.2,z.2.2)) :=
    measurable_from_prod_countable_right (fun b => hf b.1 b.2)
  exact h.comp ((measurable_fst.snd.prodMk measurable_snd.fst).prodMk
    (measurable_fst.fst.prodMk measurable_snd.snd))

theorem finite_choice_drift (ν : Measure B) (μ : Measure Ω)
    [IsProbabilityMeasure ν] [IsProbabilityMeasure μ]
    {f : S → B × Ω → S} {base : B → S → Ω → S} {V cost : S → ℝ} {a b : ℝ}
    (hf : Measurable (Function.uncurry f)) (hV : Measurable V) (hV0 : ∀ s, 0 ≤ V s)
    (he : ∀ s k ω, V (f s (k,ω)) = V (base k s ω))
    (hi : ∀ k s, Integrable (fun ω => V (base k s ω)) μ)
    (hd : ∀ k s, (∫ ω, V (base k s ω) ∂μ)+cost s ≤ a*V s+b)
    (s : S) :
    Integrable (fun z => V (f s z)) (ν.prod μ) ∧
      (∫ z, V (f s z) ∂ν.prod μ)+cost s ≤ a*V s+b := by
  let F : B × Ω → ℝ := fun z => V (f s z)
  have hFm : Measurable F := hV.comp (hf.comp (measurable_const.prodMk measurable_id))
  have hFi k : Integrable (fun ω => F (k,ω)) μ := by
    simpa only [F, he] using hi k s
  have hF : Integrable F (ν.prod μ) := kernel_integrable_nonneg hFm
    (fun z => hV0 _) hFi (fun _ => le_rfl) (Integrable.of_finite)
  refine ⟨hF, ?_⟩
  change (∫ z, F z ∂ν.prod μ)+cost s ≤ _
  rw [integral_prod F hF]
  have hbound : (∫ k, (∫ ω, F (k,ω) ∂μ)+cost s ∂ν) ≤ ∫ _ : B, a*V s+b ∂ν :=
    integral_mono (hF.integral_prod_left.add (integrable_const _)) (integrable_const _)
      (fun k => by simpa only [F, he] using hd k s)
  rw [integral_add hF.integral_prod_left (integrable_const _)] at hbound
  simpa only [integral_const, probReal_univ, one_smul] using hbound

theorem finite_choice_convergence (ν : Measure B) (μ : Measure Ω)
    [IsProbabilityMeasure ν] [IsProbabilityMeasure μ]
    {f : S → B × Ω → S} {base : B → S → Ω → S} {V cost : S → ℝ} {a b : ℝ}
    (ha : 1 ≤ a) (hb : 0 ≤ b)
    (hf : Measurable (Function.uncurry f)) (hcost : Measurable cost)
    (hV : Measurable V) (hcost0 : ∀ s, 0 ≤ cost s) (hV0 : ∀ s, 0 ≤ V s)
    (he : ∀ s k ω, V (f s (k,ω)) = V (base k s ω))
    (hi : ∀ k s, Integrable (fun ω => V (base k s ω)) μ)
    (hd : ∀ k s, (∫ ω, V (base k s ω) ∂μ)+cost s ≤ a*V s+b)
    (T : ℕ) (s : S) :
    Integrable (fun ξ => ((run f T s ξ).map cost).sum) (pathMeasure (ν.prod μ) T) ∧
      (∫ ξ, ((run f T s ξ).map cost).sum ∂pathMeasure (ν.prod μ) T) ≤
        a^T*(V s+(T:ℝ)*b) :=
  accumulated_cost_geometric_bound (ν.prod μ) ha hb hf hcost hV hcost0 hV0
    (fun s => (finite_choice_drift ν μ hf hV hV0 he hi hd s).1)
    (fun s => (finite_choice_drift ν μ hf hV hV0 he hi hd s).2) T s

end EF21Muon.Probability
