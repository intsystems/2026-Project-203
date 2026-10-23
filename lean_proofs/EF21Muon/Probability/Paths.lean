import EF21Muon.Stochastic.FiniteSampling
import EF21Muon.Probability.Moments

noncomputable section

namespace EF21Muon.Probability

open MeasureTheory
open EF21Muon.Stochastic (Samples run)

variable {Ω S : Type*} [MeasurableSpace Ω] [MeasurableSpace S]

instance samplesMeasurableSpace : (T : ℕ) → MeasurableSpace (Samples Ω T)
  | 0 => inferInstanceAs (MeasurableSpace PUnit)
  | T + 1 => by
    letI := samplesMeasurableSpace T
    exact inferInstanceAs (MeasurableSpace (Ω × Samples Ω T))

def pathMeasure (p : Measure Ω) : (T : ℕ) → Measure (Samples Ω T)
  | 0 => Measure.dirac PUnit.unit
  | T + 1 => p.prod (pathMeasure p T)

instance pathMeasure_probability (p : Measure Ω) [IsProbabilityMeasure p] :
    (T : ℕ) → IsProbabilityMeasure (pathMeasure p T)
  | 0 => inferInstanceAs (IsProbabilityMeasure (Measure.dirac PUnit.unit))
  | T + 1 => by
    let := pathMeasure_probability p T
    exact inferInstanceAs (IsProbabilityMeasure (p.prod (pathMeasure p T)))

def terminal (next : S → Ω → S) : (T : ℕ) → S → Samples Ω T → S
  | 0, s, _ => s
  | T + 1, s, ξ => terminal next T (next s ξ.1) ξ.2

def totalPotential (next : S → Ω → S) (cost V : S → ℝ) :
    (T : ℕ) → S → Samples Ω T → ℝ
  | 0, s, _ => V s
  | T + 1, s, ξ => cost s + totalPotential next cost V T (next s ξ.1) ξ.2

omit [MeasurableSpace Ω] [MeasurableSpace S] in
theorem totalPotential_eq (next : S → Ω → S) (cost V : S → ℝ)
    (T : ℕ) (s : S) (ξ : Samples Ω T) :
    totalPotential next cost V T s ξ =
      ((run next T s ξ).map cost).sum + V (terminal next T s ξ) := by
  induction T generalizing s with
  | zero => simp [totalPotential, run, terminal]
  | succ T ih => simp [totalPotential, run, terminal, ih, add_assoc]

omit [MeasurableSpace Ω] [MeasurableSpace S] in
theorem totalPotential_nonneg {next : S → Ω → S} {cost V : S → ℝ}
    (hc : ∀ s, 0 ≤ cost s) (hV : ∀ s, 0 ≤ V s) (T : ℕ) (s : S) (ξ : Samples Ω T) :
    0 ≤ totalPotential next cost V T s ξ := by
  induction T generalizing s with
  | zero => exact hV s
  | succ T ih => exact add_nonneg (hc s) (ih _ _)

theorem totalPotential_measurable {next : S → Ω → S} {cost V : S → ℝ}
    (hn : Measurable (Function.uncurry next)) (hc : Measurable cost) (hV : Measurable V)
    (T : ℕ) : Measurable (Function.uncurry (totalPotential next cost V T)) := by
  induction T with
  | zero => exact hV.comp measurable_fst
  | succ T ih =>
    change Measurable (fun z : S × (Ω × Samples Ω T) =>
      cost z.1 + totalPotential next cost V T (next z.1 z.2.1) z.2.2)
    exact (hc.comp measurable_fst).add
      (ih.comp ((hn.comp (measurable_fst.prodMk measurable_snd.fst)).prodMk measurable_snd.snd))

theorem kernel_integrable_nonneg {A B : Type*} [MeasurableSpace A] [MeasurableSpace B]
    {μ : Measure A} {ν : Measure B} [SFinite ν] {F : A × B → ℝ} {M : A → ℝ}
    (hF : Measurable F) (hpos : ∀ z, 0 ≤ F z)
    (hint : ∀ a, Integrable (fun b => F (a, b)) ν)
    (hbound : ∀ a, (∫ b, F (a, b) ∂ν) ≤ M a) (hM : Integrable M μ) :
    Integrable F (μ.prod ν) := by
  apply (integrable_prod_iff hF.aestronglyMeasurable).mpr
  refine ⟨Filter.Eventually.of_forall hint, ?_⟩
  apply hM.mono' (hF.norm.stronglyMeasurable.integral_prod_right').aestronglyMeasurable
  filter_upwards [] with a
  rw [Real.norm_eq_abs, abs_of_nonneg (integral_nonneg fun b => norm_nonneg (F (a, b)))]
  have he : (fun b => ‖F (a, b)‖) = (fun b => F (a, b)) :=
    funext (fun b => Real.norm_of_nonneg (hpos (a, b)))
  rw [he]
  exact hbound a

/-- Integrability is proved together with the drift bound, for arbitrary noise laws. -/
theorem totalPotential_integral_bound (p : Measure Ω) [IsProbabilityMeasure p]
    {next : S → Ω → S} {cost V : S → ℝ} {b : ℝ}
    (hn : Measurable (Function.uncurry next)) (hc : Measurable cost) (hVm : Measurable V)
    (hc0 : ∀ s, 0 ≤ cost s) (hV0 : ∀ s, 0 ≤ V s)
    (hint : ∀ s, Integrable (fun ω => V (next s ω)) p)
    (hdrift : ∀ s, (∫ ω, V (next s ω) ∂p) + cost s ≤ V s + b)
    (T : ℕ) (s : S) :
    Integrable (totalPotential next cost V T s) (pathMeasure p T) ∧
      (∫ ξ, totalPotential next cost V T s ξ ∂pathMeasure p T) ≤ V s + (T : ℝ) * b := by
  induction T generalizing s with
  | zero =>
    refine ⟨integrable_const _, ?_⟩
    simp only [totalPotential, Nat.cast_zero, zero_mul, add_zero, integral_const,
      probReal_univ, one_smul, le_refl]
  | succ T ih =>
    let F : Ω × Samples Ω T → ℝ := fun z => totalPotential next cost V T (next s z.1) z.2
    have hF : Measurable F := (totalPotential_measurable hn hc hVm T).comp
      ((hn.comp (measurable_const.prodMk measurable_fst)).prodMk measurable_snd)
    have hiF : Integrable F (p.prod (pathMeasure p T)) := kernel_integrable_nonneg hF
      (fun z => totalPotential_nonneg hc0 hV0 _ _ _) (fun ω => (ih (next s ω)).1)
      (fun ω => (ih (next s ω)).2) ((hint s).add (integrable_const ((T : ℝ) * b)))
    have hi : Integrable (fun ξ : Samples Ω (T + 1) =>
        totalPotential next cost V (T + 1) s ξ) (pathMeasure p (T + 1)) :=
      (integrable_const (cost s)).add hiF
    refine ⟨hi, ?_⟩
    change (∫ ξ, cost s + F ξ ∂p.prod (pathMeasure p T)) ≤ _
    rw [integral_add (integrable_const _) hiF, integral_const, integral_prod F hiF]
    have hbound := integral_mono hiF.integral_prod_left
      ((hint s).add (integrable_const ((T : ℝ) * b))) (fun ω => (ih (next s ω)).2)
    simp only [Pi.add_apply] at hbound
    rw [integral_add (hint s) (integrable_const _)] at hbound
    simp only [probReal_univ, smul_eq_mul, one_mul, integral_const] at *
    push_cast
    linarith [hdrift s]

end EF21Muon.Probability
