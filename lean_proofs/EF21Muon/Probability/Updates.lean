import EF21Muon.Probability.Moments

noncomputable section

namespace EF21Muon.Probability

open MeasureTheory
open EF21Muon.Stochastic

variable {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
    {p : Measure Ω} [IsProbabilityMeasure p]

/-- The fresh stochastic momentum update, with its conditional mean exposed. -/
def newMomentum (β : ℝ) (M q : E) (noise : Ω → E) (ω : Ω) : E :=
  (1 - β) • M + β • (q + noise ω)

omit [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem newMomentum_memLp (β : ℝ) (M q : E) {noise : Ω → E}
    (hn : MemLp noise 2 p) : MemLp (newMomentum β M q noise) 2 p := by
  have hM : MemLp (fun _ : Ω => (1 - β) • M) 2 p := memLp_const _
  have hq : MemLp (fun _ : Ω => q) 2 p := memLp_const _
  exact hM.add ((hq.add hn).const_smul β)

omit [InnerProductSpace ℝ E] [CompleteSpace E] [IsProbabilityMeasure p] in
/-- L2 integrability of the actual compression residual follows from contraction. -/
theorem residual_memLp {C : E → E} (hm : Measurable C) {α : ℝ} (hα : 0 < α)
    (hc : ∀ z, ‖C z - z‖ ^ 2 ≤ (1 - α) * ‖z‖ ^ 2)
    {X : Ω → E} (hX : MemLp X 2 p) : MemLp (fun ω => C (X ω) - X ω) 2 p := by
  apply hX.of_le ((hm.comp_aemeasurable hX.aestronglyMeasurable.aemeasurable).aestronglyMeasurable.sub
    hX.aestronglyMeasurable)
  filter_upwards [] with ω
  change ‖C (X ω) - X ω‖ ≤ ‖X ω‖
  have h := hc (X ω)
  apply (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  nlinarith only [h, mul_nonneg hα.le (sq_nonneg ‖X ω‖)]

omit [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem momentum_recurrence {β v : ℝ} (hβ : 0 < β) (hβ1 : β ≤ 1)
    (M old new query : E) {noise : Ω → E} (hn : MemLp noise 2 p)
    (hz : ∫ ω, noise ω ∂p = 0) (hv : (∫ ω, ‖noise ω‖ ^ 2 ∂p) ≤ v) :
    (∫ ω, ‖newMomentum β M query noise ω - new‖ ^ 2 ∂p) ≤
      (1 - β / 2) * ‖M - old‖ ^ 2 + 2 / β * ‖old - new‖ ^ 2 +
        β ^ 2 * (1 + 2 / β) * ‖query - new‖ ^ 2 + β ^ 2 * v := by
  let a := (1 - β) • ((M - old) + (old - new)) + β • (query - new)
  have he (ω : Ω) : newMomentum β M query noise ω - new = a + β • noise ω := by
    dsimp [newMomentum, a]; module
  simp_rw [he]
  rw [affine_variance_identity a β hn hz]
  have hy := momentum_mean_sq (M - old) (old - new) (query - new) hβ hβ1
  have hh := mul_le_mul_of_nonneg_left hv (sq_nonneg β)
  change ‖a‖ ^ 2 ≤ _ at hy
  linarith only [hy, hh]

theorem uplink_recurrence {β α v : ℝ} (hα : 0 < α) (hα1 : α ≤ 1)
    (M G old new query : E) {C : E → E} (hm : Measurable C)
    (hc : ∀ z, ‖C z - z‖ ^ 2 ≤ (1 - α) * ‖z‖ ^ 2)
    {noise : Ω → E} (hn : MemLp noise 2 p) (hz : ∫ ω, noise ω ∂p = 0)
    (hv : (∫ ω, ‖noise ω‖ ^ 2 ∂p) ≤ v) :
    (∫ ω, ‖C (newMomentum β M query noise ω - G) -
      (newMomentum β M query noise ω - G)‖ ^ 2 ∂p) ≤
      (1 - α / 2) * ‖M - G‖ ^ 2 +
        6 * β ^ 2 / α * (‖M - old‖ ^ 2 + ‖old - new‖ ^ 2 + ‖query - new‖ ^ 2) +
          (1 - α) * β ^ 2 * v := by
  have hX := (newMomentum_memLp β M query hn).sub (memLp_const G)
  have hi := square_integrable (residual_memLp hm hα hc hX)
  have hb := integral_mono hi ((square_integrable hX).const_mul (1 - α)) (fun ω => hc _)
  change (∫ ω, ‖C (newMomentum β M query noise ω - G) -
    (newMomentum β M query noise ω - G)‖ ^ 2 ∂p) ≤
      ∫ ω, (1 - α) * ‖newMomentum β M query noise ω - G‖ ^ 2 ∂p at hb
  rw [integral_const_mul] at hb
  let a := (1 - β) • M + β • query - G
  have he (ω : Ω) : newMomentum β M query noise ω - G = a + β • noise ω := by
    dsimp [newMomentum, a]; module
  simp_rw [he] at hb
  rw [affine_variance_identity a β hn hz] at hb
  have hy := contracted_add_sq (M - G) (β • (query - M)) hα hα1
  have ha : (M - G) + β • (query - M) = a := by dsimp [a]; module
  rw [ha, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs] at hy
  have ht := three_sq (query - new) (new - old) (old - M)
  rw [sub_add_sub_cancel, sub_add_sub_cancel, norm_sub_rev new old, norm_sub_rev old M] at ht
  have ht' := mul_le_mul_of_nonneg_left ht (by positivity : 0 ≤ 2 / α * β ^ 2)
  have hv' := mul_le_mul_of_nonneg_left hv (mul_nonneg (sub_nonneg.mpr hα1) (sq_nonneg β))
  simp_rw [he]
  simp only [div_eq_mul_inv] at hy ht' ⊢
  nlinarith only [hb, hy, ht', hv']

end EF21Muon.Probability
