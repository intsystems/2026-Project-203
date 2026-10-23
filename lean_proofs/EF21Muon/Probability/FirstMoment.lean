import EF21Muon.Probability.Updates

/-! First-moment tracking with a separate EMA of centered noise.
The noise-free tracking component contracts without a fresh-noise term;
the noise EMA retains the squared-norm variance recursion. -/

noncomputable section

namespace EF21Muon.Probability

open MeasureTheory

variable {Ω E : Type*} [MeasurableSpace Ω]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

omit [InnerProductSpace ℝ E] [CompleteSpace E] [MeasurableSpace E]
  [BorelSpace E] [SecondCountableTopology E] in
theorem integral_norm_sq_le {X : Ω → E} (hX : MemLp X 2 μ) :
    (∫ ω, ‖X ω‖ ∂μ)^2 ≤ ∫ ω, ‖X ω‖^2 ∂μ := by
  let a := ∫ ω, ‖X ω‖ ∂μ
  have hn := (hX.integrable (by norm_num)).norm
  have hs := square_integrable hX
  have hp : 0 ≤ ∫ ω, (‖X ω‖-a)^2 ∂μ := integral_nonneg (fun _ => sq_nonneg _)
  have he (ω : Ω) : (‖X ω‖-a)^2 = ‖X ω‖^2 + (-2*a)*‖X ω‖ + a^2 := by ring
  simp_rw [he] at hp
  have hm : Integrable (fun ω => (-2*a)*‖X ω‖) μ := hn.const_mul (-2*a)
  have hsum : Integrable (fun ω => ‖X ω‖^2 + (-2*a)*‖X ω‖) μ := hs.add hm
  rw [integral_add hsum (integrable_const (a^2)), integral_add hs hm,
    integral_const_mul, integral_const] at hp
  simp only [probReal_univ, one_smul] at hp
  change a^2 ≤ _
  change 0 ≤ (∫ ω, ‖X ω‖^2 ∂μ) + (-2*a)*a + a^2 at hp
  nlinarith only [hp]

omit [InnerProductSpace ℝ E] [CompleteSpace E] [MeasurableSpace E]
  [BorelSpace E] [SecondCountableTopology E] in
theorem integral_norm_le_of_sq {X : Ω → E} (hX : MemLp X 2 μ)
    {σ : ℝ} (hσ : 0 ≤ σ) (hv : (∫ ω, ‖X ω‖^2 ∂μ) ≤ σ^2) :
    (∫ ω, ‖X ω‖ ∂μ) ≤ σ :=
  (sq_le_sq₀ (integral_nonneg fun _ => norm_nonneg _) hσ).mp
    ((integral_norm_sq_le hX).trans hv)

omit [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem noise_ema_variance (β : ℝ) (Z : E) {noise : Ω → E}
    (hn : MemLp noise 2 μ) (hz : ∫ ω, noise ω ∂μ = 0) :
    (∫ ω, ‖newMomentum β Z 0 noise ω‖^2 ∂μ) =
      (1-β)^2*‖Z‖^2 + β^2*(∫ ω, ‖noise ω‖^2 ∂μ) := by
  have h := affine_variance_identity ((1-β) • Z) β hn hz
  simpa only [newMomentum, zero_add, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs] using h

omit [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem noise_ema_recurrence {β v : ℝ} (βpos : 0 ≤ β) (βone : β ≤ 1)
    (Z : E) {noise : Ω → E} (hn : MemLp noise 2 μ)
    (hz : ∫ ω, noise ω ∂μ = 0) (hv : (∫ ω, ‖noise ω‖^2 ∂μ) ≤ v) :
    (∫ ω, ‖newMomentum β Z 0 noise ω‖^2 ∂μ) ≤
      (1-β)*‖Z‖^2 + β^2*v := by
  rw [noise_ema_variance β Z hn hz]
  have hm := mul_le_mul_of_nonneg_left hv (sq_nonneg β)
  have hr : (1-β)^2 ≤ 1-β := by nlinarith
  have ht := mul_le_mul_of_nonneg_right hr (sq_nonneg ‖Z‖)
  linarith only [hm, ht]

omit [MeasurableSpace Ω] [CompleteSpace E] [MeasurableSpace E] [BorelSpace E]
  [SecondCountableTopology E] [IsProbabilityMeasure μ] in
theorem debiased_momentum_identity (β : ℝ) (M Z old new : E) (noise : Ω → E) (ω : Ω) :
    newMomentum β M new noise ω - new - newMomentum β Z 0 noise ω =
      (1-β) • ((M-old-Z)+(old-new)) := by
  dsimp [newMomentum]
  module

omit [MeasurableSpace Ω] [CompleteSpace E] [MeasurableSpace E] [BorelSpace E]
  [SecondCountableTopology E] [IsProbabilityMeasure μ] in
theorem debiased_momentum_bound {β : ℝ} (hβ : β ≤ 1)
    (M Z old new : E) (noise : Ω → E) (ω : Ω) :
    ‖newMomentum β M new noise ω - new - newMomentum β Z 0 noise ω‖ ≤
      (1-β)*(‖M-old-Z‖ + ‖old-new‖) := by
  rw [debiased_momentum_identity, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (sub_nonneg.mpr hβ)]
  exact mul_le_mul_of_nonneg_left (norm_add_le _ _) (sub_nonneg.mpr hβ)

omit [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem momentum_first_moment {β σ : ℝ} (hβ : 0 ≤ β) (hβ1 : β ≤ 1)
    (M old new : E) {noise : Ω → E} (hn : MemLp noise 2 μ)
    (hσ : 0 ≤ σ) (hv : (∫ ω, ‖noise ω‖^2 ∂μ) ≤ σ^2) :
    (∫ ω, ‖newMomentum β M new noise ω-new‖ ∂μ) ≤
      (1-β)*‖M-old‖ + (1-β)*‖old-new‖ + β*σ := by
  have hleft := ((newMomentum_memLp β M new hn).sub (memLp_const new)).integrable
    (by norm_num)
  have hright := (integrable_const ((1-β)*‖M-old‖ + (1-β)*‖old-new‖)).add
    ((hn.integrable (by norm_num)).norm.const_mul β)
  have hp (ω : Ω) : ‖newMomentum β M new noise ω-new‖ ≤
      (1-β)*‖M-old‖ + (1-β)*‖old-new‖ + β*‖noise ω‖ := by
    have he : newMomentum β M new noise ω-new =
        (1-β) • ((M-old)+(old-new)) + β • noise ω := by dsimp [newMomentum]; module
    rw [he]
    have ht := norm_add_le ((1-β) • ((M-old)+(old-new))) (β • noise ω)
    simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg hβ,
      abs_of_nonneg (sub_nonneg.mpr hβ1)] at ht
    have hb := mul_le_mul_of_nonneg_left (norm_add_le (M-old) (old-new))
      (sub_nonneg.mpr hβ1)
    nlinarith only [ht, hb]
  have hi := integral_mono hleft.norm hright hp
  simp only [Pi.add_apply, Pi.sub_apply] at hi
  rw [integral_add (integrable_const ((1-β)*‖M-old‖ + (1-β)*‖old-new‖))
    ((hn.integrable (by norm_num)).norm.const_mul β),
    integral_const, integral_const_mul] at hi
  simp only [probReal_univ, one_smul] at hi
  exact hi.trans (add_le_add le_rfl
    (mul_le_mul_of_nonneg_left (integral_norm_le_of_sq hn hσ hv) hβ))

omit [InnerProductSpace ℝ E] [CompleteSpace E] [MeasurableSpace E]
  [BorelSpace E] [SecondCountableTopology E] in
theorem sqrt_contraction {C : E → E} {α : ℝ} (hα : α ≤ 1)
    (hc : ∀ z, ‖C z-z‖^2 ≤ (1-α)*‖z‖^2) (z : E) :
    ‖C z-z‖ ≤ Real.sqrt (1-α)*‖z‖ := by
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Real.sqrt_nonneg _) (norm_nonneg _))).mp
  rw [mul_pow, Real.sq_sqrt (sub_nonneg.mpr hα)]
  exact hc z

omit [CompleteSpace E] in
theorem compression_first_moment {β α σ : ℝ} (hβ : 0 ≤ β)
    (hα : 0 < α) (hα1 : α ≤ 1) (M G old new : E) {C : E → E}
    (hm : Measurable C) (hc : ∀ z, ‖C z-z‖^2 ≤ (1-α)*‖z‖^2)
    {noise : Ω → E} (hn : MemLp noise 2 μ) (hσ : 0 ≤ σ)
    (hv : (∫ ω, ‖noise ω‖^2 ∂μ) ≤ σ^2) :
    (∫ ω, ‖C (newMomentum β M new noise ω-G) -
      (newMomentum β M new noise ω-G)‖ ∂μ) ≤
      Real.sqrt (1-α)*‖M-G‖ + Real.sqrt (1-α)*β*‖M-old‖ +
        Real.sqrt (1-α)*β*‖old-new‖ + Real.sqrt (1-α)*β*σ := by
  let q := Real.sqrt (1-α)
  have hq : 0 ≤ q := Real.sqrt_nonneg _
  have hx := (newMomentum_memLp β M new hn).sub (memLp_const G)
  have hl := (residual_memLp hm hα hc hx).integrable (by norm_num)
  have hr := (integrable_const (q*‖M-G‖ + q*β*‖M-old‖ + q*β*‖old-new‖)).add
    ((hn.integrable (by norm_num)).norm.const_mul (q*β))
  have hp (ω : Ω) : ‖C (newMomentum β M new noise ω-G) -
      (newMomentum β M new noise ω-G)‖ ≤
      q*‖M-G‖ + q*β*‖M-old‖ + q*β*‖old-new‖ + q*β*‖noise ω‖ := by
    have he : newMomentum β M new noise ω-G =
        (M-G) + β • ((old-M)+(new-old)+noise ω) := by dsimp [newMomentum]; module
    have ht := norm_add_le ((old-M)+(new-old)) (noise ω)
    have ht' := norm_add_le (old-M) (new-old)
    have hb : ‖newMomentum β M new noise ω-G‖ ≤
        ‖M-G‖ + β*(‖M-old‖ + ‖old-new‖ + ‖noise ω‖) := by
      rw [he]
      have hh := norm_add_le (M-G) (β • ((old-M)+(new-old)+noise ω))
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hβ] at hh
      rw [norm_sub_rev old M, norm_sub_rev new old] at ht'
      have ht'' : ‖(old-M)+(new-old)+noise ω‖ ≤
          ‖M-old‖ + ‖old-new‖ + ‖noise ω‖ := by linarith only [ht, ht']
      exact hh.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left ht'' hβ))
    have hh := (sqrt_contraction hα1 hc _).trans (mul_le_mul_of_nonneg_left hb hq)
    dsimp [q] at *
    nlinarith only [hh]
  have hi := integral_mono hl.norm hr hp
  simp only [Pi.add_apply, Pi.sub_apply] at hi
  rw [integral_add (integrable_const (q*‖M-G‖ + q*β*‖M-old‖ + q*β*‖old-new‖))
    ((hn.integrable (by norm_num)).norm.const_mul (q*β)),
    integral_const, integral_const_mul] at hi
  simp only [probReal_univ, one_smul] at hi
  exact hi.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left
    (integral_norm_le_of_sq hn hσ hv) (mul_nonneg hq hβ)))

omit [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem norm_average_le {N : ℕ} (z : Fin N → E) :
    ‖average z‖ ≤ average (fun j => ‖z j‖) := by
  rw [average, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (inv_nonneg.mpr (Nat.cast_nonneg N))]
  exact mul_le_mul_of_nonneg_left (norm_sum_le _ _) (inv_nonneg.mpr (Nat.cast_nonneg N))

end EF21Muon.Probability
