import EF21Muon.Stochastic.SquaredNorm
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Probability.Independence.Integration

noncomputable section

namespace EF21Muon.Probability

open MeasureTheory ProbabilityTheory
open scoped BigOperators

variable {Ω E : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
    {p : Measure Ω} [IsProbabilityMeasure p]

omit [InnerProductSpace ℝ E] [CompleteSpace E] [MeasurableSpace E] [BorelSpace E]
  [SecondCountableTopology E] [IsProbabilityMeasure p] in
theorem square_integrable {X : Ω → E} (hX : MemLp X 2 p) :
    Integrable (fun ω => ‖X ω‖ ^ 2) p :=
  (memLp_two_iff_integrable_sq_norm hX.aestronglyMeasurable).mp hX

omit [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
  [IsProbabilityMeasure p] in
theorem inner_integrable {X Y : Ω → E} (hX : MemLp X 2 p) (hY : MemLp Y 2 p) :
    Integrable (fun ω => inner ℝ (X ω) (Y ω)) p := by
  apply ((square_integrable hX).add (square_integrable hY)).mono'
    (hX.aestronglyMeasurable.inner hY.aestronglyMeasurable)
  filter_upwards [] with ω
  change ‖inner ℝ (X ω) (Y ω)‖ ≤ ‖X ω‖ ^ 2 + ‖Y ω‖ ^ 2
  have hi := norm_inner_le_norm (𝕜 := ℝ) (X ω) (Y ω)
  nlinarith only [hi, sq_nonneg (‖X ω‖ - ‖Y ω‖), sq_nonneg ‖X ω‖, sq_nonneg ‖Y ω‖]

omit [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem affine_variance_identity (a : E) (β : ℝ) {noise : Ω → E}
    (hnoise : MemLp noise 2 p) (hzero : ∫ ω, noise ω ∂p = 0) :
    (∫ ω, ‖a + β • noise ω‖ ^ 2 ∂p) =
      ‖a‖ ^ 2 + β ^ 2 * ∫ ω, ‖noise ω‖ ^ 2 ∂p := by
  have hi : Integrable (fun ω => inner ℝ a (noise ω)) p :=
    (hnoise.integrable (by norm_num)).const_inner a
  have hs := square_integrable hnoise
  have he (ω : Ω) : ‖a + β • noise ω‖ ^ 2 =
      ‖a‖ ^ 2 + 2 * β * inner ℝ a (noise ω) + β ^ 2 * ‖noise ω‖ ^ 2 := by
    rw [norm_add_sq_real, inner_smul_right, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
    ring
  simp_rw [he]
  have hsum : Integrable (fun ω => ‖a‖ ^ 2 + 2 * β * inner ℝ a (noise ω)) p :=
    (integrable_const _).add (hi.const_mul _)
  rw [integral_add hsum (hs.const_mul _),
    integral_add (integrable_const _) (hi.const_mul _), integral_const_mul,
    integral_const_mul, integral_inner (hnoise.integrable (by norm_num)), hzero]
  simp

def average {N : ℕ} (z : Fin N → E) : E := (N : ℝ)⁻¹ • ∑ j, z j

omit [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem average_add {N : ℕ} (x y : Fin N → E) :
    average (fun j => x j + y j) = average x + average y := by
  simp [average, Finset.sum_add_distrib, smul_add]

omit [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem average_sub {N : ℕ} (x y : Fin N → E) :
    average (fun j => x j - y j) = average x - average y := by
  simp [average, Finset.sum_sub_distrib, smul_sub]

omit [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem average_const {N : ℕ} (hN : 0 < N) (x : E) : average (fun _ : Fin N => x) = x := by
  have hNr : (N : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hN
  simp [average, ← Nat.cast_smul_eq_nsmul ℝ, smul_smul, hNr]

omit [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E] in
theorem average_sq_le {N : ℕ} (hN : 0 < N) (z : Fin N → E) :
    ‖average z‖ ^ 2 ≤ (N : ℝ)⁻¹ * ∑ j, ‖z j‖ ^ 2 := by
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  have hnorm := norm_sum_le Finset.univ z
  have hsq := mul_self_le_mul_self (norm_nonneg _) hnorm
  have hc := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun j : Fin N => ‖z j‖)
    (fun _ => (1 : ℝ))
  simp only [mul_one, one_pow, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul] at hc
  rw [average, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
  have hm := mul_le_mul_of_nonneg_left (show ‖∑ j, z j‖ ^ 2 ≤ N * ∑ j, ‖z j‖ ^ 2 by
    nlinarith only [hsq, hc]) (sq_nonneg (N : ℝ)⁻¹)
  convert hm using 1; field_simp

omit [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
  [IsProbabilityMeasure p] in
theorem average_memLp {N : ℕ} {z : Fin N → Ω → E} (hz : ∀ j, MemLp (z j) 2 p) :
    MemLp (fun ω => average (fun j => z j ω)) 2 p :=
  (memLp_finsetSum Finset.univ (fun j _ => hz j)).const_smul _

omit [CompleteSpace E] [MeasurableSpace E] [BorelSpace E] [SecondCountableTopology E]
  [IsProbabilityMeasure p] in
theorem sum_variance_identity {N : ℕ} {z : Fin N → Ω → E}
    (hz : ∀ j, MemLp (z j) 2 p)
    (hcross : ∀ j k, j ≠ k → (∫ ω, inner ℝ (z j ω) (z k ω) ∂p) = 0) :
    (∫ ω, ‖∑ j, z j ω‖ ^ 2 ∂p) = ∑ j, ∫ ω, ‖z j ω‖ ^ 2 ∂p := by
  have he (ω : Ω) : ‖∑ j, z j ω‖ ^ 2 = ∑ j, ∑ k, inner ℝ (z j ω) (z k ω) := by
    rw [← real_inner_self_eq_norm_sq, sum_inner]
    simp only [inner_sum]
  simp_rw [he]
  rw [integral_finsetSum _ (fun j _ =>
    integrable_finsetSum _ (fun k _ => inner_integrable (hz j) (hz k)))]
  apply Finset.sum_congr rfl
  intro j _
  rw [integral_finsetSum _ (fun k _ => inner_integrable (hz j) (hz k))]
  rw [Finset.sum_eq_single j]
  · simp only [real_inner_self_eq_norm_sq]
  · intro k _ hkj
    exact hcross j k (Ne.symm hkj)
  · simp

omit [SecondCountableTopology E] in
/-- Independent, centered workers give the actual 1/N variance reduction. -/
theorem independent_average_variance {N : ℕ} (hN : 0 < N) {z : Fin N → Ω → E}
    (hz : ∀ j, MemLp (z j) 2 p) (hind : iIndepFun z p)
    (hzero : ∀ j, ∫ ω, z j ω ∂p = 0) {σ : ℝ}
    (hvar : ∀ j, (∫ ω, ‖z j ω‖ ^ 2 ∂p) ≤ σ ^ 2) :
    (∫ ω, ‖average (fun j => z j ω)‖ ^ 2 ∂p) ≤ σ ^ 2 / N := by
  have hc (j k : Fin N) (hjk : j ≠ k) : (∫ ω, inner ℝ (z j ω) (z k ω) ∂p) = 0 := by
    have h := (hind.indepFun hjk).integral_bilin
      ((hz j).integrable (by norm_num)) ((hz k).integrable (by norm_num)) (innerSL ℝ)
    change (∫ ω, inner ℝ (z j ω) (z k ω) ∂p) =
      inner ℝ (∫ ω, z j ω ∂p) (∫ ω, z k ω ∂p) at h
    simpa only [hzero, inner_zero_left] using h
  have hsum := sum_variance_identity hz hc
  have hbound := Finset.sum_le_sum (fun j (_ : j ∈ Finset.univ) => hvar j)
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hbound
  have he (ω : Ω) : ‖average (fun j => z j ω)‖ ^ 2 =
      (N : ℝ)⁻¹ ^ 2 * ‖∑ j, z j ω‖ ^ 2 := by
    simp [average, norm_smul, mul_pow]
  simp_rw [he]
  rw [integral_const_mul, hsum]
  have h := mul_le_mul_of_nonneg_left hbound (sq_nonneg (N : ℝ)⁻¹)
  have hNr : (N : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hN
  convert h using 1; field_simp

end EF21Muon.Probability
