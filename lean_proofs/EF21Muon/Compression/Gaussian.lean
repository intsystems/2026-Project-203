import EF21Muon.Compression.Asymptotics
import Mathlib.Probability.Distributions.Gaussian.Real

/-! The Gaussian assertion in Lemma 5, with its exact interpretation: centered
iid entries of fixed positive variance, dimension tending to infinity, almost
sure convergence. Gaussian moments are derived from the actual Gaussian law. -/

noncomputable section

namespace EF21Muon.Compression

open MeasureTheory ProbabilityTheory Filter Real Set
open scoped Topology NNReal

private theorem integral_mul_exp_Ioi {b : ℝ} (hb : 0 < b) :
    ∫ x : ℝ in Ioi 0, x * exp (-b * x ^ 2) = (2 * b)⁻¹ := by
  have hd (x : ℝ) : HasDerivAt (fun x => -(2 * b)⁻¹ * exp (-b * x ^ 2))
      (x * exp (-b * x ^ 2)) x := by
    convert (((hasDerivAt_pow 2 x).const_mul (-b)).exp.const_mul (-(2 * b)⁻¹)) using 1
    field_simp
    ring
  have ht : Tendsto (fun y : ℝ => -(2 * b)⁻¹ * exp (-b * y ^ 2)) atTop (𝓝 0) := by
    have h := tendsto_exp_atBot.comp
      ((tendsto_pow_atTop (by norm_num : (2 : ℕ) ≠ 0)).const_mul_atTop_of_neg
        (neg_lt_zero.mpr hb))
    simpa using h.const_mul (-(2 * b)⁻¹)
  convert integral_Ioi_of_hasDerivAt_of_tendsto' (fun x _ => hd x)
    (integrable_mul_exp_neg_mul_sq hb).integrableOn ht using 1
  simp

theorem integral_abs_centered_gaussian {v : ℝ≥0} (hv : v ≠ 0) :
    (∫ x : ℝ, |x| ∂gaussianReal 0 v) = 2 * (v : ℝ) / sqrt (2 * π * v) := by
  have hv' : (0 : ℝ) < v := by exact_mod_cast (pos_iff_ne_zero.mpr hv)
  rw [integral_gaussianReal_eq_integral_smul hv]
  simp only [gaussianPDFReal, sub_zero, smul_eq_mul]
  have he : (fun x : ℝ => (sqrt (2 * π * v))⁻¹ * exp (-(x ^ 2) / (2 * v)) * |x|) =
      (fun x : ℝ => (sqrt (2 * π * v))⁻¹ *
        ((fun y : ℝ => y * exp (-(2 * (v : ℝ))⁻¹ * y ^ 2)) |x|)) := by
    funext x
    dsimp only
    rw [sq_abs]
    have hex : -(x ^ 2) / (2 * (v : ℝ)) = -(2 * (v : ℝ))⁻¹ * x ^ 2 := by ring
    rw [hex]
    ring
  rw [he, integral_const_mul,
    integral_comp_abs (f := fun y : ℝ => y * exp (-(2 * (v : ℝ))⁻¹ * y ^ 2)),
    integral_mul_exp_Ioi (inv_pos.mpr (by positivity))]
  field_simp

theorem integral_sq_centered_gaussian (v : ℝ≥0) :
    (∫ x : ℝ, x ^ 2 ∂gaussianReal 0 v) = v := by
  have h := variance_fun_id_gaussianReal (μ := 0) (v := v)
  rw [variance_eq_integral measurable_id'.aemeasurable] at h
  simpa using h

theorem gaussian_contraction_moment_ratio {v : ℝ≥0} (hv : v ≠ 0) :
    (∫ x : ℝ, |x| ∂gaussianReal 0 v) ^ 2 /
      (∫ x : ℝ, x ^ 2 ∂gaussianReal 0 v) = 2 / π := by
  rw [integral_abs_centered_gaussian hv, integral_sq_centered_gaussian]
  have hv' : (0 : ℝ) < v := by exact_mod_cast (pos_iff_ne_zero.mpr hv)
  have hs := Real.sq_sqrt (show 0 ≤ 2 * π * (v : ℝ) by positivity)
  have hp := Real.pi_pos
  rw [div_pow, hs]
  field_simp

/-- Lemma 5: the realized contraction converges almost surely to 2/pi.
The entries have the centered Gaussian law, with one fixed nonzero variance. -/
theorem realizedContraction_gaussian_ae {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} (X : ℕ → Ω → ℝ) {v : ℝ≥0} (hv : v ≠ 0)
    (hindep : Pairwise (fun i j => IndepFun (X i) (X j) μ))
    (hlaw : ∀ i, IdentDistrib (X i) id μ (gaussianReal 0 v)) :
    ∀ᵐ ω ∂μ, Tendsto (fun n => realizedContraction (fun i : Fin n => X i ω))
      atTop (𝓝 (2 / π)) := by
  have habs : Integrable (fun x : ℝ => |x|) (gaussianReal 0 v) := by
    simpa [id, Real.norm_eq_abs] using
      (memLp_id_gaussianReal (μ := 0) (v := v) 2).integrable (by norm_num) |>.norm
  have hsq : Integrable (fun x : ℝ => x ^ 2) (gaussianReal 0 v) := by
    exact (memLp_id_gaussianReal (μ := 0) (v := v) 2).integrable_sq
  have ha := ((hlaw 0).comp (by fun_prop : Measurable (fun x : ℝ => |x|))).integrable_iff.mpr habs
  have hb := ((hlaw 0).comp (by fun_prop : Measurable (fun x : ℝ => x ^ 2))).integrable_iff.mpr hsq
  have heqa : (∫ ω, |X 0 ω| ∂μ) = ∫ x : ℝ, |x| ∂gaussianReal 0 v :=
    ((hlaw 0).comp (by fun_prop : Measurable (fun x : ℝ => |x|))).integral_eq
  have heqb : (∫ ω, X 0 ω ^ 2 ∂μ) = ∫ x : ℝ, x ^ 2 ∂gaussianReal 0 v :=
    ((hlaw 0).comp (by fun_prop : Measurable (fun x : ℝ => x ^ 2))).integral_eq
  have hsecond : (∫ ω, X 0 ω ^ 2 ∂μ) ≠ 0 := by
    rw [heqb, integral_sq_centered_gaussian]
    exact_mod_cast hv
  have h := realizedContraction_iid_ae X ha hb hindep
    (fun i => (hlaw i).trans (hlaw 0).symm) hsecond
  simpa only [heqa, heqb, gaussian_contraction_moment_ratio hv] using h

end EF21Muon.Compression
