import Mathlib.Probability.Distributions.Bernoulli
import Mathlib.Analysis.SpecificLimits.Basic

/-! A correction to the rate comparison after Corollary 2: convergence of
the first moment does not imply convergence of the second moment. -/

noncomputable section

namespace EF21Muon.Probability

open MeasureTheory ProbabilityTheory Filter
open scoped Topology

def rareProbability (a : ℝ) (ha : 1 ≤ a) : unitInterval :=
  ⟨1 / a ^ 2, by
    have hp : 0 < a := by linarith
    constructor
    · positivity
    · apply (div_le_one (sq_pos_of_pos hp)).mpr
      nlinarith⟩

/-- Value a with probability a^(-2), zero otherwise. -/
def rareLaw (a : ℝ) (ha : 1 ≤ a) : Measure ℝ :=
  bernoulliMeasure a 0 (rareProbability a ha)

instance rareLaw_probability (a : ℝ) (ha : 1 ≤ a) : IsProbabilityMeasure (rareLaw a ha) := by
  unfold rareLaw
  infer_instance

theorem rareLaw_moments (a : ℝ) (ha : 1 ≤ a) :
    (∫ z, |z| ∂rareLaw a ha) = 1 / a ∧
      (∫ z, z ^ 2 ∂rareLaw a ha) = 1 := by
  have hp : 0 < a := by linarith
  constructor <;>
    rw [rareLaw, integral_bernoulliMeasure] <;>
    simp only [rareProbability, smul_eq_mul, abs_zero, zero_pow (by norm_num : (2 : ℕ) ≠ 0),
      mul_zero, add_zero, abs_of_pos hp] <;> field_simp

theorem rareLaw_first_moment_tendsto :
    Tendsto (fun n : ℕ => ∫ z, |z| ∂rareLaw ((n : ℝ) + 1)
      (by have := Nat.cast_nonneg (α := ℝ) n; linarith))
      atTop (𝓝 0) := by
  simp only [(rareLaw_moments _ _).1]
  exact tendsto_const_nhds.div_atTop
    (tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop)

theorem rareLaw_second_moment_not_tendsto :
    ¬ Tendsto (fun n : ℕ => ∫ z, z ^ 2 ∂rareLaw ((n : ℝ) + 1)
      (by have := Nat.cast_nonneg (α := ℝ) n; linarith))
      atTop (𝓝 0) := by
  simp only [(rareLaw_moments _ _).2]
  intro h
  have he : (0 : ℝ) = 1 := tendsto_nhds_unique h tendsto_const_nhds
  norm_num at he

end EF21Muon.Probability
