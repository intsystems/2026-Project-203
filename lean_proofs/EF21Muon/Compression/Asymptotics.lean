import EF21Muon.Compression.ScaledSign
import Mathlib.Probability.StrongLaw

/-! Almost-sure asymptotics of the realized scaled-sign contraction. The
probabilistic claim uses an iid sequence and its first absolute and second
moments, rather than treating "dense" as a distributional assumption. -/

noncomputable section

namespace EF21Muon.Compression

open MeasureTheory ProbabilityTheory Filter
open scoped BigOperators Topology

/-- The normalized-moment expression for a finite sample, including n = 0. -/
theorem realizedContraction_sample (x : ℕ → ℝ) (n : ℕ) :
    realizedContraction (fun i : Fin n => x i) =
      ((∑ i ∈ Finset.range n, |x i|) / (n : ℝ)) ^ 2 /
        ((∑ i ∈ Finset.range n, x i ^ 2) / (n : ℝ)) := by
  unfold realizedContraction entrywiseL1 squaredNorm
  simp only [Fintype.card_fin, ← Fin.sum_univ_eq_sum_range]
  by_cases hn : (n : ℝ) = 0
  · simp [hn]
  · field_simp

theorem realizedContraction_tendsto_of_moments (x : ℕ → ℝ) {a b : ℝ}
    (ha : Tendsto (fun n : ℕ => (∑ i ∈ Finset.range n, |x i|) / (n : ℝ))
      atTop (𝓝 a))
    (hb : Tendsto (fun n : ℕ => (∑ i ∈ Finset.range n, x i ^ 2) / (n : ℝ))
      atTop (𝓝 b)) (hb0 : b ≠ 0) :
    Tendsto (fun n => realizedContraction (fun i : Fin n => x i))
      atTop (𝓝 (a ^ 2 / b)) := by
  convert (ha.pow 2).div hb hb0 using 1
  funext n
  exact realizedContraction_sample x n

/-- The strong law gives alpha_n -> (E|X|)^2 / E[X^2] almost surely.
Pairwise independence suffices; no lower bound on the sample entries is used. -/
theorem realizedContraction_iid_ae {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    (X : ℕ → Ω → ℝ)
    (habs : Integrable (fun ω => |X 0 ω|) μ)
    (hsq : Integrable (fun ω => X 0 ω ^ 2) μ)
    (hindep : Pairwise (fun i j => IndepFun (X i) (X j) μ))
    (hident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (hsecond : (∫ ω, X 0 ω ^ 2 ∂μ) ≠ 0) :
    ∀ᵐ ω ∂μ, Tendsto (fun n => realizedContraction (fun i : Fin n => X i ω))
      atTop (𝓝 ((∫ ω, |X 0 ω| ∂μ) ^ 2 / (∫ ω, X 0 ω ^ 2 ∂μ))) := by
  have hmabs : Measurable (fun x : ℝ => |x|) := by
    convert (measurable_norm : Measurable (fun x : ℝ => ‖x‖)) using 1
  have hmsq : Measurable (fun x : ℝ => x ^ 2) := measurable_id.pow_const 2
  have haind : Pairwise (fun i j =>
      IndepFun (fun ω => |X i ω|) (fun ω => |X j ω|) μ) := by
    intro i j hij
    exact (hindep hij).comp hmabs hmabs
  have hbInd : Pairwise (fun i j =>
      IndepFun (fun ω => X i ω ^ 2) (fun ω => X j ω ^ 2) μ) := by
    intro i j hij
    exact (hindep hij).comp hmsq hmsq
  have haid (i : ℕ) : IdentDistrib (fun ω => |X i ω|) (fun ω => |X 0 ω|) μ μ :=
    (hident i).comp hmabs
  have hbid (i : ℕ) : IdentDistrib (fun ω => X i ω ^ 2) (fun ω => X 0 ω ^ 2) μ μ :=
    (hident i).comp hmsq
  have ha := strong_law_ae_real (μ := μ) (fun i ω => |X i ω|) habs haind haid
  have hb := strong_law_ae_real (μ := μ) (fun i ω => X i ω ^ 2) hsq hbInd hbid
  filter_upwards [ha, hb] with ω haω hbω
  exact realizedContraction_tendsto_of_moments (fun i => X i ω) haω hbω hsecond

end EF21Muon.Compression
