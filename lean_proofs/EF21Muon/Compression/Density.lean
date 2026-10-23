import EF21Muon.Compression.ScaledSign
import Mathlib.Analysis.SpecificLimits.Basic

/-! A correction to the article: full support alone does not give a
dimension-independent contraction factor. Effective support, measured by
the l1/l2 ratio, is the required quantitative condition. -/

noncomputable section

namespace EF21Muon.Compression

open Filter
open scoped Topology

def denseVector (n : ℕ) : Fin (n + 1) → ℝ :=
  Fin.cases 1 (fun _ => 1 / (n + 1 : ℝ))

theorem denseVector_pos (n : ℕ) (i : Fin (n + 1)) : 0 < denseVector n i := by
  refine Fin.cases ?_ (fun j => ?_) i
  · norm_num [denseVector]
  · simp only [denseVector, Fin.cases_succ]
    positivity

theorem denseVector_norms (n : ℕ) :
    entrywiseL1 (denseVector n) = 1 + n / (n + 1 : ℝ) ∧
    squaredNorm (denseVector n) = 1 + n / (n + 1 : ℝ) ^ 2 := by
  constructor
  · simp [entrywiseL1, Fin.sum_univ_succ, denseVector,
      abs_of_nonneg (by positivity : (0 : ℝ) ≤ n + 1), div_eq_mul_inv]
  · simp [squaredNorm, Fin.sum_univ_succ, denseVector, div_eq_mul_inv]

theorem denseVector_contraction_bound (n : ℕ) :
    0 ≤ realizedContraction (denseVector n) ∧
      realizedContraction (denseVector n) ≤ 4 / (n + 1 : ℝ) := by
  have hd : (0 : ℝ) < n + 1 := by positivity
  have hq : (0 : ℝ) ≤ n / (n + 1 : ℝ) := by positivity
  have hq1 : (n : ℝ) / (n + 1 : ℝ) ≤ 1 := (div_le_one hd).mpr (by linarith)
  have hs : 1 ≤ squaredNorm (denseVector n) := by
    rw [(denseVector_norms n).2]
    have : (0 : ℝ) ≤ n / (n + 1 : ℝ) ^ 2 := by positivity
    linarith
  have hl : entrywiseL1 (denseVector n) ^ 2 ≤ 4 := by
    rw [(denseVector_norms n).1]
    nlinarith
  have hden : 0 < (n + 1 : ℝ) * squaredNorm (denseVector n) := by positivity
  constructor
  · unfold realizedContraction
    positivity
  · unfold realizedContraction
    simp only [Fintype.card_fin, Nat.cast_add, Nat.cast_one]
    apply (div_le_div_iff₀ hden hd).mpr
    nlinarith [mul_le_mul_of_nonneg_left hs hd.le]

/-- Every entry is nonzero, yet the realized contraction tends to zero. -/
theorem denseVector_contraction_tendsto :
    Tendsto (fun n => realizedContraction (denseVector n)) atTop (𝓝 0) := by
  have h : Tendsto (fun n : ℕ => (n : ℝ) + 1) atTop atTop :=
    tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop
  have hz : Tendsto (fun n : ℕ => 4 / ((n : ℝ) + 1)) atTop (𝓝 0) :=
    Tendsto.div_atTop (tendsto_const_nhds (x := (4 : ℝ))) h
  exact squeeze_zero (fun n => (denseVector_contraction_bound n).1)
    (fun n => (denseVector_contraction_bound n).2) hz

/-- A uniform lower bound needs quantitative effective support. -/
theorem effective_support_contraction {ι : Type*} [Fintype ι] [Nonempty ι]
    (x : ι → ℝ) {κ : ℝ} (hx : 0 < squaredNorm x)
    (hspread : κ * (Fintype.card ι : ℝ) * squaredNorm x ≤ entrywiseL1 x ^ 2) :
    κ ≤ realizedContraction x := by
  have hd : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  exact (le_div_iff₀ (mul_pos hd hx)).mpr (by simpa [mul_assoc] using hspread)

end EF21Muon.Compression
