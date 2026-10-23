import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

open scoped BigOperators

noncomputable section

namespace EF21Muon.Compression

variable {ι : Type*} [Fintype ι]

def squaredNorm (x : ι → ℝ) : ℝ := ∑ i, x i ^ 2

def entrywiseL1 (x : ι → ℝ) : ℝ := ∑ i, |x i|

def CompatibleSigns (x s : ι → ℝ) : Prop :=
  (∀ i, s i ^ 2 = 1) ∧ ∀ i, x i * s i = |x i|

def signBit (a : ℝ) : ℝ := if 0 ≤ a then 1 else -1

omit [Fintype ι] in
theorem compatible_signBit (x : ι → ℝ) : CompatibleSigns x (fun i => signBit (x i)) := by
  constructor
  · intro i
    by_cases h : 0 ≤ x i <;> simp [signBit, h]
  · intro i
    by_cases h : 0 ≤ x i
    · simp [signBit, h, abs_of_nonneg h]
    · simp [signBit, h, abs_of_neg (lt_of_not_ge h)]

def scaledSignWith (x s : ι → ℝ) : ι → ℝ :=
  fun i => (entrywiseL1 x / (Fintype.card ι : ℝ)) * s i

def scaledSign (x : ι → ℝ) : ι → ℝ := scaledSignWith x (fun i => signBit (x i))

theorem squared_error_expansion (x s : ι → ℝ) (hs : CompatibleSigns x s) (c : ℝ) :
    (∑ i, (c * s i - x i) ^ 2) =
      squaredNorm x - 2 * c * entrywiseL1 x + (Fintype.card ι : ℝ) * c ^ 2 := by
  classical
  have hpoint (i : ι) : (c * s i - x i) ^ 2 = x i ^ 2 - 2 * c * |x i| + c ^ 2 := by
    calc
      _ = x i ^ 2 - 2 * c * (x i * s i) + c ^ 2 * s i ^ 2 := by ring
      _ = _ := by rw [hs.1 i, hs.2 i]; ring
  simp only [hpoint, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
    squaredNorm, entrywiseL1]

theorem scaledSign_error_identity [Nonempty ι] (x s : ι → ℝ)
    (hs : CompatibleSigns x s) :
    squaredNorm (fun i => scaledSignWith x s i - x i) =
      squaredNorm x - entrywiseL1 x ^ 2 / (Fintype.card ι : ℝ) := by
  have hd : (Fintype.card ι : ℝ) ≠ 0 := by
    exact_mod_cast (Fintype.card_ne_zero : Fintype.card ι ≠ 0)
  change (∑ i, ((entrywiseL1 x / (Fintype.card ι : ℝ)) * s i - x i) ^ 2) = _
  rw [squared_error_expansion x s hs]
  field_simp
  ring

theorem squaredNorm_le_entrywiseL1_sq (x : ι → ℝ) :
    squaredNorm x ≤ entrywiseL1 x ^ 2 := by
  classical
  have hle (i : ι) : |x i| ≤ ∑ j, |x j| :=
    Finset.single_le_sum (fun j _ => abs_nonneg (x j)) (Finset.mem_univ i)
  calc
    squaredNorm x = ∑ i, |x i| * |x i| := by
      unfold squaredNorm
      apply Finset.sum_congr rfl
      intro i _
      rw [← pow_two, sq_abs]
    _ ≤ ∑ i, (∑ j, |x j|) * |x i| := by
      apply Finset.sum_le_sum
      intro i _
      exact mul_le_mul_of_nonneg_right (hle i) (abs_nonneg (x i))
    _ = entrywiseL1 x ^ 2 := by
      rw [← Finset.mul_sum]
      unfold entrywiseL1
      ring

theorem scaledSign_contracts [Nonempty ι] (x s : ι → ℝ)
    (hs : CompatibleSigns x s) :
    squaredNorm (fun i => scaledSignWith x s i - x i) ≤
      (1 - 1 / (Fintype.card ι : ℝ)) * squaredNorm x := by
  have hd : 0 < (Fintype.card ι : ℝ) := by
    exact_mod_cast (Fintype.card_pos : 0 < Fintype.card ι)
  have hquot := div_le_div_of_nonneg_right (squaredNorm_le_entrywiseL1_sq x) hd.le
  calc
    _ = squaredNorm x - entrywiseL1 x ^ 2 / (Fintype.card ι : ℝ) :=
      scaledSign_error_identity x s hs
    _ ≤ squaredNorm x - squaredNorm x / (Fintype.card ι : ℝ) :=
      sub_le_sub_left hquot _
    _ = _ := by ring

theorem scaledSign_contractivity [Nonempty ι] (x : ι → ℝ) :
    squaredNorm (fun i => scaledSign x i - x i) ≤
      (1 - 1 / (Fintype.card ι : ℝ)) * squaredNorm x :=
  scaledSign_contracts x _ (compatible_signBit x)

theorem scaledSign_euclidean_contracts [Nonempty ι] (x : EuclideanSpace ℝ ι)
    (s : ι → ℝ) (hs : CompatibleSigns (fun i => x i) s) :
    ‖WithLp.toLp 2 (scaledSignWith (fun i => x i) s) - x‖ ^ 2 ≤
      (1 - 1 / (Fintype.card ι : ℝ)) * ‖x‖ ^ 2 := by
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  change squaredNorm (fun i => scaledSignWith (fun j => x j) s i - x i) ≤
    (1 - 1 / (Fintype.card ι : ℝ)) * squaredNorm (fun i => x i)
  exact scaledSign_contracts (fun i => x i) s hs

/-- The realized contraction factor in Lemma 5, for a nonzero input. -/
def realizedContraction (x : ι → ℝ) : ℝ :=
  entrywiseL1 x ^ 2 / ((Fintype.card ι : ℝ) * squaredNorm x)

theorem scaledSign_realized_identity [Nonempty ι] (x s : ι → ℝ)
    (hs : CompatibleSigns x s) (hx : squaredNorm x ≠ 0) :
    squaredNorm (fun i => scaledSignWith x s i - x i) =
      (1 - realizedContraction x) * squaredNorm x := by
  rw [scaledSign_error_identity x s hs]
  unfold realizedContraction
  field_simp

theorem one_sparse_norm_identity (x : ι → ℝ) (i : ι)
    (hx : ∀ j, j ≠ i → x j = 0) :
    squaredNorm x = x i ^ 2 ∧ entrywiseL1 x = |x i| := by
  classical
  constructor
  · unfold squaredNorm
    exact Finset.sum_eq_single i (fun j _ hji => by simp [hx j hji]) (by simp)
  · unfold entrywiseL1
    exact Finset.sum_eq_single i (fun j _ hji => by simp [hx j hji]) (by simp)

/-- The worst-case contraction 1/d is attained by every nonzero one-sparse input. -/
theorem one_sparse_realizedContraction (x : ι → ℝ) (i : ι)
    (hx : ∀ j, j ≠ i → x j = 0) (hi : x i ≠ 0) :
    realizedContraction x = 1 / (Fintype.card ι : ℝ) := by
  rw [realizedContraction, (one_sparse_norm_identity x i hx).1,
    (one_sparse_norm_identity x i hx).2, sq_abs]
  field_simp

/-- Equal-magnitude dense inputs are reconstructed exactly, independently of
the compatible sign choice at zero. Density alone does not imply this property. -/
theorem constant_magnitude_exact [Nonempty ι] (x s : ι → ℝ)
    (hs : CompatibleSigns x s) {a : ℝ} (ha : ∀ i, |x i| = a) :
    scaledSignWith x s = x := by
  have hd : (Fintype.card ι : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hc : entrywiseL1 x / (Fintype.card ι : ℝ) = a := by
    simp [entrywiseL1, ha, hd]
  funext i
  change (entrywiseL1 x / (Fintype.card ι : ℝ)) * s i = x i
  rw [hc, ← ha i, ← hs.2 i]
  rw [mul_assoc, ← pow_two, hs.1 i, mul_one]

end EF21Muon.Compression
