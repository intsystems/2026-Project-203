import EF21Muon.Algorithm
import Mathlib.Analysis.InnerProductSpace.Basic

/-! Exact expectations for independent sampling from a finite distribution.
The weights need not be uniform. No bounded-gradient assumption is made. -/

noncomputable section

namespace EF21Muon.Stochastic

open scoped BigOperators

universe u

structure Sampling (Ω : Type*) [Fintype Ω] where
  mass : Ω → ℝ
  nonneg : ∀ ω, 0 ≤ mass ω
  total : ∑ ω, mass ω = 1

variable {Ω : Type*} [Fintype Ω]
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def Sampling.expect (p : Sampling Ω) (X : Ω → E) : E := ∑ ω, p.mass ω • X ω

@[simp] theorem Sampling.expect_const (p : Sampling Ω) (a : E) :
    p.expect (fun _ => a) = a := by
  simp [Sampling.expect, ← Finset.sum_smul, p.total]

@[simp] theorem Sampling.expect_add (p : Sampling Ω) (X Y : Ω → E) :
    p.expect (fun ω => X ω + Y ω) = p.expect X + p.expect Y := by
  simp [Sampling.expect, smul_add, Finset.sum_add_distrib]

@[simp] theorem Sampling.expect_sub (p : Sampling Ω) (X Y : Ω → E) :
    p.expect (fun ω => X ω - Y ω) = p.expect X - p.expect Y := by
  simp [Sampling.expect, smul_sub, Finset.sum_sub_distrib]

@[simp] theorem Sampling.expect_smul (p : Sampling Ω) (a : ℝ) (X : Ω → E) :
    p.expect (fun ω => a • X ω) = a • p.expect X := by
  unfold Sampling.expect
  rw [Finset.smul_sum]
  exact Finset.sum_congr rfl (fun ω _ => smul_comm (p.mass ω) a (X ω))

@[simp] theorem Sampling.expect_mul (p : Sampling Ω) (a : ℝ) (X : Ω → ℝ) :
    p.expect (fun ω => a * X ω) = a * p.expect X := expect_smul p a X

theorem Sampling.expect_mono (p : Sampling Ω) {X Y : Ω → ℝ}
    (h : ∀ ω, X ω ≤ Y ω) : p.expect X ≤ p.expect Y := by
  exact Finset.sum_le_sum fun ω _ => mul_le_mul_of_nonneg_left (h ω) (p.nonneg ω)

theorem Sampling.expect_nonneg (p : Sampling Ω) {X : Ω → ℝ}
    (h : ∀ ω, 0 ≤ X ω) : 0 ≤ p.expect X := by
  simpa using p.expect_mono h

theorem Sampling.expect_inner (p : Sampling Ω) (a : E) (X : Ω → E) :
    p.expect (fun ω => inner ℝ a (X ω)) = inner ℝ a (p.expect X) := by
  simp [Sampling.expect, inner_sum, inner_smul_right]

/-- Unbiased noise cancels the cross term before any Young inequality is used. -/
theorem Sampling.variance_identity (p : Sampling Ω) (a : E) (β : ℝ)
    (noise : Ω → E) (hzero : p.expect noise = 0) :
    p.expect (fun ω => ‖a + β • noise ω‖ ^ 2) =
      ‖a‖ ^ 2 + β ^ 2 * p.expect (fun ω => ‖noise ω‖ ^ 2) := by
  have hpoint (ω : Ω) : ‖a + β • noise ω‖ ^ 2 =
      ‖a‖ ^ 2 + 2 * β * inner ℝ a (noise ω) + β ^ 2 * ‖noise ω‖ ^ 2 := by
    rw [norm_add_sq_real, inner_smul_right, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
    ring
  simp only [hpoint, expect_add, expect_const, expect_mul, expect_inner, hzero,
    inner_zero_right, mul_zero, add_zero]

/-- A finite stochastic oracle, unbiased at every possible adaptive query. -/
structure Oracle (p : Sampling Ω) (grad : E → E) (σ : ℝ) where
  reply : E → Ω → E
  unbiased : ∀ x, p.expect (reply x) = grad x
  variance : ∀ x, p.expect (fun ω => ‖reply x ω - grad x‖ ^ 2) ≤ σ ^ 2

theorem Oracle.affine_variance {p : Sampling Ω} {grad : E → E} {σ : ℝ}
    (O : Oracle p grad σ) (x a : E) (β : ℝ) :
    p.expect (fun ω => ‖a + β • (O.reply x ω - grad x)‖ ^ 2) ≤
      ‖a‖ ^ 2 + β ^ 2 * σ ^ 2 := by
  rw [p.variance_identity a β _ (by simp [O.unbiased])]
  exact add_le_add le_rfl (mul_le_mul_of_nonneg_left (O.variance x) (sq_nonneg β))

/-- Exact expected sum of costs over `T` fresh, independent samples.
This definition evaluates the actual update at every branch of the sampling tree. -/
def expectedTotal {S : Type*} (p : Sampling Ω) (next : S → Ω → S)
    (cost : S → ℝ) : ℕ → S → ℝ
  | 0, _ => 0
  | T + 1, s => cost s + p.expect (fun ω => expectedTotal p next cost T (next s ω))

theorem expectedTotal_mul {S : Type*} (p : Sampling Ω) (next : S → Ω → S)
    (cost : S → ℝ) (a : ℝ) (T : ℕ) (s : S) :
    expectedTotal p next (fun s => a * cost s) T s = a * expectedTotal p next cost T s := by
  induction T generalizing s with
  | zero => simp [expectedTotal]
  | succ T ih => simp only [expectedTotal, ih, Sampling.expect_mul, mul_add]

theorem expectedTotal_le {S : Type*} (p : Sampling Ω) (next : S → Ω → S)
    (cost V : S → ℝ) (b : ℝ)
    (hV : ∀ s, 0 ≤ V s)
    (hdrift : ∀ s, p.expect (fun ω => V (next s ω)) + cost s ≤ V s + b)
    (T : ℕ) (s : S) : expectedTotal p next cost T s ≤ V s + (T : ℝ) * b := by
  induction T generalizing s with
  | zero => simpa [expectedTotal] using hV s
  | succ T ih =>
    have hi := p.expect_mono (fun ω => ih (next s ω))
    simp only [p.expect_add, p.expect_const] at hi
    dsimp only [expectedTotal]
    push_cast
    linarith [hdrift s]

def Sampling.prod {Ξ : Type*} [Fintype Ξ] (p : Sampling Ω) (q : Sampling Ξ) :
    Sampling (Ω × Ξ) where
  mass z := p.mass z.1 * q.mass z.2
  nonneg z := mul_nonneg (p.nonneg z.1) (q.nonneg z.2)
  total := by simp [Fintype.sum_prod_type, ← Finset.mul_sum, p.total, q.total]

theorem Sampling.expect_prod {Ξ : Type*} [Fintype Ξ] (p : Sampling Ω) (q : Sampling Ξ)
    (X : Ω × Ξ → ℝ) :
    (p.prod q).expect X = p.expect (fun ω => q.expect (fun ξ => X (ω, ξ))) := by
  simp only [expect, prod, smul_eq_mul, Fintype.sum_prod_type, Finset.mul_sum, mul_assoc]

/-- One fresh sample per update. Product weights encode independence exactly. -/
def Samples (Ω : Type u) : ℕ → Type u
  | 0 => PUnit
  | T + 1 => Ω × Samples Ω T

instance samplesFintype : (T : ℕ) → Fintype (Samples Ω T)
  | 0 => inferInstanceAs (Fintype PUnit)
  | T + 1 => @instFintypeProd Ω (Samples Ω T) _ (samplesFintype T)

def Sampling.paths (p : Sampling Ω) : (T : ℕ) → Sampling (Samples Ω T)
  | 0 => ⟨fun _ => 1, fun _ => by norm_num, by
      change (∑ _ : PUnit, (1 : ℝ)) = 1
      norm_num⟩
  | T + 1 => p.prod (p.paths T)

def run {S : Type*} (next : S → Ω → S) : (T : ℕ) → S → Samples Ω T → List S
  | 0, _, _ => []
  | T + 1, s, ξ => s :: run next T (next s ξ.1) ξ.2

/-- The recursive expectation is exactly the expected sum along actual sample paths. -/
theorem expectedTotal_eq_paths {S : Type*} (p : Sampling Ω) (next : S → Ω → S)
    (cost : S → ℝ) (T : ℕ) (s : S) :
    expectedTotal p next cost T s =
      (p.paths T).expect (fun ξ => ((run next T s ξ).map cost).sum) := by
  induction T generalizing s with
  | zero => simp [expectedTotal, run]
  | succ T ih =>
    change cost s + p.expect (fun ω => expectedTotal p next cost T (next s ω)) =
      (p.prod (p.paths T)).expect (fun ξ =>
        ((s :: run next T (next s ξ.1) ξ.2).map cost).sum)
    rw [Sampling.expect_prod]
    simp only [List.map_cons, List.sum_cons, Sampling.expect_add, Sampling.expect_const, ih]

end EF21Muon.Stochastic
