import LongTails.Proofs.MatrixPolarCharacterization
import Mathlib.Analysis.Matrix.PosDef

/-! Certificates for the exact minimum-norm polar oracle used by LongTails. -/

noncomputable section

namespace EF21Muon.Divergence

open LongTails.Algorithms.MatrixPolar
open LongTails.Proofs.MatrixPolarCharacterization
open Matrix InnerProductSpace

def ofMatrix {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) : Space m n :=
  (matrixEquiv m n).symm A

@[simp] theorem ofMatrix_apply {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
    (i : Fin m) (j : Fin n) : ofMatrix A (i, j) = A i j := rfl

theorem operator_ofMatrix {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ)
    (v : EuclideanSpace ℝ (Fin n)) :
    operatorEquiv m n (ofMatrix A) v = WithLp.toLp 2 (A *ᵥ ⇑v) := rfl

theorem orthogonal_inner {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ)
    (hQ : Qᵀ * Q = 1) (v w : EuclideanSpace ℝ (Fin n)) :
    inner ℝ (operatorEquiv n n (ofMatrix Q) v)
      (operatorEquiv n n (ofMatrix Q) w) = inner ℝ v w := by
  simp only [operator_ofMatrix, EuclideanSpace.inner_eq_star_dotProduct,
    star_trivial]
  rw [← Matrix.dotProduct_transpose_mulVec, Matrix.mulVec_mulVec, hQ,
    Matrix.one_mulVec]
  exact dotProduct_comm _ _

theorem polar_of_orthogonal_posDef {n : ℕ} (Q H : Matrix (Fin n) (Fin n) ℝ)
    (hQ : Qᵀ * Q = 1) (hH : H.PosDef) :
    polar (ofMatrix (Q * H)) = ofMatrix Q := by
  let b := hH.isHermitian.eigenvectorBasis
  let e := hH.isHermitian.eigenvalues
  have he (i) : 0 < e i := hH.eigenvalues_pos i
  have hcol (i) : operatorEquiv n n (ofMatrix (Q * H)) (b i) =
      e i • operatorEquiv n n (ofMatrix Q) (b i) := by
    simp only [operator_ofMatrix, ← Matrix.mulVec_mulVec]
    change WithLp.toLp 2 (Q *ᵥ (H *ᵥ ⇑(hH.isHermitian.eigenvectorBasis i))) = _
    rw [hH.isHermitian.mulVec_eigenvectorBasis, Matrix.mulVec_smul, WithLp.toLp_smul]
  have horth (i j) (hij : i ≠ j) :
      inner ℝ (operatorEquiv n n (ofMatrix (Q * H)) (b i))
        (operatorEquiv n n (ofMatrix (Q * H)) (b j)) = 0 := by
    rw [hcol, hcol, real_inner_smul_left, real_inner_smul_right,
      orthogonal_inner Q hQ, b.inner_eq_zero hij, mul_zero, mul_zero]
  have hnorm (i) : ‖operatorEquiv n n (ofMatrix Q) (b i)‖ = 1 := by
    have h := orthogonal_inner Q hQ (b i) (b i)
    simp only [real_inner_self_eq_norm_sq, b.norm_eq_one, one_pow] at h
    nlinarith [norm_nonneg (operatorEquiv n n (ofMatrix Q) (b i))]
  apply (operatorEquiv n n).injective
  apply ContinuousLinearMap.coe_injective
  apply b.toBasis.ext
  intro i
  change operatorEquiv n n (polar (ofMatrix (Q * H))) (b i) =
    operatorEquiv n n (ofMatrix Q) (b i)
  have hp := polar_apply_of_orthogonal_images (ofMatrix (Q * H)) b horth i
  simpa only [hcol, norm_smul, Real.norm_eq_abs, abs_of_pos (he i), hnorm,
    mul_one, smul_smul, inv_mul_cancel₀ (ne_of_gt (he i)), one_smul] using hp

theorem polar_smul_pos {m n : ℕ} {c : ℝ} (hc : 0 < c) (X : Space m n) :
    polar (c • X) = polar X := by
  have hface : supportFace (spectralBall m n) (innerSL ℝ (c • X)) =
      supportFace (spectralBall m n) (innerSL ℝ X) := by
    ext Y
    simp only [supportFace, Set.mem_ofPred_eq, innerSL_apply_apply, real_inner_smul_left,
      mul_le_mul_iff_right₀ hc]
  have hs := minimumNormSupport_spec (spectralBall m n) (spectralBall_compact m n)
    ⟨0, zero_mem_spectralBall m n⟩ (innerSL ℝ (c • X))
  rw [hface] at hs
  exact minimumNormSupport_unique (spectralBall_compact m n)
    ⟨0, zero_mem_spectralBall m n⟩ (spectralBall_convex m n) (innerSL ℝ X) hs.1 hs.2

end EF21Muon.Divergence
