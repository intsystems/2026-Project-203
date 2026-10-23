import EF21Muon.Stochastic.SquaredNorm
import LongTails.Proofs.MatrixPolarCharacterization

noncomputable section

namespace EF21Muon.Stochastic

open scoped BigOperators
open LongTails.Algorithms.MatrixPolar
open LongTails.Proofs.MatrixPolarCharacterization

variable {m n : ℕ}

def transpose (X : Space m n) : Space n m := WithLp.toLp 2 (fun ij => X (ij.2, ij.1))

theorem transpose_operator (X : Space m n) :
    operatorEquiv n m (transpose X) = (operatorEquiv m n X).adjoint := by
  apply (ContinuousLinearMap.eq_adjoint_iff _ _).mpr
  intro u v
  simp only [PiLp.inner_apply, RCLike.inner_apply, conj_trivial,
    operatorEquiv_apply_coordinate]
  dsimp only [transpose]
  simp only [Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem transpose_norm (X : Space m n) : ‖transpose X‖ = ‖X‖ := by
  apply (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  simp only [Fintype.sum_prod_type]
  change (∑ i : Fin n, ∑ j : Fin m, X (j, i) ^ 2) = ∑ i : Fin m, ∑ j : Fin n, X (i, j) ^ 2
  exact Finset.sum_comm

theorem transpose_spectralNorm (X : Space m n) : spectralNorm (transpose X) = spectralNorm X := by
  unfold spectralNorm
  rw [transpose_operator]
  exact ContinuousLinearMap.adjoint.norm_map _

theorem frobenius_le_sqrt_columns (X : Space m n) :
    ‖X‖ ≤ Real.sqrt n * spectralNorm X := by
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg (Real.sqrt_nonneg _) (norm_nonneg _))).mp
  rw [mul_pow, Real.sq_sqrt (Nat.cast_nonneg n),
    norm_sq_eq_sum_columns X (EuclideanSpace.basisFun (Fin n) ℝ)]
  calc
    _ ≤ ∑ _j : Fin n, spectralNorm X ^ 2 := by
      apply Finset.sum_le_sum
      intro j _
      have h := (operatorEquiv m n X).le_opNorm (EuclideanSpace.basisFun (Fin n) ℝ j)
      simp only [OrthonormalBasis.norm_eq_one, mul_one] at h
      exact pow_le_pow_left₀ (norm_nonneg _) h 2
    _ = _ := by simp [spectralNorm]

/-- The paper's matrix norm equivalence, with r = min(m,n). -/
theorem frobenius_le_sqrt_rank (X : Space m n) :
    ‖X‖ ≤ Real.sqrt ((min m n : ℕ) : ℝ) * spectralNorm X := by
  rcases le_total m n with h | h
  · rw [Nat.min_eq_left h]
    simpa only [transpose_norm, transpose_spectralNorm] using frobenius_le_sqrt_columns (transpose X)
  · rw [Nat.min_eq_right h]
    exact frobenius_le_sqrt_columns X

theorem nuclearNorm_le_sqrt_rank (X : Space m n) :
    nuclearNorm X ≤ Real.sqrt ((min m n : ℕ) : ℝ) * ‖X‖ := by
  apply ContinuousLinearMap.opNorm_le_bound _ (mul_nonneg (Real.sqrt_nonneg _) (norm_nonneg _))
  intro U
  have hi := norm_inner_le_norm (𝕜 := ℝ) X ((operatorEquiv m n).symm U)
  have hn := mul_le_mul_of_nonneg_left (frobenius_le_sqrt_rank ((operatorEquiv m n).symm U))
    (norm_nonneg X)
  have he : spectralNorm ((operatorEquiv m n).symm U) = ‖U‖ := by simp [spectralNorm]
  rw [he] at hn
  change ‖inner ℝ X ((operatorEquiv m n).symm U)‖ ≤ _
  nlinarith only [hi, hn]

theorem polar_frobenius_le_sqrt_rank (X : Space m n) :
    ‖polar X‖ ≤ Real.sqrt ((min m n : ℕ) : ℝ) := by
  exact (frobenius_le_sqrt_rank (polar X)).trans
    (by simpa only [mul_one] using
      mul_le_mul_of_nonneg_left (polar_mem X) (Real.sqrt_nonneg ((min m n : ℕ) : ℝ)))

end EF21Muon.Stochastic
