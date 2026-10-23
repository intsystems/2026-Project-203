import EF21Muon.Divergence.EF21Cycle

noncomputable section

namespace EF21Muon.Divergence

open LongTails.Algorithms.MatrixPolar
open LongTails.Proofs.MatrixPolarCharacterization Matrix

theorem posDef_two {a b d : ℝ} (ha : 0 < a) (hd : 0 < a*d-b^2) :
    (!![a,b;b,d] : Matrix (Fin 2) (Fin 2) ℝ).PosDef := by
  apply Matrix.PosDef.of_dotProduct_mulVec_pos
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [Matrix.conjTranspose_apply]
  · intro x hx
    have hq : a * (star x ⬝ᵥ (!![a,b;b,d] *ᵥ x)) =
        (a*x 0 + b*x 1)^2 + (a*d-b^2)*x 1^2 := by
      simp [dotProduct, Matrix.mulVec, Fin.sum_univ_succ]
      ring
    by_contra! hn
    have hmul := mul_nonpos_of_nonneg_of_nonpos ha.le hn
    have hterm := mul_nonneg hd.le (sq_nonneg (x 1))
    have h1sq : (a*d-b^2)*x 1^2 ≤ 0 := by nlinarith only [hq, hmul, sq_nonneg (a*x 0+b*x 1)]
    have h1 : x 1 = 0 := by
      have h := (mul_le_mul_iff_right₀ hd).mp
        (show (a*d-b^2)*x 1^2 ≤ (a*d-b^2)*0 by simpa using h1sq)
      nlinarith only [h]
    rw [h1] at hq
    have h0 : a*x 0 = 0 := by nlinarith only [hq, hmul]
    have hx0 : x 0 = 0 := (mul_eq_zero.mp h0).resolve_left ha.ne'
    apply hx
    ext i
    fin_cases i
    · exact hx0
    · exact h1

namespace EF21SignMuon

theorem polar_firstTarget : polar firstTarget = firstTarget := by
  have hQ : (!![-4/5,3/5;-3/5,-4/5] : Matrix (Fin 2) (Fin 2) ℝ)ᵀ *
      !![-4/5,3/5;-3/5,-4/5] = 1 := by
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]
  simpa [Matrix.mul_one, firstTarget] using polar_of_orthogonal_posDef
    !![-4/5,3/5;-3/5,-4/5] 1 hQ Matrix.PosDef.one

theorem polar_secondTarget : polar secondTarget = secondTarget := by
  let v : Fin 2 → EuclideanSpace ℝ (Fin 2) :=
    ![WithLp.toLp 2 ![3/5,-4/5], WithLp.toLp 2 ![4/5,3/5]]
  have hv : Orthonormal ℝ v := by
    rw [orthonormal_iff_ite]
    intro i j
    simp only [PiLp.inner_apply]
    fin_cases i <;> fin_cases j <;> norm_num [v, PiLp.inner_apply, Fin.sum_univ_succ]
  let B := basisOfOrthonormalOfCardEqFinrank hv (by simp)
  let b := B.toOrthonormalBasis (by simpa [B] using hv)
  have hb (i) : b i = v i := by simp [b, B]
  have hcol₀ : operatorEquiv 2 2 secondTarget (b 0) = EuclideanSpace.single 0 (1 : ℝ) := by
    rw [hb]
    ext i
    fin_cases i <;> norm_num [secondTarget, operator_ofMatrix, v, Matrix.mulVec,
      dotProduct, Fin.sum_univ_succ, EuclideanSpace.single]
  have hcol₁ : operatorEquiv 2 2 secondTarget (b 1) = 0 := by
    rw [hb]
    ext i
    fin_cases i <;> norm_num [secondTarget, operator_ofMatrix, v, Matrix.mulVec,
      dotProduct, Fin.sum_univ_succ]
  have horth (i j : Fin 2) (hij : i ≠ j) :
      inner ℝ (operatorEquiv 2 2 secondTarget (b i))
        (operatorEquiv 2 2 secondTarget (b j)) = 0 := by
    fin_cases i <;> fin_cases j <;> simp_all
  apply (operatorEquiv 2 2).injective
  apply ContinuousLinearMap.coe_injective
  apply b.toBasis.ext
  intro i
  change operatorEquiv 2 2 (polar secondTarget) (b i) = operatorEquiv 2 2 secondTarget (b i)
  rw [polar_apply_of_orthogonal_images secondTarget b horth]
  fin_cases i <;> simp [hcol₀, hcol₁]

def tailMomentum (v : ℝ) : Space 2 2 := ofMatrix !![0,v;v,-7/12]

theorem polar_tailMomentum (sign : ℝ) (hs : sign = 1 ∨ sign = -1) :
    polar (tailMomentum sign) = ofMatrix !![7/25, sign*24/25; sign*24/25,-7/25] := by
  let Q : Matrix (Fin 2) (Fin 2) ℝ := !![7/25,sign*24/25;sign*24/25,-7/25]
  let H : Matrix (Fin 2) (Fin 2) ℝ := !![24/25,-sign*7/25;-sign*7/25,337/300]
  have hQ : Qᵀ*Q = 1 := by
    rcases hs with rfl | rfl <;> ext i j <;> fin_cases i <;> fin_cases j <;>
      norm_num [Q, Matrix.mul_apply, Fin.sum_univ_succ]
  have hH : H.PosDef := by
    apply posDef_two (by norm_num)
    rcases hs with rfl | rfl <;> norm_num
  have hfac : Q*H = !![0,sign;sign,-7/12] := by
    rcases hs with rfl | rfl <;> ext i j <;> fin_cases i <;> fin_cases j <;>
      norm_num [Q, H, Matrix.mul_apply, Fin.sum_univ_succ]
  change polar (ofMatrix _) = ofMatrix Q
  rw [← hfac]
  exact polar_of_orthogonal_posDef Q H hQ hH

end EF21SignMuon
end EF21Muon.Divergence
