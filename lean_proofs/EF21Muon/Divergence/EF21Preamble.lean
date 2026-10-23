import EF21Muon.Divergence.PolarTwo
import EF21Muon.Divergence.EF21Algorithm

/-! Polar certificates for the momentum-dependent Nesterov preamble in
Lemma `lem:efsm_realize`. The zero-momentum case uses the standard preamble. -/

noncomputable section

namespace EF21Muon.Divergence.EF21SignMuon

open LongTails.Algorithms.MatrixPolar Matrix

def tau (μ : ℝ) : ℝ :=
  (140*μ^2 + 192*μ + 96)*(1+μ)/(117*μ*(1+2*μ))

def nesterovFirst (μ : ℝ) : Space 2 2 :=
  (1+μ)⁻¹ • ofMatrix
    ((!![-4/5,3/5;-3/5,-4/5] : Matrix (Fin 2) (Fin 2) ℝ) *
      !![1,0;0,1+tau μ])

def nesterovSecond (μ : ℝ) : Space 2 2 :=
  (1+μ)⁻¹ • (secondTarget + μ • nesterovFirst μ)

def nesterovThird (μ : ℝ) : Space 2 2 := tailMomentum (-(1+2*μ)⁻¹)

def thirdFactor (μ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let den := 125*(1+μ)^2*(1+2*μ)
  let a := (120+339*μ+253*μ^2-10*μ^3)/den
  let b := (35+177*μ+204*μ^2-55*μ^3)/den
  let d := (43750*μ^4+132805*μ^3+190971*μ^2+188223*μ+65715)/
    (58500*(1+μ)^2*(1+2*μ))
  !![a,b;b,d]

theorem tau_pos {μ : ℝ} (hμ : 0 < μ) : 0 < tau μ := by
  unfold tau
  positivity

theorem polar_nesterovFirst {μ : ℝ} (hμ : 0 < μ) :
    polar (effective .nesterov μ 0 (nesterovFirst μ)) = firstTarget := by
  have hne : 1+μ ≠ 0 := by positivity
  simp only [effective, nesterovFirst, smul_zero, sub_zero, smul_smul,
    mul_inv_cancel₀ hne, one_smul]
  have hQ : (!![-4/5,3/5;-3/5,-4/5] : Matrix (Fin 2) (Fin 2) ℝ)ᵀ *
      !![-4/5,3/5;-3/5,-4/5] = 1 := by
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]
  have hH : (!![1,0;0,1+tau μ] : Matrix (Fin 2) (Fin 2) ℝ).PosDef := by
    apply posDef_two (by norm_num)
    simp only [one_mul, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, sub_zero]
    linarith [tau_pos hμ]
  exact polar_of_orthogonal_posDef _ _ hQ hH

theorem polar_nesterovSecond {μ : ℝ} (hμ : 0 < μ) :
    polar (effective .nesterov μ (nesterovFirst μ) (nesterovSecond μ)) = secondTarget := by
  have hne : 1+μ ≠ 0 := by positivity
  simp only [effective, nesterovSecond, smul_smul, mul_inv_cancel₀ hne, one_smul,
    add_sub_cancel_right, polar_secondTarget]

set_option maxHeartbeats 800000 in
theorem thirdFactor_posDef {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1) : (thirdFactor μ).PosDef := by
  have hp : 0 < 1+μ := by positivity
  have hq : 0 < 1+2*μ := by positivity
  have hpne := hp.ne'
  have hqne := hq.ne'
  have h3 : μ^3 ≤ μ^2 := by nlinarith [mul_le_mul_of_nonneg_left hμ1.le (sq_nonneg μ)]
  have h7 : μ^7 ≤ μ^6 := by
    have hh := mul_le_mul_of_nonneg_left hμ1.le (pow_nonneg hμ 6)
    simpa only [← pow_succ, mul_one] using hh
  unfold thirdFactor
  apply posDef_two
  · apply div_pos _ (by positivity)
    have hrest : 0 ≤ 339*μ + 243*μ^2 := by positivity
    nlinarith only [h3, hrest]
  · have hid :
        (120+339*μ+253*μ^2-10*μ^3)/(125*(1+μ)^2*(1+2*μ)) *
          ((43750*μ^4+132805*μ^3+190971*μ^2+188223*μ+65715)/
            (58500*(1+μ)^2*(1+2*μ))) -
          ((35+177*μ+204*μ^2-55*μ^3)/(125*(1+μ)^2*(1+2*μ)))^2 =
        (11700+62505*μ+131208*μ^2+153030*μ^3+138144*μ^4+
          91237*μ^5+13320*μ^6-700*μ^7)/(11700*(1+μ)^4*(1+2*μ)^2) := by
      field_simp
      ring
    rw [hid]
    apply div_pos _ (by positivity)
    have hrest : 0 ≤ 62505*μ+131208*μ^2+153030*μ^3+138144*μ^4+
        91237*μ^5+12620*μ^6 := by positivity
    nlinarith only [h7, hrest]

set_option maxHeartbeats 800000 in
theorem nesterovThird_factorization {μ : ℝ} (hμ : 0 < μ) :
    effective .nesterov μ (nesterovSecond μ) (nesterovThird μ) =
      ofMatrix ((!![7/25,-24/25;-24/25,-7/25] : Matrix (Fin 2) (Fin 2) ℝ) * thirdFactor μ) := by
  have hm : μ ≠ 0 := hμ.ne'
  have hp : 1+μ ≠ 0 := by positivity
  have hq : 1+2*μ ≠ 0 := by positivity
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    simp [effective, nesterovSecond, nesterovFirst, nesterovThird, tailMomentum,
      secondTarget, thirdFactor, tau, ofMatrix, matrixEquiv]
    <;> field_simp <;> ring

theorem polar_nesterovThird {μ : ℝ} (hμ : 0 < μ) (hμ1 : μ < 1) :
    polar (effective .nesterov μ (nesterovSecond μ) (nesterovThird μ)) = negativeTarget := by
  rw [nesterovThird_factorization hμ]
  apply polar_of_orthogonal_posDef
  · ext i j
    fin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]
  · exact thirdFactor_posDef hμ.le hμ1

end EF21Muon.Divergence.EF21SignMuon
