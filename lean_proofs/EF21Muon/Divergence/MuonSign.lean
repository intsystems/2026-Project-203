import EF21Muon.Divergence.Linear

/-! Theorems `th:2` and `th:3`: a shared 5 by 5 counterexample, with an
exact algebraic polar certificate (all entries lie in `ℚ(√17)`). -/

noncomputable section

namespace EF21Muon.Divergence.MuonSign

open LongTails.Algorithms.MatrixPolar Matrix

def pattern : Matrix (Fin 5) (Fin 5) ℝ :=
  !![-1, -1, 1, 1, 1; -1, -1, 1, -1, -1; 1, -1, 1, 1, -1;
    1, 1, -1, -1, 1; 1, 1, 1, -1, 1]

def gradient : Matrix (Fin 5) (Fin 5) ℝ :=
  !![-1, -1, 1, 1, 1; -1, -1, 1, -1, -1; 1, -1, 1, 1, -1;
    1, 100, -1, -1, 1; 1, 1, 1, -1, 1]

def direction : Matrix (Fin 5) (Fin 5) ℝ :=
  let r := Real.sqrt 17
  let a := 1 / 4 + r / 68
  let b := 3 / 4 - r / 68
  let c := 2 * r / 17
  !![-a, -c, a, a, b; -a, -c, a, -b, -a; b, -c, a, a, -a;
    c, -r / 17, -c, -c, c; a, c, b, -a, a]

def signedDirection : Matrix (Fin 5) (Fin 5) ℝ :=
  !![-1, -1, 1, 1, 1; -1, -1, 1, -1, -1; 1, -1, 1, 1, -1;
    1, -1, -1, -1, 1; 1, 1, 1, -1, 1]

def positiveFactor : Matrix (Fin 5) (Fin 5) ℝ :=
  let r := Real.sqrt 17
  let a := 5 * r / 34 - 1 / 2
  let b := 3 * r / 17
  let c := 7 * r / 17
  !![2+a, b, -a, -a, a; b, c, -b, -b, b; -a, -b, 2+a, a, -a;
    -a, -b, a, 2+a, -a; a, b, -a, -a, 2+a]

theorem sqrt_bounds : 4 < Real.sqrt 17 ∧ Real.sqrt 17 < 9 / 2 := by
  have hs := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 17)
  have hn := Real.sqrt_nonneg (17 : ℝ)
  constructor <;> nlinarith

theorem orthogonal_certificate : directionᵀ * direction = 1 := by
  have hs := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 17)
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [direction, Matrix.mul_apply, Fin.sum_univ_succ] <;> nlinarith only [hs]

theorem factorization : direction * positiveFactor = pattern := by
  have hs := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 17)
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [direction, positiveFactor, pattern, Matrix.mul_apply,
      Fin.sum_univ_succ] <;> nlinarith only [hs]

theorem positiveFactor_posDef : positiveFactor.PosDef := by
  apply Matrix.PosDef.of_dotProduct_mulVec_pos
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [positiveFactor, Matrix.conjTranspose_apply]
  · intro x hx
    let r := Real.sqrt 17
    let a := 5 * r / 34 - 1 / 2
    let c := 3 - 11 * r / 17
    have hr : 0 < r := by dsimp [r]; linarith [sqrt_bounds.1]
    have ha : 0 ≤ a := by dsimp [a, r]; linarith [sqrt_bounds.1]
    have hc : 0 < c := by dsimp [c, r]; linarith [sqrt_bounds.2]
    have hq : star x ⬝ᵥ (positiveFactor *ᵥ x) =
        c * (x 0 ^ 2 + x 2 ^ 2 + x 3 ^ 2 + x 4 ^ 2) + r / 17 * x 1 ^ 2 +
        a * ((x 0 - x 2)^2 + (x 0 - x 3)^2 + (x 0 + x 4)^2 +
          (x 2 + x 3)^2 + (x 4 - x 2)^2 + (x 4 - x 3)^2) +
        (6*r/17) * ((x 0 + x 1/2)^2 + (-x 2 + x 1/2)^2 +
          (-x 3 + x 1/2)^2 + (x 4 + x 1/2)^2) := by
      simp [positiveFactor, Matrix.mulVec, dotProduct, Fin.sum_univ_succ, a, c, r]
      ring
    by_contra! hn
    have hrest₁ : 0 ≤ a * ((x 0 - x 2)^2 + (x 0 - x 3)^2 + (x 0 + x 4)^2 +
        (x 2 + x 3)^2 + (x 4 - x 2)^2 + (x 4 - x 3)^2) := by positivity
    have hrest₂ : 0 ≤ (6*r/17) * ((x 0 + x 1/2)^2 + (-x 2 + x 1/2)^2 +
        (-x 3 + x 1/2)^2 + (x 4 + x 1/2)^2) := by positivity
    have hs : 0 ≤ x 0 ^ 2 + x 2 ^ 2 + x 3 ^ 2 + x 4 ^ 2 := by positivity
    have hz : 0 < r / 17 := by positivity
    have hcs := mul_nonneg hc.le hs
    have hrz := mul_nonneg hz.le (sq_nonneg (x 1))
    have hsle : c * (x 0 ^ 2 + x 2 ^ 2 + x 3 ^ 2 + x 4 ^ 2) ≤ 0 := by linarith
    have hzle : r / 17 * x 1 ^ 2 ≤ 0 := by linarith
    have hs0 := (mul_le_mul_iff_right₀ hc).mp (show
      c * (x 0 ^ 2 + x 2 ^ 2 + x 3 ^ 2 + x 4 ^ 2) ≤ c * 0 by simpa using hsle)
    have hz0 := (mul_le_mul_iff_right₀ hz).mp (show r / 17 * x 1 ^ 2 ≤ r / 17 * 0 by
      simpa using hzle)
    have h0 : x 0 = 0 := by nlinarith only [hs0, sq_nonneg (x 2), sq_nonneg (x 3), sq_nonneg (x 4)]
    have h1 : x 1 = 0 := by nlinarith only [hz0]
    have h2 : x 2 = 0 := by nlinarith only [hs0, sq_nonneg (x 0), sq_nonneg (x 3), sq_nonneg (x 4)]
    have h3 : x 3 = 0 := by nlinarith only [hs0, sq_nonneg (x 0), sq_nonneg (x 2), sq_nonneg (x 4)]
    have h4 : x 4 = 0 := by nlinarith only [hs0, sq_nonneg (x 0), sq_nonneg (x 2), sq_nonneg (x 3)]
    apply hx
    funext i
    fin_cases i
    · exact h0
    · exact h1
    · exact h2
    · exact h3
    · exact h4

theorem polar_pattern : polar (ofMatrix pattern) = ofMatrix direction := by
  rw [← factorization]
  exact polar_of_orthogonal_posDef direction positiveFactor orthogonal_certificate
    positiveFactor_posDef

theorem sign_gradient : entrySign (ofMatrix gradient) = ofMatrix pattern := by
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [entrySign, ofMatrix, matrixEquiv, gradient, pattern, Compression.signBit]

theorem sign_direction : entrySign (ofMatrix direction) = ofMatrix signedDirection := by
  have hr := sqrt_bounds
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [entrySign, ofMatrix, matrixEquiv, direction, signedDirection,
      Compression.signBit] <;> linarith [hr.1, hr.2]

theorem uplink_pairing : inner ℝ (ofMatrix gradient)
    (Placement.muonUSign.direction (ofMatrix gradient)) = 6 - 82 * Real.sqrt 17 / 17 := by
  rw [Placement.direction, sign_gradient, polar_pattern]
  norm_num [ofMatrix, matrixEquiv, gradient, direction, PiLp.inner_apply,
    Fintype.sum_prod_type, Fin.sum_univ_succ]
  ring

theorem uplink_pairing_neg : inner ℝ (ofMatrix gradient)
    (Placement.muonUSign.direction (ofMatrix gradient)) < 0 := by
  rw [uplink_pairing]
  linarith [sqrt_bounds.1]

theorem bidirectional_pairing : inner ℝ (ofMatrix gradient)
    (Placement.muonSign.direction (ofMatrix gradient)) = -76 := by
  rw [Placement.direction, sign_gradient, polar_pattern, sign_direction]
  norm_num [ofMatrix, matrixEquiv, gradient, signedDirection, PiLp.inner_apply,
    Fintype.sum_prod_type, Fin.sum_univ_succ]

theorem ascent (method : Placement) (hmethod : method = .muonUSign ∨ method = .muonSign)
    (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (η : ℕ → ℝ) (hη : ∀ t, 0 < η t) (X₀ : Space 5 5) :
    StrictMono (fun t => inner ℝ (ofMatrix gradient)
      (linearRun method variant μ η (ofMatrix gradient) X₀ t)) := by
  apply linear_objective_strictAscent method variant hμ hμ1 η hη
  rcases hmethod with rfl | rfl
  · exact uplink_pairing_neg
  · rw [bidirectional_pairing]; norm_num

theorem divergence (method : Placement) (hmethod : method = .muonUSign ∨ method = .muonSign)
    (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (η : ℕ → ℝ) (X₀ : Space 5 5)
    (hη : Filter.Tendsto (fun t => ∑ k ∈ Finset.range t, η k) Filter.atTop Filter.atTop) :
    Filter.Tendsto (fun t => inner ℝ (ofMatrix gradient)
      (linearRun method variant μ η (ofMatrix gradient) X₀ t)) Filter.atTop Filter.atTop := by
  apply linear_objective_tendsto_atTop method variant hμ hμ1 η _ X₀ _ hη
  rcases hmethod with rfl | rfl
  · exact uplink_pairing_neg
  · rw [bidirectional_pairing]; norm_num

end EF21Muon.Divergence.MuonSign
