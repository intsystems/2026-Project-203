import EF21Muon.Divergence.Linear

/-! Theorem `th:1`: the article's rational 4 by 4 SignMuon counterexample. -/

noncomputable section

namespace EF21Muon.Divergence.SignMuon

open LongTails.Algorithms.MatrixPolar Matrix

def orthogonal : Matrix (Fin 4) (Fin 4) ℝ :=
  (1 / 103 : ℝ) • !![101, 20, 2, -2; -20, 97, 20, -20;
    -2, 20, 2, 101; -2, 20, -101, -2]

def gradient : Matrix (Fin 4) (Fin 4) ℝ :=
  (1 / 309 : ℝ) • !![100303, 30060, -99994, 99994; -30060, -8709, 30060, -30060;
    99994, 30060, -99994, 100303; 99994, 30060, -100303, 99994]

def principal : Fin 4 → ℝ := ![10, 3, -10, 10]

def positiveFactor : Matrix (Fin 4) (Fin 4) ℝ :=
  1 + (1000 / 309 : ℝ) • vecMulVec principal principal

theorem orthogonal_certificate : orthogonalᵀ * orthogonal = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [orthogonal, Matrix.mul_apply, Fin.sum_univ_succ]

theorem factorization : orthogonal * positiveFactor = gradient := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [orthogonal, positiveFactor, principal, gradient, Matrix.mul_apply,
      vecMulVec, Fin.sum_univ_succ, Matrix.one_apply]

theorem positiveFactor_posDef : positiveFactor.PosDef := by
  apply (Matrix.PosDef.one : (1 : Matrix (Fin 4) (Fin 4) ℝ).PosDef).add_posSemidef
  have h : (vecMulVec principal principal).PosSemidef := by
    simpa only [star_trivial] using Matrix.posSemidef_vecMulVec_self_star principal
  exact h.smul (by norm_num : (0 : ℝ) ≤ 1000 / 309)

theorem polar_gradient : polar (ofMatrix gradient) = ofMatrix orthogonal := by
  rw [← factorization]
  exact polar_of_orthogonal_posDef orthogonal positiveFactor orthogonal_certificate
    positiveFactor_posDef

theorem pairing : inner ℝ (ofMatrix gradient)
    (Placement.signMuon.direction (ofMatrix gradient)) = -42468 / 103 := by
  rw [Placement.direction, polar_gradient]
  norm_num [entrySign, ofMatrix, matrixEquiv, gradient, orthogonal,
    PiLp.inner_apply, Compression.signBit, Fintype.sum_prod_type, Fin.sum_univ_succ]

theorem ascent (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (η : ℕ → ℝ) (hη : ∀ t, 0 < η t) (X₀ : Space 4 4) :
    StrictMono (fun t => inner ℝ (ofMatrix gradient)
      (linearRun .signMuon variant μ η (ofMatrix gradient) X₀ t)) := by
  apply linear_objective_strictAscent .signMuon variant hμ hμ1 η hη
  rw [pairing]
  norm_num

theorem divergence (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (η : ℕ → ℝ) (X₀ : Space 4 4)
    (hη : Filter.Tendsto (fun t => ∑ k ∈ Finset.range t, η k) Filter.atTop Filter.atTop) :
    Filter.Tendsto (fun t => inner ℝ (ofMatrix gradient)
      (linearRun .signMuon variant μ η (ofMatrix gradient) X₀ t)) Filter.atTop Filter.atTop := by
  apply linear_objective_tendsto_atTop .signMuon variant hμ hμ1 η _ X₀ _ hη
  rw [pairing]
  norm_num

end EF21Muon.Divergence.SignMuon
