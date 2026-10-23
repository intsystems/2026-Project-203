import EF21Muon.Divergence.EF21Field

noncomputable section

namespace EF21Muon.Divergence.EF21SignMuon

open LongTails.Algorithms.MatrixPolar

theorem derivatives_odd (k : ℕ) :
    derivative₁ (position (2*k+3) (0,1)) = 1 ∧
    derivative₂ (position (2*k+3) (1,0)) = 1 := by
  have h₁ := (derivative₁_periodic.nat_mul k) (131/200)
  have h₂ := derivative₂_periodic.sub_nat_mul_eq k (x := 131/200)
  have h₂base : derivative₂ (131/200) = 1 := by
    have h := (derivative₂_periodic (61/200)).trans derivative₂_plus
    norm_num at h
    exact h
  constructor
  · rw [(position_tail k).1, position_three]
    convert h₁.trans derivative₁_plus using 1
    norm_num [drift, ofMatrix, matrixEquiv]
  · rw [(position_tail k).1, position_three]
    apply (congrArg derivative₂ ?_).trans (h₂.trans h₂base)
    norm_num [drift, ofMatrix, matrixEquiv]
    ring

theorem derivatives_even (k : ℕ) :
    derivative₁ (position (2*k+4) (0,1)) = -1 ∧
    derivative₂ (position (2*k+4) (1,0)) = -1 := by
  have h₁ := (derivative₁_periodic.nat_mul k) (7/10)
  have h₂ := derivative₂_periodic.sub_nat_mul_eq k (x := 0)
  constructor
  · rw [(position_tail k).2, position_four]
    convert h₁.trans derivative₁_minus using 1
    norm_num [drift, ofMatrix, matrixEquiv]
  · rw [(position_tail k).2, position_four]
    convert h₂.trans derivative₂_minus using 1
    norm_num [drift, ofMatrix, matrixEquiv]

theorem baseField_gradient_tail (variant : MomentumVariant) {μ : ℝ} (hμ1 : μ < 1)
    (t : ℕ) (ht : 3 ≤ t) :
    gradient (baseField (amplitude variant μ)) (position t) = requiredGradient variant μ t := by
  have hm := Nat.mod_add_div t 2
  have hml := Nat.mod_lt t (by omega : 0 < 2)
  have hcases : (∃ k, t = 2*k+3) ∨ (∃ k, t = 2*k+4) := by
    by_cases he : t%2 = 0
    · exact Or.inr ⟨t/2-2, by omega⟩
    · exact Or.inl ⟨t/2-1, by omega⟩
  rcases hcases with ⟨k,rfl⟩ | ⟨k,rfl⟩
  · rw [baseField_gradient, requiredGradient_odd variant hμ1, baseGradient,
      (derivatives_odd k).1, (derivatives_odd k).2]
    simp [fieldGradient]
  · rw [baseField_gradient, requiredGradient_even variant hμ1, baseGradient,
      (derivatives_even k).1, (derivatives_even k).2]
    simp [fieldGradient]

theorem tail_coord_lower (t : ℕ) (ht : 3 ≤ t) : 131/200 ≤ position t (0,1) := by
  have hm := Nat.mod_add_div t 2
  have hml := Nat.mod_lt t (by omega : 0 < 2)
  have hcases : (∃ k, t = 2*k+3) ∨ (∃ k, t = 2*k+4) := by
    by_cases he : t%2 = 0
    · exact Or.inr ⟨t/2-2, by omega⟩
    · exact Or.inl ⟨t/2-1, by omega⟩
  rcases hcases with ⟨k,rfl⟩ | ⟨k,rfl⟩
  · rw [(position_tail k).1, position_three]
    norm_num [drift, ofMatrix, matrixEquiv]
  · rw [(position_tail k).2, position_four]
    norm_num [drift, ofMatrix, matrixEquiv]
    nlinarith [Nat.cast_nonneg (α := ℝ) k]

end EF21Muon.Divergence.EF21SignMuon
