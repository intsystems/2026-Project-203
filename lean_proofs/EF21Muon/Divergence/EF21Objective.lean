import EF21Muon.Divergence.EF21Cycle
import EF21Muon.Divergence.Sequences

/-! The objective-value part of Theorem `th:ef_div`. These results explicitly
state their periodic-field and realization hypotheses. -/

noncomputable section

namespace EF21Muon.Divergence.EF21SignMuon

open LongTails.Algorithms.MatrixPolar Filter

def field (A : ℝ) (Φ₁ Φ₂ : ℝ → ℝ) (X : Space 2 2) : ℝ :=
  -(7/12 : ℝ) * X (1,1) + A * Φ₁ (X (0,1)) + A * Φ₂ (X (1,0))

theorem field_period (A : ℝ) (Φ₁ Φ₂ : ℝ → ℝ)
    (h₁ : Function.Periodic Φ₁ (21/20)) (h₂ : Function.Periodic Φ₂ (7/20))
    (t : ℕ) (ht : 3 ≤ t) :
    field A Φ₁ Φ₂ (position (t+2)) - field A Φ₁ Φ₂ (position t) = 49/240 := by
  have hs := position_period t ht
  have h₁' := h₁ (position t (0,1))
  have h₂' := h₂.sub_eq (position t (1,0))
  simp only [field, hs, PiLp.add_apply]
  norm_num [drift, ofMatrix, matrixEquiv] at *
  simp only [sub_eq_add_neg] at h₂'
  rw [h₁', h₂']
  ring

theorem objective_period {f : Space 2 2 → ℝ} (A : ℝ) (Φ₁ Φ₂ : ℝ → ℝ)
    (h₁ : Function.Periodic Φ₁ (21/20)) (h₂ : Function.Periodic Φ₂ (7/20))
    (hf : ∀ t, 3 ≤ t → f (position t) = field A Φ₁ Φ₂ (position t))
    (t : ℕ) (ht : 3 ≤ t) : f (position (t+2)) - f (position t) = 49/240 := by
  rw [hf (t+2) (by omega), hf t ht]
  exact field_period A Φ₁ Φ₂ h₁ h₂ t ht

theorem objective_divergence {f : Space 2 2 → ℝ} (A : ℝ) (Φ₁ Φ₂ : ℝ → ℝ)
    (h₁ : Function.Periodic Φ₁ (21/20)) (h₂ : Function.Periodic Φ₂ (7/20))
    (hf : ∀ t, 3 ≤ t → f (position t) = field A Φ₁ Φ₂ (position t)) :
    Tendsto (fun t => f (position t)) atTop atTop := by
  apply tendsto_atTop_of_twoStep (c := 49/240) (by norm_num) 3
  intro t ht
  have h := objective_period A Φ₁ Φ₂ h₁ h₂ hf t ht
  linarith

theorem diagonal_bounded_above (t : ℕ) : position t (1,1) ≤ 7/10 := by
  by_cases ht : t < 3
  · interval_cases t <;> norm_num [position, estimate_one, estimate_two,
      firstEstimate, secondEstimate, ofMatrix, matrixEquiv]
  · have ht3 : 3 ≤ t := by omega
    have hm := Nat.mod_add_div t 2
    have hml := Nat.mod_lt t (by omega : 0 < 2)
    have hcases : (∃ k, t = 2*k+3) ∨ (∃ k, t = 2*k+4) := by
      by_cases he : t%2 = 0
      · exact Or.inr ⟨t/2-2, by omega⟩
      · exact Or.inl ⟨t/2-1, by omega⟩
    rcases hcases with ⟨k,rfl⟩ | ⟨k,rfl⟩
    · rw [(position_tail k).1]
      simp only [position_three, PiLp.add_apply, PiLp.smul_apply, smul_eq_mul]
      norm_num [drift, ofMatrix, matrixEquiv]
      nlinarith [Nat.cast_nonneg (α := ℝ) k]
    · rw [(position_tail k).2]
      simp only [position_four, PiLp.add_apply, PiLp.smul_apply, smul_eq_mul]
      norm_num [drift, ofMatrix, matrixEquiv]
      nlinarith [Nat.cast_nonneg (α := ℝ) k]

end EF21Muon.Divergence.EF21SignMuon
