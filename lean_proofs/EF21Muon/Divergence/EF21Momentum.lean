import EF21Muon.Divergence.EF21Preamble

/-! The prescribed gradients and momentum buffers for every `0 ≤ μ < 1`
and both momentum rules, with the polar targets proved from the buffers. -/

noncomputable section

namespace EF21Muon.Divergence.EF21SignMuon

open LongTails.Algorithms.MatrixPolar

def tailScale (variant : MomentumVariant) (μ : ℝ) : ℝ :=
  match variant with
  | .standard => 1
  | .nesterov => (1+2*μ)⁻¹

def amplitude (variant : MomentumVariant) (μ : ℝ) : ℝ :=
  (1+μ)*tailScale variant μ/(1-μ)

def momentumSeq (variant : MomentumVariant) (μ : ℝ) : ℕ → Space 2 2
  | 0 => 0
  | 1 => match variant with
      | .standard => firstTarget
      | .nesterov => nesterovFirst μ
  | 2 => match variant with
      | .standard => secondTarget
      | .nesterov => nesterovSecond μ
  | t+3 => tailMomentum (if (t+3)%2 = 0 then tailScale variant μ else -tailScale variant μ)

def requiredGradient (variant : MomentumVariant) (μ : ℝ) (t : ℕ) : Space 2 2 :=
  (1-μ)⁻¹ • (momentumSeq variant μ (t+1) - μ • momentumSeq variant μ t)

def fieldGradient (variant : MomentumVariant) (μ s : ℝ) : Space 2 2 :=
  ofMatrix !![0, s*amplitude variant μ; s*amplitude variant μ, -7/12]

theorem momentumSeq_odd (variant : MomentumVariant) (μ : ℝ) (k : ℕ) :
    momentumSeq variant μ (2*k+3) = tailMomentum (-tailScale variant μ) := by
  simp [momentumSeq]

theorem momentumSeq_even (variant : MomentumVariant) (μ : ℝ) (k : ℕ) :
    momentumSeq variant μ (2*k+4) = tailMomentum (tailScale variant μ) := by
  rw [show 2*k+4 = (2*k+1)+3 by omega]
  simp [momentumSeq, show (2*k+1+3)%2 = 0 by omega]

theorem requiredGradient_rec (variant : MomentumVariant) {μ : ℝ} (hμ1 : μ < 1) (t : ℕ) :
    momentumSeq variant μ (t+1) = μ • momentumSeq variant μ t +
      (1-μ) • requiredGradient variant μ t := by
  have hne : 1-μ ≠ 0 := by linarith
  simp only [requiredGradient, smul_smul, mul_inv_cancel₀ hne, one_smul]
  abel

theorem tail_effective (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (s : ℝ) :
    effective variant μ (tailMomentum (-s*tailScale variant μ))
      (tailMomentum (s*tailScale variant μ)) = tailMomentum s := by
  have hne : 1+2*μ ≠ 0 := by positivity
  cases variant
  · simp [effective, tailScale]
  · ext ⟨i,j⟩
    fin_cases i <;> fin_cases j <;>
      norm_num [effective, tailScale, tailMomentum, ofMatrix, matrixEquiv] <;>
      field_simp <;> ring

theorem momentumSeq_zero : momentumSeq .nesterov 0 = momentumSeq .standard 0 := by
  funext t
  match t with
  | 0 => rfl
  | 1 =>
    ext ⟨i,j⟩
    fin_cases i <;> fin_cases j <;>
      norm_num [momentumSeq, nesterovFirst, firstTarget, tau, ofMatrix, matrixEquiv]
  | 2 => simp [momentumSeq, nesterovSecond]
  | t+3 => simp [momentumSeq, tailScale]

theorem prescribed_polar (variant : MomentumVariant) {μ : ℝ}
    (hμ : 0 ≤ μ) (hμ1 : μ < 1) (t : ℕ) :
    polar (effective variant μ (momentumSeq variant μ t) (momentumSeq variant μ (t+1))) =
      target (t+1) := by
  match t with
  | 0 =>
    cases variant
    · simpa [momentumSeq, effective, target] using polar_firstTarget
    · by_cases hz : μ = 0
      · subst μ
        rw [momentumSeq_zero]
        simpa [effective, momentumSeq, target] using polar_firstTarget
      · simpa [momentumSeq, target] using polar_nesterovFirst (lt_of_le_of_ne hμ (Ne.symm hz))
  | 1 =>
    cases variant
    · simpa [momentumSeq, effective, target] using polar_secondTarget
    · by_cases hz : μ = 0
      · subst μ
        rw [momentumSeq_zero]
        simpa [effective, momentumSeq, target] using polar_secondTarget
      · simpa [momentumSeq, target] using polar_nesterovSecond (lt_of_le_of_ne hμ (Ne.symm hz))
  | 2 =>
    cases variant
    · simpa [momentumSeq, effective, tailScale, target, negativeTarget] using
        polar_tailMomentum (-1) (Or.inr rfl)
    · by_cases hz : μ = 0
      · subst μ
        rw [momentumSeq_zero]
        simpa [effective, momentumSeq, tailScale, target, negativeTarget] using
          polar_tailMomentum (-1) (Or.inr rfl)
      · simpa [momentumSeq, tailScale, nesterovThird, target] using
          polar_nesterovThird (lt_of_le_of_ne hμ (Ne.symm hz)) hμ1
  | t+3 =>
    by_cases he : (t+3)%2 = 0
    · have ht : t+3 = 2*((t+3)/2-2)+4 := by omega
      rw [ht, momentumSeq_even, show 2*((t+3)/2-2)+4+1 = 2*(((t+3)/2-2)+1)+3 by omega,
        momentumSeq_odd, target_odd]
      have hp := tail_effective variant hμ (-1)
      simp only [neg_neg, neg_one_mul, one_mul] at hp
      rw [hp]
      simpa [negativeTarget] using polar_tailMomentum (-1) (Or.inr rfl)
    · have ht : t+3 = 2*((t+3)/2-1)+3 := by omega
      rw [ht, momentumSeq_odd, show 2*((t+3)/2-1)+3+1 = 2*((t+3)/2-1)+4 by omega,
        momentumSeq_even, target_even]
      have hp := tail_effective variant hμ 1
      simp only [neg_one_mul, one_mul] at hp
      rw [hp]
      simpa [positiveTarget] using polar_tailMomentum 1 (Or.inl rfl)

theorem requiredGradient_odd (variant : MomentumVariant) {μ : ℝ} (hμ1 : μ < 1) (k : ℕ) :
    requiredGradient variant μ (2*k+3) = fieldGradient variant μ 1 := by
  have hne : 1-μ ≠ 0 := by linarith
  unfold requiredGradient
  rw [show 2*k+3+1 = 2*k+4 by omega, momentumSeq_even, momentumSeq_odd]
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [tailMomentum, fieldGradient, amplitude, ofMatrix, matrixEquiv] <;>
    field_simp
  all_goals ring

theorem requiredGradient_even (variant : MomentumVariant) {μ : ℝ} (hμ1 : μ < 1) (k : ℕ) :
    requiredGradient variant μ (2*k+4) = fieldGradient variant μ (-1) := by
  have hne : 1-μ ≠ 0 := by linarith
  unfold requiredGradient
  rw [show 2*k+4+1 = 2*(k+1)+3 by omega, momentumSeq_odd, momentumSeq_even]
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [tailMomentum, fieldGradient, amplitude, ofMatrix, matrixEquiv] <;>
    field_simp <;> ring

end EF21Muon.Divergence.EF21SignMuon
