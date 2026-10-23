import EF21Muon.Divergence.Linear
import EF21Muon.Algorithm

/-! Lemma `lem:efsm_cycle`: the exact wrong-sign cycle of EF21-SignMuon.
Targets here are prescribed; realization by a smooth objective is a separate
obligation, rather than an assumption hidden in the cycle theorem. -/

noncomputable section

namespace EF21Muon.Divergence.EF21SignMuon

open LongTails.Algorithms.MatrixPolar Matrix Filter

def firstTarget : Space 2 2 := ofMatrix !![-4/5, 3/5; -3/5, -4/5]
def secondTarget : Space 2 2 := ofMatrix !![3/5, -4/5; 0, 0]
def positiveTarget : Space 2 2 := ofMatrix !![7/25, 24/25; 24/25, -7/25]
def negativeTarget : Space 2 2 := ofMatrix !![7/25, -24/25; -24/25, -7/25]
def firstEstimate : Space 2 2 := ofMatrix !![-7/10, 7/10; -7/10, -7/10]
def secondEstimate : Space 2 2 := ofMatrix !![7/20, -7/20; 7/20, 7/20]
def cycleB : Space 2 2 := ofMatrix !![-61/200, -201/200; -61/200, -61/200]
def cycleA : Space 2 2 := ofMatrix !![131/200, -9/200; 131/200, 131/200]
def drift : Space 2 2 := ofMatrix !![-7/20, 21/20; -7/20, -7/20]

def estimateStep (target estimate : Space 2 2) : Space 2 2 :=
  estimate + signCompressor (target - estimate)

theorem preamble_one : estimateStep firstTarget 0 = firstEstimate := by
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [estimateStep, signCompressor, Compression.scaledSign,
      Compression.scaledSignWith, Compression.entrywiseL1, Compression.signBit,
      firstTarget, firstEstimate, ofMatrix, matrixEquiv,
      Fintype.sum_prod_type, Fin.sum_univ_succ]

theorem preamble_two : estimateStep secondTarget firstEstimate = secondEstimate := by
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [estimateStep, signCompressor, Compression.scaledSign,
      Compression.scaledSignWith, Compression.entrywiseL1, Compression.signBit,
      secondTarget, firstEstimate, secondEstimate, ofMatrix, matrixEquiv,
      Fintype.sum_prod_type, Fin.sum_univ_succ]

theorem enter_cycle : estimateStep negativeTarget secondEstimate = cycleB := by
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [estimateStep, signCompressor, Compression.scaledSign,
      Compression.scaledSignWith, Compression.entrywiseL1, Compression.signBit,
      negativeTarget, secondEstimate, cycleB, ofMatrix, matrixEquiv,
      Fintype.sum_prod_type, Fin.sum_univ_succ]

theorem cycle_B_to_A : estimateStep positiveTarget cycleB = cycleA := by
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [estimateStep, signCompressor, Compression.scaledSign,
      Compression.scaledSignWith, Compression.entrywiseL1, Compression.signBit,
      positiveTarget, cycleB, cycleA, ofMatrix, matrixEquiv,
      Fintype.sum_prod_type, Fin.sum_univ_succ]

theorem cycle_A_to_B : estimateStep negativeTarget cycleA = cycleB := by
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [estimateStep, signCompressor, Compression.scaledSign,
      Compression.scaledSignWith, Compression.entrywiseL1, Compression.signBit,
      negativeTarget, cycleB, cycleA, ofMatrix, matrixEquiv,
      Fintype.sum_prod_type, Fin.sum_univ_succ]

def target (t : ℕ) : Space 2 2 :=
  if t = 0 then 0 else if t = 1 then firstTarget else if t = 2 then secondTarget
  else if t % 2 = 0 then positiveTarget else negativeTarget

def estimate : ℕ → Space 2 2
  | 0 => 0
  | t + 1 => estimateStep (target (t + 1)) (estimate t)

theorem estimate_one : estimate 1 = firstEstimate := by
  change estimateStep (target 1) 0 = _
  simpa [target] using preamble_one

theorem estimate_two : estimate 2 = secondEstimate := by
  change estimateStep (target 2) (estimate 1) = _
  rw [estimate_one]
  norm_num [target]
  exact preamble_two

theorem estimate_three : estimate 3 = cycleB := by
  change estimateStep (target 3) (estimate 2) = _
  rw [estimate_two]
  norm_num [target]
  exact enter_cycle

theorem estimate_four : estimate 4 = cycleA := by
  change estimateStep (target 4) (estimate 3) = _
  rw [estimate_three]
  norm_num [target]
  exact cycle_B_to_A

theorem target_odd (k : ℕ) : target (2*k+3) = negativeTarget := by
  simp [target, show 2*k+3 ≠ 1 by omega, show 2*k+3 ≠ 2 by omega]

theorem target_even (k : ℕ) : target (2*k+4) = positiveTarget := by
  simp [target, show 2*k+4 ≠ 1 by omega,
    show 2*k+4 ≠ 2 by omega, show (2*k+4)%2 = 0 by omega]

theorem wrong_sign_cycle (k : ℕ) :
    estimate (2*k+3) = cycleB ∧ estimate (2*k+4) = cycleA := by
  induction k with
  | zero =>
    norm_num [estimate, target, preamble_one, preamble_two, enter_cycle, cycle_B_to_A]
  | succ k ih =>
    have hB : estimate (2*(k+1)+3) = cycleB := by
      rw [show 2*(k+1)+3 = (2*k+4)+1 by omega, estimate, ih.2]
      rw [show 2*k+4+1 = 2*(k+1)+3 by omega, target_odd, cycle_A_to_B]
    constructor
    · exact hB
    · rw [show 2*(k+1)+4 = (2*(k+1)+3)+1 by omega, estimate, hB]
      rw [show 2*(k+1)+3+1 = 2*(k+1)+4 by omega, target_even, cycle_B_to_A]

theorem wrong_sign_average : (cycleA (1,1) + cycleB (1,1)) / 2 = 7/40 ∧
    positiveTarget (1,1) = -7/25 ∧ negativeTarget (1,1) = -7/25 := by
  norm_num [cycleA, cycleB, positiveTarget, negativeTarget, ofMatrix, matrixEquiv]

theorem drift_eq : -(cycleA + cycleB) = drift := by
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [cycleA, cycleB, drift, ofMatrix, matrixEquiv]

def position : ℕ → Space 2 2
  | 0 => 0
  | t + 1 => position t - estimate (t + 1)

theorem position_one : position 1 = ofMatrix !![7/10, -7/10; 7/10, 7/10] := by
  rw [position, estimate_one]
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;> norm_num [position, firstEstimate, ofMatrix, matrixEquiv]

theorem position_two : position 2 = ofMatrix !![7/20, -7/20; 7/20, 7/20] := by
  rw [position, estimate_two, position_one]
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;> norm_num [secondEstimate, ofMatrix, matrixEquiv]

theorem position_three : position 3 = ofMatrix !![131/200, 131/200; 131/200, 131/200] := by
  change ((0 - estimate 1) - estimate 2) - estimate 3 = _
  rw [estimate_one, estimate_two, estimate_three]
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [firstEstimate, secondEstimate, cycleB, ofMatrix, matrixEquiv]

theorem position_four : position 4 = ofMatrix !![0, 7/10; 0, 0] := by
  rw [position, estimate_four, position_three]
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;> norm_num [cycleA, ofMatrix, matrixEquiv]

theorem position_tail (k : ℕ) :
    position (2*k+3) = position 3 + (k : ℝ) • drift ∧
    position (2*k+4) = position 4 + (k : ℝ) • drift := by
  induction k with
  | zero => simp
  | succ k ih =>
    constructor
    · rw [show 2*(k+1)+3 = ((2*k+3)+1)+1 by omega, position, position,
        show 2*k+3+1 = 2*k+4 by omega, (wrong_sign_cycle k).2,
        show 2*k+4+1 = 2*(k+1)+3 by omega, (wrong_sign_cycle (k+1)).1, ih.1]
      rw [Nat.cast_add, Nat.cast_one, add_smul, one_smul, ← drift_eq]
      abel
    · rw [show 2*(k+1)+4 = ((2*k+4)+1)+1 by omega, position, position,
        show 2*k+4+1 = 2*(k+1)+3 by omega, (wrong_sign_cycle (k+1)).1,
        show 2*(k+1)+3+1 = 2*(k+1)+4 by omega, (wrong_sign_cycle (k+1)).2, ih.2]
      rw [Nat.cast_add, Nat.cast_one, add_smul, one_smul, ← drift_eq]
      abel

theorem position_period (t : ℕ) (ht : 3 ≤ t) : position (t+2) = position t + drift := by
  have hp := Nat.mod_add_div t 2
  have hm := Nat.mod_lt t (by omega : 0 < 2)
  have hcases : (∃ k, t = 2*k+3) ∨ (∃ k, t = 2*k+4) := by
    by_cases he : t % 2 = 0
    · exact Or.inr ⟨t/2-2, by omega⟩
    · exact Or.inl ⟨t/2-1, by omega⟩
  rcases hcases with ⟨k,rfl⟩ | ⟨k,rfl⟩
  · rw [show 2*k+3+2 = 2*(k+1)+3 by omega, (position_tail (k+1)).1,
      (position_tail k).1, Nat.cast_add, Nat.cast_one, add_smul, one_smul]
    abel
  · rw [show 2*k+4+2 = 2*(k+1)+4 by omega, (position_tail (k+1)).2,
      (position_tail k).2, Nat.cast_add, Nat.cast_one, add_smul, one_smul]
    abel

theorem diagonal_period (t : ℕ) (ht : 3 ≤ t) :
    position (t+2) (1,1) - position t (1,1) = -7/20 := by
  rw [position_period t ht]
  simp [drift, ofMatrix, matrixEquiv]

theorem slope_period (t : ℕ) (ht : 3 ≤ t) :
    -(7/12 : ℝ) * position (t+2) (1,1) -
      (-(7/12 : ℝ) * position t (1,1)) = 49/240 := by
  have h := diagonal_period t ht
  linarith

end EF21Muon.Divergence.EF21SignMuon
