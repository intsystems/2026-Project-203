import EF21Muon.Divergence.EF21Cycle

/-! Lemma `lem:efsm_twocycles`: alternating targets from zero reach the
cycle with the correct diagonal sign. The two-step preamble selects a
different cycle in the divergence construction. -/

noncomputable section

namespace EF21Muon.Divergence.EF21SignMuon

open LongTails.Algorithms.MatrixPolar

def reflection (a b : ℝ) : Space 2 2 := ofMatrix !![a,b;b,-a]
def goodOdd (a b : ℝ) : Space 2 2 :=
  ofMatrix !![(a+b)/2,-(a+b)/2;-(a+b)/2,-(a+b)/2]
def goodEven (a b : ℝ) : Space 2 2 :=
  ofMatrix !![-(b-a)/2,(b-a)/2;(b-a)/2,(b-a)/2]

theorem compressor_reflection_neg {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    signCompressor (reflection a (-b)) = goodOdd a b := by
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    simp [signCompressor, Compression.scaledSign, Compression.scaledSignWith,
      Compression.entrywiseL1, Compression.signBit, reflection, goodOdd, ofMatrix, matrixEquiv,
      Fintype.sum_prod_type, Fin.sum_univ_succ, ha.le,
      not_le.mpr (neg_neg_of_pos ha), not_le.mpr (neg_neg_of_pos hb),
      abs_of_pos ha, abs_of_pos hb] <;> ring

theorem compressor_reflection_pos {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    signCompressor (reflection (-a) b) =
      ofMatrix !![-(a+b)/2,(a+b)/2;(a+b)/2,(a+b)/2] := by
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    simp [signCompressor, Compression.scaledSign, Compression.scaledSignWith,
      Compression.entrywiseL1, Compression.signBit, reflection, ofMatrix, matrixEquiv,
      Fintype.sum_prod_type, Fin.sum_univ_succ, ha.le, hb.le,
      not_le.mpr (neg_neg_of_pos ha), abs_of_pos ha, abs_of_pos hb] <;> ring

theorem good_preamble {a b : ℝ} (ha : 0 < a) (hab : a < b) :
    estimateStep (reflection a (-b)) 0 = goodOdd a b := by
  simpa only [estimateStep, sub_zero, zero_add] using compressor_reflection_neg ha (ha.trans hab)

theorem good_odd_to_even {a b : ℝ} (ha : 0 < a) (hab : a < b) :
    estimateStep (reflection a b) (goodOdd a b) = goodEven a b := by
  have hres : reflection a b - goodOdd a b = reflection (-((b-a)/2)) ((a+3*b)/2) := by
    ext ⟨i,j⟩
    fin_cases i <;> fin_cases j <;> simp [reflection, goodOdd, ofMatrix, matrixEquiv] <;> ring
  rw [estimateStep, hres, compressor_reflection_pos (by linarith) (by linarith)]
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;> simp [goodOdd, goodEven, ofMatrix, matrixEquiv] <;> ring

theorem good_even_to_odd {a b : ℝ} (ha : 0 < a) (hab : a < b) :
    estimateStep (reflection a (-b)) (goodEven a b) = goodOdd a b := by
  have hres : reflection a (-b) - goodEven a b = reflection ((a+b)/2) (-((3*b-a)/2)) := by
    ext ⟨i,j⟩
    fin_cases i <;> fin_cases j <;> simp [reflection, goodEven, ofMatrix, matrixEquiv] <;> ring
  rw [estimateStep, hres, compressor_reflection_neg (by linarith) (by linarith)]
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;> simp [goodOdd, goodEven, ofMatrix, matrixEquiv] <;> ring

def alternatingEstimate (a b : ℝ) : ℕ → Space 2 2
  | 0 => 0
  | t+1 => estimateStep (reflection a (if (t+1)%2 = 0 then b else -b))
      (alternatingEstimate a b t)

theorem harmless_cycle {a b : ℝ} (ha : 0 < a) (hab : a < b) (k : ℕ) :
    alternatingEstimate a b (2*k+1) = goodOdd a b ∧
    alternatingEstimate a b (2*k+2) = goodEven a b := by
  induction k with
  | zero =>
    constructor
    · simpa [alternatingEstimate] using good_preamble ha hab
    · simpa [alternatingEstimate, good_preamble ha hab] using good_odd_to_even ha hab
  | succ k ih =>
    have hodd : alternatingEstimate a b (2*(k+1)+1) = goodOdd a b := by
      rw [show 2*(k+1)+1 = (2*k+2)+1 by omega, alternatingEstimate, ih.2]
      simpa [show (2*k+2+1)%2 ≠ 0 by omega] using good_even_to_odd ha hab
    refine ⟨hodd, ?_⟩
    rw [show 2*(k+1)+2 = (2*(k+1)+1)+1 by omega, alternatingEstimate, hodd]
    simpa [show (2*(k+1)+1+1)%2 = 0 by omega] using good_odd_to_even ha hab

theorem harmless_average {a b : ℝ} (ha : 0 < a) :
    (goodOdd a b (1,1)+goodEven a b (1,1))/2 = -a/2 ∧ -a/2 < 0 := by
  constructor
  · simp [goodOdd, goodEven, ofMatrix, matrixEquiv]
    ring
  · linarith

end EF21Muon.Divergence.EF21SignMuon
