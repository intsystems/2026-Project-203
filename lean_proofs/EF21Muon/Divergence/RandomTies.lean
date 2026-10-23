import EF21Muon.Divergence.SignMuon
import EF21Muon.Divergence.MuonSign
import EF21Muon.Divergence.EF21Cycle
import EF21Muon.Compression.RandomTies

/-! Literal binary ties for the linear-objective reduction. The paper's added
no-zero condition makes the directions independent of all tie realizations. -/

noncomputable section
namespace EF21Muon.Divergence.RandomTies
open LongTails.Algorithms.MatrixPolar
open EF21Muon.RandomTies (tieSign tieSign_eq_of_ne_zero)

variable {m n : ℕ}
abbrev Bits (m n : ℕ) := Fin m × Fin n → Bool

def entrySignWith (b : Bits m n) (x : Space m n) : Space m n :=
  WithLp.toLp 2 (fun ij => tieSign (b ij) (x ij))

theorem entrySignWith_eq (b : Bits m n) (x : Space m n)
    (hx : ∀ ij, x ij ≠ 0) : entrySignWith b x = entrySign x := by
  ext ij
  exact tieSign_eq_of_ne_zero _ (hx ij)

def NoTies (method : Placement) (x : Space m n) : Prop :=
  match method with
  | .signMuon => ∀ ij, polar x ij ≠ 0
  | .muonUSign => ∀ ij, x ij ≠ 0
  | .muonSign => (∀ ij, x ij ≠ 0) ∧ (∀ ij, polar (entrySign x) ij ≠ 0)

def direction (method : Placement) (innerBits outerBits : Bits m n) (x : Space m n) : Space m n :=
  match method with
  | .signMuon => entrySignWith outerBits (polar x)
  | .muonUSign => polar (entrySignWith innerBits x)
  | .muonSign => entrySignWith outerBits (polar (entrySignWith innerBits x))

theorem direction_eq (method : Placement) (innerBits outerBits : Bits m n)
    (x : Space m n) (hx : NoTies method x) :
    direction method innerBits outerBits x = method.direction x := by
  cases method with
  | signMuon => exact entrySignWith_eq _ _ hx
  | muonUSign => simp only [direction, Placement.direction, entrySignWith_eq _ _ hx]
  | muonSign => simp only [direction, Placement.direction, entrySignWith_eq _ _ hx.1,
      entrySignWith_eq _ _ hx.2]

theorem noTies_smul_pos (method : Placement) {a : ℝ} (ha : 0 < a)
    (x : Space m n) (hx : NoTies method x) : NoTies method (a • x) := by
  have hentries (h : ∀ ij, x ij ≠ 0) : ∀ ij, (a • x) ij ≠ 0 :=
    fun ij => mul_ne_zero ha.ne' (h ij)
  cases method with
  | signMuon => simpa only [NoTies, polar_smul_pos ha] using hx
  | muonUSign => exact hentries hx
  | muonSign =>
    exact ⟨hentries hx.1, by simpa only [entrySign_smul_pos ha] using hx.2⟩

def run (method : Placement) (variant : MomentumVariant) (μ : ℝ) (η : ℕ → ℝ)
    (innerBits outerBits : ℕ → Bits m n) (G x : Space m n) : ℕ → Space m n
  | 0 => x
  | t+1 => run method variant μ η innerBits outerBits G x t-
      η t • direction method (innerBits t) (outerBits t) (effectiveMomentum variant μ G t)

/-- The corrected proposition, for every possible sequence of random tie outcomes. -/
theorem run_eq (method : Placement) (variant : MomentumVariant) {μ : ℝ}
    (hμ : 0 ≤ μ) (hμ1 : μ < 1) (η : ℕ → ℝ) (innerBits outerBits : ℕ → Bits m n)
    (G x : Space m n) (hG : NoTies method G) (t : ℕ) :
    run method variant μ η innerBits outerBits G x t = linearRun method variant μ η G x t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [run, linearRun, ih, direction_eq]
    rw [effectiveMomentum_eq]
    exact noTies_smul_pos method (momentumCoefficient_pos variant hμ hμ1 t) G hG

theorem constant_direction (method : Placement) (variant : MomentumVariant) {μ : ℝ}
    (hμ : 0 ≤ μ) (hμ1 : μ < 1) (innerBits outerBits : Bits m n)
    (G : Space m n) (hG : NoTies method G) (t : ℕ) :
    direction method innerBits outerBits (effectiveMomentum variant μ G t) = method.direction G := by
  rw [effectiveMomentum_eq, direction_eq method innerBits outerBits _
    (noTies_smul_pos method (momentumCoefficient_pos variant hμ hμ1 t) G hG),
    method.direction_smul_pos (momentumCoefficient_pos variant hμ hμ1 t)]

theorem signMuon_noTies : NoTies .signMuon (ofMatrix SignMuon.gradient) := by
  intro ij
  rw [SignMuon.polar_gradient]
  rcases ij with ⟨i,j⟩
  fin_cases i <;> fin_cases j <;>
    norm_num [ofMatrix, matrixEquiv, SignMuon.orthogonal]

theorem muonUSign_noTies : NoTies .muonUSign (ofMatrix MuonSign.gradient) := by
  rintro ⟨i,j⟩
  fin_cases i <;> fin_cases j <;> norm_num [ofMatrix, matrixEquiv, MuonSign.gradient]

theorem muonSign_noTies : NoTies .muonSign (ofMatrix MuonSign.gradient) := by
  refine ⟨muonUSign_noTies, ?_⟩
  intro ij
  rw [MuonSign.sign_gradient, MuonSign.polar_pattern]
  rcases ij with ⟨i,j⟩
  have hlo := MuonSign.sqrt_bounds.1
  have hhi := MuonSign.sqrt_bounds.2
  fin_cases i <;> fin_cases j <;>
    norm_num [ofMatrix, matrixEquiv, MuonSign.direction] <;> linarith

/-- Randomized signs preserve the article's exact counterexample trajectories. -/
theorem signMuon_run (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (η : ℕ → ℝ) (innerBits outerBits : ℕ → Bits 4 4) (x : Space 4 4) (t : ℕ) :
    run .signMuon variant μ η innerBits outerBits (ofMatrix SignMuon.gradient) x t =
      linearRun .signMuon variant μ η (ofMatrix SignMuon.gradient) x t :=
  run_eq _ _ hμ hμ1 _ _ _ _ _ signMuon_noTies t

theorem muonUSign_run (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (η : ℕ → ℝ) (innerBits outerBits : ℕ → Bits 5 5) (x : Space 5 5) (t : ℕ) :
    run .muonUSign variant μ η innerBits outerBits (ofMatrix MuonSign.gradient) x t =
      linearRun .muonUSign variant μ η (ofMatrix MuonSign.gradient) x t :=
  run_eq _ _ hμ hμ1 _ _ _ _ _ muonUSign_noTies t

theorem muonSign_run (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (η : ℕ → ℝ) (innerBits outerBits : ℕ → Bits 5 5) (x : Space 5 5) (t : ℕ) :
    run .muonSign variant μ η innerBits outerBits (ofMatrix MuonSign.gradient) x t =
      linearRun .muonSign variant μ η (ofMatrix MuonSign.gradient) x t :=
  run_eq _ _ hμ hμ1 _ _ _ _ _ muonSign_noTies t

end EF21Muon.Divergence.RandomTies

namespace EF21Muon.Divergence.EF21SignMuon
open LongTails.Algorithms.MatrixPolar

theorem residual_no_zeros (t : ℕ) (ij : Fin 2 × Fin 2) :
    (target (t+1)-estimate t) ij ≠ 0 := by
  rcases ij with ⟨i,j⟩
  rcases lt_or_ge t 3 with ht | ht
  · interval_cases t <;>
      fin_cases i <;> fin_cases j <;>
      norm_num [target, estimate_one, estimate_two, show estimate 0 = 0 from rfl, firstTarget,
        secondTarget, negativeTarget, firstEstimate, secondEstimate,
        ofMatrix, matrixEquiv, preamble_one, preamble_two]
  · by_cases hp : t%2 = 0
    · let k := (t-4)/2
      have he : t = 2*k+4 := by dsimp [k]; omega
      rw [he, (wrong_sign_cycle k).2,
        show 2*k+4+1 = 2*(k+1)+3 by omega, target_odd]
      fin_cases i <;> fin_cases j <;>
        norm_num [negativeTarget, cycleA, ofMatrix, matrixEquiv]
    · let k := (t-3)/2
      have he : t = 2*k+3 := by dsimp [k]; omega
      rw [he, (wrong_sign_cycle k).1, show 2*k+3+1 = 2*k+4 by omega, target_even]
      fin_cases i <;> fin_cases j <;>
        norm_num [positiveTarget, cycleB, ofMatrix, matrixEquiv]

def estimateWith (bits : ℕ → Fin 2 × Fin 2 → Bool) : ℕ → Space 2 2
  | 0 => 0
  | t+1 => estimateWith bits t+
      EF21Muon.RandomTies.compressor (bits t) (target (t+1)-estimateWith bits t)

theorem estimateWith_eq (bits : ℕ → Fin 2 × Fin 2 → Bool) (t : ℕ) :
    estimateWith bits t = estimate t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [estimateWith, ih,
      EF21Muon.RandomTies.compressor_eq_of_no_zeros _ _ (residual_no_zeros t)]
    rfl

def positionWith (bits : ℕ → Fin 2 × Fin 2 → Bool) : ℕ → Space 2 2
  | 0 => 0
  | t+1 => positionWith bits t-estimateWith bits (t+1)

theorem positionWith_eq (bits : ℕ → Fin 2 × Fin 2 → Bool) (t : ℕ) :
    positionWith bits t = position t := by
  induction t with
  | zero => rfl
  | succ t ih => simp only [positionWith, position, ih, estimateWith_eq]

end EF21Muon.Divergence.EF21SignMuon
