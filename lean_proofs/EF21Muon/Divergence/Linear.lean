import EF21Muon.Divergence.Polar
import EF21Muon.Compression.ScaledSign

/-! Proposition `prop:reduction`: ascent on linear objectives, including both
momentum rules and arbitrary positive, nonsummable learning rates. -/

noncomputable section

namespace EF21Muon.Divergence

open LongTails.Algorithms.MatrixPolar Filter
open scoped BigOperators

def entrySign {m n : ℕ} (X : Space m n) : Space m n :=
  WithLp.toLp 2 (fun p => Compression.signBit (X p))

theorem entrySign_smul_pos {m n : ℕ} {c : ℝ} (hc : 0 < c) (X : Space m n) :
    entrySign (c • X) = entrySign X := by
  ext p
  simp only [entrySign, PiLp.toLp_apply, PiLp.smul_apply, smul_eq_mul,
    Compression.signBit, mul_nonneg_iff_of_pos_left hc]

inductive Placement where
  | signMuon | muonUSign | muonSign

def Placement.direction {m n : ℕ} (method : Placement) (X : Space m n) : Space m n :=
  match method with
  | .signMuon => entrySign (polar X)
  | .muonUSign => polar (entrySign X)
  | .muonSign => entrySign (polar (entrySign X))

theorem Placement.direction_smul_pos {m n : ℕ} (method : Placement)
    {c : ℝ} (hc : 0 < c) (X : Space m n) :
    method.direction (c • X) = method.direction X := by
  cases method <;> simp [Placement.direction, entrySign_smul_pos hc, polar_smul_pos hc]

inductive MomentumVariant where
  | standard | nesterov

def linearMomentum {m n : ℕ} (μ : ℝ) (G : Space m n) : ℕ → Space m n
  | 0 => 0
  | t + 1 => μ • linearMomentum μ G t + (1 - μ) • G

theorem linearMomentum_eq {m n : ℕ} (μ : ℝ) (G : Space m n) (t : ℕ) :
    linearMomentum μ G t = (1 - μ ^ t) • G := by
  induction t with
  | zero => simp [linearMomentum]
  | succ t ih =>
    rw [linearMomentum, ih, smul_smul, ← add_smul, pow_succ]
    congr 1
    ring

def effectiveMomentum {m n : ℕ} (variant : MomentumVariant) (μ : ℝ)
    (G : Space m n) (t : ℕ) : Space m n :=
  match variant with
  | .standard => linearMomentum μ G (t + 1)
  | .nesterov => (1 - μ) • G + μ • linearMomentum μ G (t + 1)

def momentumCoefficient (variant : MomentumVariant) (μ : ℝ) (t : ℕ) : ℝ :=
  match variant with
  | .standard => 1 - μ ^ (t + 1)
  | .nesterov => 1 - μ ^ (t + 2)

theorem effectiveMomentum_eq {m n : ℕ} (variant : MomentumVariant) (μ : ℝ)
    (G : Space m n) (t : ℕ) :
    effectiveMomentum variant μ G t = momentumCoefficient variant μ t • G := by
  cases variant
  · exact linearMomentum_eq μ G (t + 1)
  · simp only [effectiveMomentum, linearMomentum_eq, smul_smul, ← add_smul,
      momentumCoefficient]
    congr 1
    rw [show t + 2 = (t + 1) + 1 from rfl, pow_succ]
    ring

theorem momentumCoefficient_pos (variant : MomentumVariant) {μ : ℝ}
    (hμ : 0 ≤ μ) (hμ1 : μ < 1) (t : ℕ) : 0 < momentumCoefficient variant μ t := by
  cases variant <;> simp only [momentumCoefficient]
  · exact sub_pos.mpr (pow_lt_one₀ hμ hμ1 (by omega))
  · exact sub_pos.mpr (pow_lt_one₀ hμ hμ1 (by omega))

def linearRun {m n : ℕ} (method : Placement) (variant : MomentumVariant)
    (μ : ℝ) (η : ℕ → ℝ) (G X₀ : Space m n) : ℕ → Space m n
  | 0 => X₀
  | t + 1 => linearRun method variant μ η G X₀ t -
      η t • method.direction (effectiveMomentum variant μ G t)

theorem linearRun_step {m n : ℕ} (method : Placement) (variant : MomentumVariant)
    {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1) (η : ℕ → ℝ) (G X₀ : Space m n) (t : ℕ) :
    linearRun method variant μ η G X₀ (t + 1) =
      linearRun method variant μ η G X₀ t - η t • method.direction G := by
  rw [linearRun, effectiveMomentum_eq,
    method.direction_smul_pos (momentumCoefficient_pos variant hμ hμ1 t)]

theorem linearRun_eq {m n : ℕ} (method : Placement) (variant : MomentumVariant)
    {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1) (η : ℕ → ℝ) (G X₀ : Space m n) (t : ℕ) :
    linearRun method variant μ η G X₀ t =
      X₀ - (∑ k ∈ Finset.range t, η k) • method.direction G := by
  induction t with
  | zero => simp [linearRun]
  | succ t ih =>
    rw [linearRun_step method variant hμ hμ1, ih, Finset.sum_range_succ, add_smul]
    abel

theorem linear_objective_increment {m n : ℕ} (method : Placement)
    (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (η : ℕ → ℝ) (G X₀ : Space m n) (t : ℕ) :
    inner ℝ G (linearRun method variant μ η G X₀ (t + 1)) -
      inner ℝ G (linearRun method variant μ η G X₀ t) =
        -η t * inner ℝ G (method.direction G) := by
  rw [linearRun_step method variant hμ hμ1, inner_sub_right, real_inner_smul_right]
  ring

theorem linear_objective_strictAscent {m n : ℕ} (method : Placement)
    (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (η : ℕ → ℝ) (hη : ∀ t, 0 < η t) (G X₀ : Space m n)
    (hG : inner ℝ G (method.direction G) < 0) :
    StrictMono (fun t => inner ℝ G (linearRun method variant μ η G X₀ t)) := by
  apply strictMono_nat_of_lt_succ
  intro t
  have h := linear_objective_increment method variant hμ hμ1 η G X₀ t
  have hp := mul_pos_of_neg_of_neg (neg_lt_zero.mpr (hη t)) hG
  nlinarith

theorem linear_objective_tendsto_atTop {m n : ℕ} (method : Placement)
    (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (η : ℕ → ℝ) (G X₀ : Space m n)
    (hG : inner ℝ G (method.direction G) < 0)
    (hη : Tendsto (fun t => ∑ k ∈ Finset.range t, η k) atTop atTop) :
    Tendsto (fun t => inner ℝ G (linearRun method variant μ η G X₀ t)) atTop atTop := by
  have h := hη.atTop_mul_const (neg_pos.mpr hG)
  apply tendsto_atTop.mpr
  intro b
  filter_upwards [tendsto_atTop.mp h (b - inner ℝ G X₀)] with t ht
  rw [linearRun_eq method variant hμ hμ1, inner_sub_right, real_inner_smul_right]
  nlinarith

theorem linear_objective_constantStep {m n : ℕ} (method : Placement)
    (variant : MomentumVariant) {μ η : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    (hη : 0 < η) (G X₀ : Space m n) (hG : inner ℝ G (method.direction G) < 0) :
    Tendsto (fun t => inner ℝ G (linearRun method variant μ (fun _ => η) G X₀ t))
      atTop atTop := by
  apply linear_objective_tendsto_atTop method variant hμ hμ1 _ G X₀ hG
  simpa using (tendsto_natCast_atTop_atTop (R := ℝ)).atTop_mul_const hη

end EF21Muon.Divergence
