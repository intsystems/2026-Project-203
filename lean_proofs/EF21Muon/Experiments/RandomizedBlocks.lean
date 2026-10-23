import EF21Muon.Experiments.BlockTrajectory
import EF21Muon.Experiments.RandomizedFresh

/-! Multiple smooth matrix blocks and worker-resolved randomized communication.
The objective, block smoothness, and conditional-noise assumptions are reused
from BlockTrajectory. Randomness across blocks may be correlated. -/

noncomputable section
namespace EF21Muon.Experiments.RandomizedBlocks
open scoped BigOperators
open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Distributed EF21Muon.Stochastic
open EF21Muon.Experiments.BlockTrajectory (Problem Innovations)
open LongTails.Algorithms.MatrixPolar
open LongTails.Proofs.MatrixSmooth
set_option maxHeartbeats 800000
set_option backward.isDefEq.respectTransparency false
set_option autoImplicit false
variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Certified quality of each block's numerical direction. -/
structure Oracle {m n : ℕ} (c : RandomizedTrajectory.Controls N m n Ω) where
  ρ : ℝ
  r : ℝ
  gain : ℝ
  δ : ℕ → ℝ
  ρ_nonneg : 0 ≤ ρ
  r_nonneg : 0 ≤ r
  gain_pos : 0 < gain
  δ_nonneg : ∀ t, 0 ≤ δ t
  dual : ∀ z : Space m n, nuclearNorm z ≤ ρ*‖z‖
  radius : ∀ t h, spectralNorm (c.direction t h) ≤ r
  pairing : ∀ t h, gain*nuclearNorm h-δ t ≤ inner ℝ h (c.direction t h)

variable (P : Problem N m n) (c : ∀ i, RandomizedTrajectory.Controls N (m i) (n i) Ω)
  (O : Innovations N m n μ) (R : ∀ i, Trajectory.Roundoff N (m i) (n i) Ω)

/-- Simultaneous block updates using stochastic gradients at the full shared
broadcast model. The local model query is not an isolated block query. -/
def advance (s : States N m n) (t : ℕ) (ω : Ω) : States N m n := fun i =>
  let a := fun j => P.localGrad j (fun k => (s k).w) i+O.value t j ω i
  let M := fun j => (1-(c i).β) • (s i).M j+(c i).β • a j
  let V := fun j => (1-(c i).θ) • M j+(c i).θ • a j
  let G := fun j => (s i).G j+(c i).CU t j ω (V j-(s i).G j)+(R i).uplink t j ω
  let X := (s i).x-(c i).η t • (c i).direction t (average G)
  (X, (s i).w+(c i).CP t ω (X-(s i).w)+(R i).downlink t ω, M, G)

def run (x₀ : Model m n) : ℕ → Ω → States N m n
  | 0, _ => fun i => LayerState.start (x₀ i)
  | t+1, ω => advance P c O R (run x₀ t ω) t ω

theorem advance_block (s : States N m n) (t : ℕ) (ω : Ω) (i : Fin p) :
    advance P c O R s t ω i =
      RandomizedTrajectory.advance (P.blockGradients i) (c i) (O.block i).innovations (R i) (s i) t ω := by
  have hlocal (j : Fin N) : P.localGrad j (fun k => (s k).w) i =
      (P.blockGradients i).grad j (s i).w := (P.block_local i j (fun k => (s k).w)).symm
  unfold advance
  simp_rw [hlocal]
  rfl

theorem run_block (x₀ : Model m n) (t : ℕ) (ω : Ω) (i : Fin p) :
    run P c O R x₀ t ω i =
      RandomizedTrajectory.run (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t ω := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change advance P c O R (run P c O R x₀ t ω) t ω i = _
    rw [advance_block, ih]
    rfl

theorem run_adapted
    (hU : ∀ i t j, Measurable[O.filtration (t+1)] ((R i).uplink t j))
    (hP : ∀ i t, Measurable[O.filtration (t+1)] ((R i).downlink t))
    (hCU : ∀ i t j, Measurable[(O.filtration (t+1)).prod
      (inferInstance : MeasurableSpace (Space (m i) (n i)))]
      (fun z : Ω × Space (m i) (n i) => (c i).CU t j z.1 z.2))
    (hCP : ∀ i t, Measurable[(O.filtration (t+1)).prod
      (inferInstance : MeasurableSpace (Space (m i) (n i)))]
      (fun z : Ω × Space (m i) (n i) => (c i).CP t z.1 z.2))
    (x₀ : Model m n) (t : ℕ) : Measurable[O.filtration t] (run P c O R x₀ t) := by
  let s := run P c O R x₀ t
  have h (i : Fin p) : Measurable[O.filtration t] (fun ω => s ω i) := by
    dsimp only [s]
    simp_rw [run_block]
    exact RandomizedTrajectory.run_adapted (P.blockGradients i) (c i) (O.block i) (R i)
      (hU i) (hP i) (hCU i) (hCP i) (x₀ i) t
  let : MeasurableSpace Ω := O.filtration t
  exact Measurable.of_eval h

theorem gradient_at_run (x₀ : Model m n) (t : ℕ) (ω : Ω) (i : Fin p) :
    (P.blockGradients i).mean
      (RandomizedTrajectory.run (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t ω).x =
      P.grad (points (run P c O R x₀ t ω)) i := by
  rw [← run_block]
  exact P.block_mean i (points (run P c O R x₀ t ω))

def errorSample (x₀ : Model m n) (i : Fin p) (t : ℕ) (ω : Ω) : ℝ :=
  ‖RandomizedTrajectory.estimate (P.blockGradients i) (c i) (O.block i).innovations (R i)
      (run P c O R x₀ t ω i) t ω-P.grad (points (run P c O R x₀ t ω)) i‖

def error (x₀ : Model m n) (i : Fin p) (t : ℕ) : ℝ :=
  RandomizedTrajectory.estimatorError (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t

def budget (x₀ : Model m n) (i : Fin p) (T : ℕ) : ℝ :=
  RandomizedTrajectory.trackingBudget (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) T

theorem errorSample_eq (x₀ : Model m n) (i : Fin p) (t : ℕ) (ω : Ω) :
    errorSample P c O R x₀ i t ω =
      ‖RandomizedTrajectory.estimate (P.blockGradients i) (c i) (O.block i).innovations (R i)
        (RandomizedTrajectory.run (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t ω) t ω-
          (P.blockGradients i).mean
            (RandomizedTrajectory.run (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t ω).x‖ := by
  unfold errorSample
  rw [← gradient_at_run, run_block]

theorem error_integrable (x₀ : Model m n) (i : Fin p) (t : ℕ) :
    Integrable (errorSample P c O R x₀ i t) μ := by
  change Integrable (fun ω => errorSample P c O R x₀ i t ω) μ
  simp_rw [errorSample_eq]
  exact RandomizedTrajectory.estimatorError_integrable (P.blockGradients i) (c i)
    (O.block i).innovations (R i) (x₀ i) t

theorem integral_error (x₀ : Model m n) (i : Fin p) (t : ℕ) :
    (∫ ω, errorSample P c O R x₀ i t ω ∂μ) = error P c O R x₀ i t := by
  simp_rw [errorSample_eq]
  rfl

/-- Each tracking budget is obtained from the simultaneous full-model run;
there is no assumed tracking recurrence or cross-block independence. -/
theorem tracking (x₀ : Model m n) (i : Fin p) (T : ℕ) :
    (∑ t ∈ Finset.range T, (c i).η t*error P c O R x₀ i t) ≤ budget P c O R x₀ i T :=
  RandomizedTrajectory.tracking (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) T

theorem objective_integrable (x₀ : Model m n) (t : ℕ) :
    Integrable (fun ω => P.f (points (run P c O R x₀ t ω))) μ := by
  let X := fun ω => points (run P c O R x₀ t ω)
  have hX (i : Fin p) : MemLp (fun ω => X ω i) 2 μ := by
    dsimp only [X, points]
    simp_rw [run_block]
    exact (RandomizedTrajectory.run_memLp (P.blockGradients i) (c i) (O.block i).innovations
      (R i) (x₀ i) t).1
  have hcont : Continuous P.f := continuous_iff_continuousAt.mpr
    (fun x => (P.derivative x).continuousAt)
  have hm : AEStronglyMeasurable (fun ω => P.f (X ω)) μ := by
    apply (hcont.measurable.comp_aemeasurable _).aestronglyMeasurable
    exact AEMeasurable.of_eval (fun i => (hX i).aestronglyMeasurable.aemeasurable)
  have hi : Integrable (fun ω => |P.f 0|+|P.floor|+
      ∑ i, (‖P.grad 0 i‖*‖X ω i‖+P.L i/2*‖X ω i‖^2)) μ := by
    apply (integrable_const _).add
    exact integrable_finsetSum _ (fun i _ =>
      (((hX i).integrable (by norm_num)).norm.const_mul _).add
        ((square_integrable (hX i)).const_mul _))
  apply hi.mono' hm
  filter_upwards [] with ω
  change |P.f (X ω)| ≤ _
  have hpos : 0 ≤ ∑ i, (‖P.grad 0 i‖*‖X ω i‖+P.L i/2*‖X ω i‖^2) :=
    Finset.sum_nonneg (fun i _ => add_nonneg (mul_nonneg (norm_nonneg _) (norm_nonneg _))
      (mul_nonneg (div_nonneg (P.L_nonneg i) (by norm_num)) (sq_nonneg _)))
  have hs := smooth_upper P.derivative P.smooth 0 (X ω)
  simp only [Pi.zero_apply, sub_zero] at hs
  have hb : (∑ i, (inner ℝ (P.grad 0 i) (X ω i)+
      P.L i/2*spectralNorm (X ω i)^2)) ≤
      ∑ i, (‖P.grad 0 i‖*‖X ω i‖+P.L i/2*‖X ω i‖^2) := by
    apply Finset.sum_le_sum
    intro i _
    exact add_le_add (real_inner_le_norm _ _) (mul_le_mul_of_nonneg_left
      ((sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mpr (spectralNorm_le_frobenius _))
        (div_nonneg (P.L_nonneg i) (by norm_num)))
  apply abs_le.mpr
  constructor
  · linarith only [P.lower (X ω), neg_abs_le P.floor, abs_nonneg (P.f 0), hpos]
  · linarith only [hs, hb, le_abs_self (P.f 0), abs_nonneg P.floor]

variable (d : ∀ i, Oracle (c i))

include d in
theorem score_integrable (x₀ : Model m n) (i : Fin p) (t : ℕ) :
    Integrable (fun ω => nuclearNorm (P.grad (points (run P c O R x₀ t ω)) i)) μ := by
  have hX := RandomizedTrajectory.mean_gradient_memLp (P.blockGradients i) (c i)
    (O.block i).innovations (R i) (x₀ i) t
  simp_rw [← gradient_at_run]
  apply ((hX.integrable (by norm_num)).norm.const_mul (d i).ρ).mono'
    (continuous_nuclearNorm.measurable.comp_aemeasurable
      hX.aestronglyMeasurable.aemeasurable).aestronglyMeasurable
  filter_upwards [] with ω
  change ‖nuclearNorm ((P.blockGradients i).mean
    (RandomizedTrajectory.run (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t ω).x)‖ ≤ _
  rw [Real.norm_of_nonneg (nuclearNorm_nonneg _)]
  exact (d i).dual _

/-- Per-block smoothness is summed before telescoping the one global objective. -/
theorem objective_step (x₀ : Model m n) (t : ℕ) (ω : Ω) :
    P.f (points (run P c O R x₀ (t+1) ω)) ≤ P.f (points (run P c O R x₀ t ω))+
      ∑ i, (-(c i).η t*((d i).gain*nuclearNorm (P.grad (points (run P c O R x₀ t ω)) i))+
        (c i).η t*(((d i).gain+(d i).r)*(d i).ρ*errorSample P c O R x₀ i t ω+(d i).δ t)+
          P.L i*(d i).r^2/2*((c i).η t)^2) := by
  let s := run P c O R x₀ t ω
  have hs := smooth_upper P.derivative P.smooth (points s)
    (points (run P c O R x₀ (t+1) ω))
  apply hs.trans
  apply add_le_add (le_refl _)
  apply Finset.sum_le_sum
  intro i _
  let H := RandomizedTrajectory.estimate (P.blockGradients i) (c i) (O.block i).innovations (R i)
    (s i) t ω
  let D := (c i).direction t H
  let G := P.grad (points s) i
  have hstep : points (run P c O R x₀ (t+1) ω) i-points s i = -((c i).η t • D) := by
    change (advance P c O R s t ω i).x-(s i).x = _
    rw [advance_block]
    dsimp only [RandomizedTrajectory.advance, RandomizedTrajectory.server, LayerState.x]
    module
  have hn : spectralNorm (-((c i).η t • D)) = spectralNorm ((c i).η t • D) := by
    simp only [spectralNorm, map_neg, norm_neg]
  rw [hstep, inner_neg_right, inner_smul_right, hn, spectralNorm_smul,
    abs_of_nonneg ((c i).η_nonneg t), mul_pow]
  have hquad := mul_le_mul_of_nonneg_left
    ((sq_le_sq₀ (norm_nonneg _) (d i).r_nonneg).mpr ((d i).radius t H))
    (mul_nonneg (div_nonneg (P.L_nonneg i) (by norm_num : (0 : ℝ) ≤ 2))
      (sq_nonneg ((c i).η t)))
  have hg := radius_direction_gain G H D (d i).gain_pos.le ((d i).radius t H)
    ((d i).pairing t H)
  have hd := mul_le_mul_of_nonneg_left ((d i).dual (G-H))
    (add_nonneg (d i).gain_pos.le (d i).r_nonneg)
  rw [norm_sub_rev] at hd
  have hp : (d i).gain*nuclearNorm G-((d i).gain+(d i).r)*(d i).ρ*‖H-G‖-
      (d i).δ t ≤ inner ℝ G D := by nlinarith only [hg, hd]
  have hh := mul_le_mul_of_nonneg_left hp ((c i).η_nonneg t)
  change P.L i/2*((c i).η t)^2*spectralNorm D^2 ≤
    P.L i/2*((c i).η t)^2*(d i).r^2 at hquad
  change -((c i).η t*inner ℝ G D)+P.L i/2*(((c i).η t)^2*spectralNorm D^2) ≤
    -(c i).η t*((d i).gain*nuclearNorm G)+
      (c i).η t*(((d i).gain+(d i).r)*(d i).ρ*‖H-G‖+(d i).δ t)+
        P.L i*(d i).r^2/2*((c i).η t)^2
  nlinarith only [hquad, hh]

def score (x₀ : Model m n) (i : Fin p) (t : ℕ) : ℝ :=
  ∫ ω, nuclearNorm (P.grad (points (run P c O R x₀ t ω)) i) ∂μ

def objectiveValue (x₀ : Model m n) (t : ℕ) : ℝ :=
  ∫ ω, P.f (points (run P c O R x₀ t ω)) ∂μ

def weightedScore (x₀ : Model m n) (t : ℕ) : ℝ :=
  ∑ i, (c i).η t*((d i).gain*score P c O R x₀ i t)

def penalty (x₀ : Model m n) (i : Fin p) (t : ℕ) : ℝ :=
  (c i).η t*(((d i).gain+(d i).r)*(d i).ρ*error P c O R x₀ i t+(d i).δ t)+
    P.L i*(d i).r^2/2*((c i).η t)^2

/-- The objective gap occurs only once, regardless of the number of blocks. -/
def bound (x₀ : Model m n) (T : ℕ) : ℝ := P.f x₀-P.floor+
  ∑ i, (((d i).gain+(d i).r)*(d i).ρ*budget P c O R x₀ i T+
    (∑ t ∈ Finset.range T, (c i).η t*(d i).δ t)+
      P.L i*(d i).r^2/2*(∑ t ∈ Finset.range T, ((c i).η t)^2))

/-- Mean block weight, matching the paper's normalization by the mean layer
step. Equal schedules and unit gains therefore measure a sum of block norms. -/
def mass (T : ℕ) : ℝ := (p : ℝ)⁻¹*∑ t ∈ Finset.range T, ∑ i, (c i).η t*(d i).gain

theorem objective_drift (x₀ : Model m n) (t : ℕ) :
    objectiveValue P c O R x₀ (t+1) ≤ objectiveValue P c O R x₀ t-
      weightedScore P c O R d x₀ t+∑ i, penalty P c O R d x₀ i t := by
  let A := fun i ω => nuclearNorm (P.grad (points (run P c O R x₀ t ω)) i)
  let B := fun i ω => errorSample P c O R x₀ i t ω
  let K := fun i => ((d i).gain+(d i).r)*(d i).ρ
  let C := fun i => P.L i*(d i).r^2/2*((c i).η t)^2
  have hA i : Integrable (A i) μ := score_integrable P c O R d x₀ i t
  have hB i : Integrable (B i) μ := error_integrable P c O R x₀ i t
  have hi i : Integrable (fun ω => -(c i).η t*((d i).gain*A i ω)+
      (c i).η t*(K i*B i ω+(d i).δ t)+C i) μ :=
    ((((hA i).const_mul _).const_mul _).add
      ((((hB i).const_mul _).add (integrable_const _)).const_mul _)).add (integrable_const _)
  have he i : (∫ ω, -(c i).η t*((d i).gain*A i ω)+
      (c i).η t*(K i*B i ω+(d i).δ t)+C i ∂μ) =
      -(c i).η t*((d i).gain*score P c O R x₀ i t)+penalty P c O R d x₀ i t := by
    have h1 : Integrable (fun ω => -(c i).η t*((d i).gain*A i ω)) μ :=
      ((hA i).const_mul (d i).gain).const_mul (-(c i).η t)
    have h2 : Integrable (fun ω => (c i).η t*(K i*B i ω+(d i).δ t)) μ :=
      (((hB i).const_mul (K i)).add (integrable_const ((d i).δ t))).const_mul ((c i).η t)
    have hadd1 := integral_add h1 h2
    have hadd2 := integral_add (h1.add h2) (integrable_const (C i))
    dsimp only [Pi.add_apply] at hadd1 hadd2
    rw [hadd2, hadd1,
      integral_const_mul, integral_const_mul, integral_const_mul,
      integral_add ((hB i).const_mul (K i)) (integrable_const ((d i).δ t)), integral_const_mul]
    simp only [integral_const, probReal_univ, one_smul]
    rw [show (∫ ω, B i ω ∂μ) = error P c O R x₀ i t from integral_error P c O R x₀ i t]
    dsimp only [A, K, C, score, penalty]
    ring
  have h := integral_mono (objective_integrable P c O R x₀ (t+1))
    ((objective_integrable P c O R x₀ t).add (integrable_finsetSum _ (fun i _ => hi i)))
      (objective_step P c O R d x₀ t)
  change objectiveValue P c O R x₀ (t+1) ≤
    (∫ ω, P.f (points (run P c O R x₀ t ω))+
      ∑ i, (-(c i).η t*((d i).gain*A i ω)+
        (c i).η t*(K i*B i ω+(d i).δ t)+C i) ∂μ) at h
  rw [integral_add (objective_integrable P c O R x₀ t)
    (integrable_finsetSum _ (fun i _ => hi i)),
    integral_finsetSum _ (fun i _ => hi i)] at h
  simp_rw [he] at h
  simpa only [Finset.sum_add_distrib, neg_mul, ← Finset.sum_neg_distrib,
    objectiveValue, weightedScore, sub_eq_add_neg, add_assoc] using h

theorem score_sum_bound (x₀ : Model m n) (T : ℕ) :
    (∑ t ∈ Finset.range T, weightedScore P c O R d x₀ t) ≤ bound P c O R d x₀ T := by
  have hlower : P.floor ≤ objectiveValue P c O R x₀ T := by
    have h := integral_mono (integrable_const P.floor) (objective_integrable P c O R x₀ T)
      (fun ω => P.lower (points (run P c O R x₀ T ω)))
    simpa only [integral_const, probReal_univ, one_smul, objectiveValue] using h
  have hzero : objectiveValue P c O R x₀ 0 = P.f x₀ := by
    change (∫ _ : Ω, P.f x₀ ∂μ) = P.f x₀
    simp
  have hs := scheduled_descent_sum (objectiveValue P c O R x₀) (fun _ => 1)
    (weightedScore P c O R d x₀) (fun t => ∑ i, penalty P c O R d x₀ i t)
      P.floor 0 T hlower (fun t _ => by
        simpa only [one_mul, zero_div, zero_mul, add_zero] using objective_drift P c O R d x₀ t)
  simp only [one_mul, zero_div, zero_mul, add_zero, hzero] at hs
  have hp : (∑ t ∈ Finset.range T, ∑ i, penalty P c O R d x₀ i t) ≤
      ∑ i, (((d i).gain+(d i).r)*(d i).ρ*budget P c O R x₀ i T+
        (∑ t ∈ Finset.range T, (c i).η t*(d i).δ t)+
          P.L i*(d i).r^2/2*(∑ t ∈ Finset.range T, ((c i).η t)^2)) := by
    rw [Finset.sum_comm]
    apply Finset.sum_le_sum
    intro i _
    have he : (∑ t ∈ Finset.range T, penalty P c O R d x₀ i t) =
        ((d i).gain+(d i).r)*(d i).ρ*(∑ t ∈ Finset.range T, (c i).η t*error P c O R x₀ i t)+
          (∑ t ∈ Finset.range T, (c i).η t*(d i).δ t)+
            P.L i*(d i).r^2/2*(∑ t ∈ Finset.range T, ((c i).η t)^2) := by
      simp only [penalty, Finset.mul_sum, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro t _
      ring
    rw [he]
    exact add_le_add (add_le_add
      (mul_le_mul_of_nonneg_left (tracking P c O R x₀ i T)
        (mul_nonneg (add_nonneg (d i).gain_pos.le (d i).r_nonneg) (d i).ρ_nonneg))
          (le_refl _)) (le_refl _)
  exact hs.trans (add_le_add (le_refl _) hp)

/-- Closed expected stationarity for the full simultaneous N-worker run. -/
theorem expected_stationarity_sum (x₀ : Model m n) (T : ℕ) :
    (∫ ω, ∑ t ∈ Finset.range T, ∑ i,
      (c i).η t*((d i).gain*nuclearNorm (P.grad (points (run P c O R x₀ t ω)) i)) ∂μ) ≤
        bound P c O R d x₀ T := by
  rw [integral_finsetSum _ (fun t _ => integrable_finsetSum _
    (fun i _ => ((score_integrable P c O R d x₀ i t).const_mul _).const_mul _))]
  simp_rw [integral_finsetSum _ (fun i _ =>
    ((score_integrable P c O R d x₀ i _).const_mul _).const_mul _),
      integral_const_mul]
  exact score_sum_bound P c O R d x₀ T

/-- Different block schedules and gains are normalized by their mean weight. -/
theorem expected_stationarity (x₀ : Model m n) (T : ℕ) (hmass : 0 < mass c d T) :
    (∫ ω, (∑ t ∈ Finset.range T, ∑ i,
      (c i).η t*((d i).gain*nuclearNorm (P.grad (points (run P c O R x₀ t ω)) i)))/
        mass c d T ∂μ) ≤ bound P c O R d x₀ T/mass c d T := by
  rw [integral_div]
  exact div_le_div_of_nonneg_right (expected_stationarity_sum P c O R d x₀ T) hmass.le

/-- One theorem includes adaptation of every model and worker memory and the
closed weighted stationarity bound. θ_i=0 and θ_i=β_i give EMA and Nesterov. -/
theorem fresh_stationarity
    (hU : ∀ i t j, Measurable[O.filtration (t+1)] ((R i).uplink t j))
    (hP : ∀ i t, Measurable[O.filtration (t+1)] ((R i).downlink t))
    (hCU : ∀ i t j, Measurable[(O.filtration (t+1)).prod
      (inferInstance : MeasurableSpace (Space (m i) (n i)))]
      (fun z : Ω × Space (m i) (n i) => (c i).CU t j z.1 z.2))
    (hCP : ∀ i t, Measurable[(O.filtration (t+1)).prod
      (inferInstance : MeasurableSpace (Space (m i) (n i)))]
      (fun z : Ω × Space (m i) (n i) => (c i).CP t z.1 z.2))
    (x₀ : Model m n) (T : ℕ) (hmass : 0 < mass c d T) :
    (∀ t, Measurable[O.filtration t] (run P c O R x₀ t)) ∧
    (∫ ω, (∑ t ∈ Finset.range T, ∑ i,
      (c i).η t*((d i).gain*nuclearNorm (P.grad (points (run P c O R x₀ t ω)) i)))/
        mass c d T ∂μ) ≤ bound P c O R d x₀ T/mass c d T := by
  exact ⟨run_adapted P c O R hU hP hCU hCP x₀, expected_stationarity P c O R d x₀ T hmass⟩

end EF21Muon.Experiments.RandomizedBlocks
