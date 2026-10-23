import EF21Muon.Experiments.FreshTrajectory
import EF21Muon.Distributed.Setup

/-! Fixed-momentum experimental trajectories with multiple matrix blocks.
The arbitrary-pair layerwise spectral/nuclear smoothness assumptions are the
same as in the paper theorems. Block independence is derived from those
assumptions, rather than added as a hypothesis. All blocks use one probability
space and filtration; noise may be correlated between blocks. -/

noncomputable section
namespace EF21Muon.Experiments.BlockTrajectory
open scoped BigOperators
open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Distributed
open EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open LongTails.Proofs.MatrixSmooth


set_option maxHeartbeats 800000
set_option backward.isDefEq.respectTransparency false
set_option autoImplicit false

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Ordinary layerwise smoothness, with the same arbitrary-pair quantifiers
and individual worker constants as the earlier distributed paper results. -/
structure Problem (N : ℕ) (m n : Fin p → ℕ) where
  workers_pos : 0 < N
  f : Model m n → ℝ
  grad : Model m n → Model m n
  localGrad : Fin N → Model m n → Model m n
  L : Fin p → ℝ
  localL : Fin p → Fin N → ℝ
  L_nonneg : ∀ i, 0 ≤ L i
  localL_nonneg : ∀ i j, 0 ≤ localL i j
  floor : ℝ
  lower : ∀ x, floor ≤ f x
  derivative : ∀ x, HasFDerivAt f (gradientMap (grad x)) x
  smooth : ∀ x y i, nuclearNorm (grad x i-grad y i) ≤ L i*spectralNorm (x i-y i)
  localSmooth : ∀ j x y i,
    nuclearNorm (localGrad j x i-localGrad j y i) ≤ localL i j*spectralNorm (x i-y i)
  mean_gradient : ∀ x i, average (fun j => localGrad j x i) = grad x i

/-- Direct reuse of the earlier paper problem's objective and smoothness
assumptions. The experimental schedules and direction quality are separate. -/
def Problem.ofDistributed {a : Fin p → Parameters}
    (Q : EF21Muon.Distributed.Problem N m n μ a) : Problem N m n where
  workers_pos := Q.workers_pos
  f := Q.f
  grad := Q.grad
  localGrad := Q.localGrad
  L i := (a i).L
  localL := Q.localL
  L_nonneg i := (a i).L_nonneg
  localL_nonneg := Q.localL_nonneg
  floor := Q.fmin
  lower := Q.lower_bound
  derivative := Q.derivative
  smooth := Q.smooth
  localSmooth := Q.localSmooth
  mean_gradient := Q.mean_gradient

namespace Problem
variable (P : Problem N m n)

theorem local_eq (j : Fin N) (x y : Model m n) (i : Fin p) (h : x i = y i) :
    P.localGrad j x i = P.localGrad j y i := by
  apply sub_eq_zero.mp
  apply norm_eq_zero.mp
  apply le_antisymm _ (norm_nonneg _)
  have hs := (frobenius_le_nuclear _).trans (P.localSmooth j x y i)
  simpa only [h, sub_self, spectralNorm, map_zero, norm_zero, mul_zero] using hs

/-- The largest worker smoothness constant of a block, without an N-dependent
sum bound on the Lipschitz constant. -/
def workerL (i : Fin p) : ℝ :=
  Finset.univ.sup' ⟨⟨0, P.workers_pos⟩, Finset.mem_univ _⟩ (P.localL i)

theorem localL_le (i : Fin p) (j : Fin N) : P.localL i j ≤ P.workerL i :=
  Finset.le_sup' _ (Finset.mem_univ j)

theorem workerL_nonneg (i : Fin p) : 0 ≤ P.workerL i :=
  (P.localL_nonneg i ⟨0, P.workers_pos⟩).trans (P.localL_le i _)

def blockGradients (i : Fin p) : Trajectory.Gradients N (m i) (n i) where
  workers_pos := P.workers_pos
  grad j z := P.localGrad j (Function.update 0 i z) i
  L := P.workerL i
  L_nonneg := P.workerL_nonneg i
  lipschitz j x y := by
    have hs := (frobenius_le_nuclear _).trans
      (P.localSmooth j (Function.update 0 i x) (Function.update 0 i y) i)
    simp only [Function.update_self] at hs
    exact hs.trans ((mul_le_mul_of_nonneg_left (spectralNorm_le_frobenius _)
      (P.localL_nonneg i j)).trans
        (mul_le_mul_of_nonneg_right (P.localL_le i j) (norm_nonneg _)))
  measurable_local j := by
    apply Continuous.measurable
    apply LipschitzWith.continuous (K := ⟨P.workerL i, P.workerL_nonneg i⟩)
    apply LipschitzWith.of_dist_le_mul
    intro x y
    simp only [dist_eq_norm]
    have hs := (frobenius_le_nuclear _).trans
      (P.localSmooth j (Function.update 0 i x) (Function.update 0 i y) i)
    simp only [Function.update_self] at hs
    exact hs.trans ((mul_le_mul_of_nonneg_left (spectralNorm_le_frobenius _)
      (P.localL_nonneg i j)).trans
        (mul_le_mul_of_nonneg_right (P.localL_le i j) (norm_nonneg _)))

theorem block_local (i : Fin p) (j : Fin N) (x : Model m n) :
    (P.blockGradients i).grad j (x i) = P.localGrad j x i :=
  P.local_eq j _ x i (by simp)

theorem block_mean (i : Fin p) (x : Model m n) :
    (P.blockGradients i).mean (x i) = P.grad x i := by
  unfold Trajectory.Gradients.mean
  simp_rw [P.block_local]
  exact P.mean_gradient x i

end Problem

/-- Fresh worker innovations on a common filtration for all blocks.
There is no cross-block independence assumption. -/
structure Innovations (N : ℕ) (m n : Fin p → ℕ) (μ : Measure Ω) where
  filtration : Filtration ℕ ‹MeasurableSpace Ω›
  σ : Fin p → ℝ
  σ_nonneg : ∀ i, 0 ≤ σ i
  value : ℕ → Fin N → Ω → Model m n
  adapted : ∀ t j i, StronglyMeasurable[filtration (t+1)] (fun ω => value t j ω i)
  memLp : ∀ t j i, MemLp (fun ω => value t j ω i) 2 μ
  variance : ∀ t j i, (∫ ω, ‖value t j ω i‖^2 ∂μ) ≤ (σ i)^2
  centered : ∀ t j i, μ[fun ω => value t j ω i | filtration t] =ᵐ[μ] 0
  worker_cross : ∀ t j k i, j ≠ k →
    μ[fun ω => inner ℝ (value t j ω i) (value t k ω i) | filtration t] =ᵐ[μ] 0

def Innovations.block (O : Innovations N m n μ) (i : Fin p) :
    Trajectory.MartingaleInnovations N (m i) (n i) μ where
  filtration := O.filtration
  σ := O.σ i
  σ_nonneg := O.σ_nonneg i
  value t j ω := O.value t j ω i
  adapted t j := O.adapted t j i
  memLp t j := O.memLp t j i
  variance t j := O.variance t j i
  centered t j := O.centered t j i
  worker_cross t j k h := O.worker_cross t j k i h

/-- Certified quality of each block's numerical direction. -/
structure Oracle {m n : ℕ} (c : Trajectory.Controls m n) where
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

variable (P : Problem N m n) (c : ∀ i, Trajectory.Controls (m i) (n i))
  (O : Innovations N m n μ) (R : ∀ i, Trajectory.Roundoff N (m i) (n i) Ω)

/-- Simultaneous block updates using stochastic gradients at the full shared
broadcast model. The local model query is not an isolated block query. -/
def advance (s : States N m n) (t : ℕ) (ω : Ω) : States N m n := fun i =>
  let a := fun j => P.localGrad j (fun k => (s k).w) i+O.value t j ω i
  let M := fun j => (1-(c i).β) • (s i).M j+(c i).β • a j
  let V := fun j => (1-(c i).θ) • M j+(c i).θ • a j
  let G := fun j => (s i).G j+(c i).CU (V j-(s i).G j)+(R i).uplink t j ω
  let X := (s i).x-(c i).η t • (c i).direction t (average G)
  (X, (s i).w+(c i).CP (X-(s i).w)+(R i).downlink t ω, M, G)

def run (x₀ : Model m n) : ℕ → Ω → States N m n
  | 0, _ => fun i => LayerState.start (x₀ i)
  | t+1, ω => advance P c O R (run x₀ t ω) t ω

theorem advance_block (s : States N m n) (t : ℕ) (ω : Ω) (i : Fin p) :
    advance P c O R s t ω i =
      Trajectory.advance (P.blockGradients i) (c i) (O.block i).innovations (R i) (s i) t ω := by
  have hlocal (j : Fin N) : P.localGrad j (fun k => (s k).w) i =
      (P.blockGradients i).grad j (s i).w := (P.block_local i j (fun k => (s k).w)).symm
  unfold advance
  simp_rw [hlocal]
  rfl

theorem run_block (x₀ : Model m n) (t : ℕ) (ω : Ω) (i : Fin p) :
    run P c O R x₀ t ω i =
      Trajectory.run (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t ω := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change advance P c O R (run P c O R x₀ t ω) t ω i = _
    rw [advance_block, ih]
    rfl

theorem run_adapted
    (hU : ∀ i t j, Measurable[O.filtration (t+1)] ((R i).uplink t j))
    (hP : ∀ i t, Measurable[O.filtration (t+1)] ((R i).downlink t))
    (x₀ : Model m n) (t : ℕ) : Measurable[O.filtration t] (run P c O R x₀ t) := by
  let s := run P c O R x₀ t
  have h (i : Fin p) : Measurable[O.filtration t] (fun ω => s ω i) := by
    dsimp only [s]
    simp_rw [run_block]
    exact Trajectory.run_adapted (P.blockGradients i) (c i) (O.block i) (R i)
      (hU i) (hP i) (x₀ i) t
  let : MeasurableSpace Ω := O.filtration t
  exact Measurable.of_eval h

theorem gradient_at_run (x₀ : Model m n) (t : ℕ) (ω : Ω) (i : Fin p) :
    (P.blockGradients i).mean
      (Trajectory.run (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t ω).x =
      P.grad (points (run P c O R x₀ t ω)) i := by
  rw [← run_block]
  exact P.block_mean i (points (run P c O R x₀ t ω))

def errorSample (x₀ : Model m n) (i : Fin p) (t : ℕ) (ω : Ω) : ℝ :=
  ‖Trajectory.estimate (P.blockGradients i) (c i) (O.block i).innovations (R i)
      (run P c O R x₀ t ω i) t ω-P.grad (points (run P c O R x₀ t ω)) i‖

def error (x₀ : Model m n) (i : Fin p) (t : ℕ) : ℝ :=
  Trajectory.estimatorError (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t

def budget (x₀ : Model m n) (i : Fin p) (T : ℕ) : ℝ :=
  Trajectory.trackingBudget (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) T

theorem errorSample_eq (x₀ : Model m n) (i : Fin p) (t : ℕ) (ω : Ω) :
    errorSample P c O R x₀ i t ω =
      ‖Trajectory.estimate (P.blockGradients i) (c i) (O.block i).innovations (R i)
        (Trajectory.run (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t ω) t ω-
          (P.blockGradients i).mean
            (Trajectory.run (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t ω).x‖ := by
  unfold errorSample
  rw [← gradient_at_run, run_block]

theorem error_integrable (x₀ : Model m n) (i : Fin p) (t : ℕ) :
    Integrable (errorSample P c O R x₀ i t) μ := by
  change Integrable (fun ω => errorSample P c O R x₀ i t ω) μ
  simp_rw [errorSample_eq]
  exact Trajectory.estimatorError_integrable (P.blockGradients i) (c i)
    (O.block i).innovations (R i) (x₀ i) t

theorem integral_error (x₀ : Model m n) (i : Fin p) (t : ℕ) :
    (∫ ω, errorSample P c O R x₀ i t ω ∂μ) = error P c O R x₀ i t := by
  simp_rw [errorSample_eq]
  rfl

/-- Each tracking budget is obtained from the simultaneous full-model run;
there is no assumed tracking recurrence or cross-block independence. -/
theorem tracking (x₀ : Model m n) (i : Fin p) (T : ℕ) :
    (∑ t ∈ Finset.range T, (c i).η t*error P c O R x₀ i t) ≤ budget P c O R x₀ i T :=
  Trajectory.tracking (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) T

theorem objective_integrable (x₀ : Model m n) (t : ℕ) :
    Integrable (fun ω => P.f (points (run P c O R x₀ t ω))) μ := by
  let X := fun ω => points (run P c O R x₀ t ω)
  have hX (i : Fin p) : MemLp (fun ω => X ω i) 2 μ := by
    dsimp only [X, points]
    simp_rw [run_block]
    exact (Trajectory.run_memLp (P.blockGradients i) (c i) (O.block i).innovations
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
  have hX := Trajectory.mean_gradient_memLp (P.blockGradients i) (c i)
    (O.block i).innovations (R i) (x₀ i) t
  simp_rw [← gradient_at_run]
  apply ((hX.integrable (by norm_num)).norm.const_mul (d i).ρ).mono'
    (continuous_nuclearNorm.measurable.comp_aemeasurable
      hX.aestronglyMeasurable.aemeasurable).aestronglyMeasurable
  filter_upwards [] with ω
  change ‖nuclearNorm ((P.blockGradients i).mean
    (Trajectory.run (P.blockGradients i) (c i) (O.block i).innovations (R i) (x₀ i) t ω).x)‖ ≤ _
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
  let H := Trajectory.estimate (P.blockGradients i) (c i) (O.block i).innovations (R i)
    (s i) t ω
  let D := (c i).direction t H
  let G := P.grad (points s) i
  have hstep : points (run P c O R x₀ (t+1) ω) i-points s i = -((c i).η t • D) := by
    change (advance P c O R s t ω i).x-(s i).x = _
    rw [advance_block]
    dsimp only [Trajectory.advance, Trajectory.server, LayerState.x]
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
    (x₀ : Model m n) (T : ℕ) (hmass : 0 < mass c d T) :
    (∀ t, Measurable[O.filtration t] (run P c O R x₀ t)) ∧
    (∫ ω, (∑ t ∈ Finset.range T, ∑ i,
      (c i).η t*((d i).gain*nuclearNorm (P.grad (points (run P c O R x₀ t ω)) i)))/
        mass c d T ∂μ) ≤ bound P c O R d x₀ T/mass c d T := by
  exact ⟨run_adapted P c O R hU hP x₀, expected_stationarity P c O R d x₀ T hmass⟩

end EF21Muon.Experiments.BlockTrajectory
