import EF21Muon.Distributed.Methods
import Mathlib.Analysis.Asymptotics.Defs

/-! Horizon-dependent smooth convergence. All constants in the envelope below
are independent of momentum and the horizon. Choosing beta = 1 / sqrt(T) and
gamma = g * beta gives the rate in Theorem 5(i) and Corollary 1. -/

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

/-- An envelope for beta^2 * zeta, independent of beta and gamma. -/
def rateEnvelope (c : Parameters) (ell2 τ : ℝ) : ℝ :=
  c.ρ ^ 2 * (12 * c.L ^ 2 + 72 * τ * c.L ^ 2 / c.αP ^ 2 +
    180 * ell2 / c.αU ^ 2 + 1008 * τ * ell2 / (c.αP ^ 2 * c.αU ^ 2))

theorem rateEnvelope_nonneg (c : Parameters) {ell2 τ : ℝ}
    (hell : 0 ≤ ell2) (hτ : 0 ≤ τ) : 0 ≤ rateEnvelope c ell2 τ := by
  unfold rateEnvelope
  positivity

theorem zeta_le_rateEnvelope (c : Parameters) {ell2 τ : ℝ}
    (hell : 0 ≤ ell2) (hτ : 0 ≤ τ) :
    zeta c ell2 τ ≤ rateEnvelope c ell2 τ / c.β ^ 2 := by
  have hb := c.β_pos
  have hb1 := c.β_le_one
  have hb2 : c.β ^ 2 ≤ 1 := by nlinarith
  have hL := sq_nonneg c.L
  have hU : 0 < c.αU ^ 2 := sq_pos_of_pos c.αU_pos
  have hP : 0 < c.αP ^ 2 := sq_pos_of_pos c.αP_pos
  have h1 : 24 * τ * (c.β + 2) * c.L ^ 2 ≤ 72 * τ * c.L ^ 2 := by
    nlinarith [mul_nonneg hτ hL]
  have h2 : 36 * (c.β ^ 2 + 4) * ell2 ≤ 180 * ell2 := by
    nlinarith
  have h3 : 144 * c.β ^ 2 * (2 * c.β + 5) * ell2 ≤ 1008 * ell2 := by
    have h := mul_le_mul hb2 (show 2 * c.β + 5 ≤ 7 by linarith)
      (by positivity) (by norm_num : (0 : ℝ) ≤ 1)
    nlinarith
  have hbound : zeta c ell2 τ ≤
      c.ρ ^ 2 * (12 * c.L ^ 2 / c.β ^ 2 +
        72 * τ * c.L ^ 2 / c.αP ^ 2 + 180 * ell2 / c.αU ^ 2 +
          1008 * τ * ell2 / (c.αP ^ 2 * c.αU ^ 2)) := by
    rw [zeta_eq_paper]
    apply mul_le_mul_of_nonneg_left _ (sq_nonneg _)
    apply add_le_add
    · exact add_le_add (add_le_add le_rfl (div_le_div_of_nonneg_right h1 hP.le))
        (div_le_div_of_nonneg_right h2 hU.le)
    · exact div_le_div_of_nonneg_right
        (by nlinarith [mul_le_mul_of_nonneg_left h3 hτ]) (mul_pos hP hU).le
  apply hbound.trans
  rw [le_div_iff₀ (sq_pos_of_pos hb)]
  have hrest : 0 ≤ 72 * τ * c.L ^ 2 / c.αP ^ 2 +
      180 * ell2 / c.αU ^ 2 + 1008 * τ * ell2 / (c.αP ^ 2 * c.αU ^ 2) := by
    positivity
  have hmul := mul_le_mul_of_nonneg_left hb2 hrest
  have he : (12 * c.L ^ 2 / c.β ^ 2 +
      72 * τ * c.L ^ 2 / c.αP ^ 2 + 180 * ell2 / c.αU ^ 2 +
      1008 * τ * ell2 / (c.αP ^ 2 * c.αU ^ 2)) * c.β ^ 2 =
      12 * c.L ^ 2 + (72 * τ * c.L ^ 2 / c.αP ^ 2 +
        180 * ell2 / c.αU ^ 2 + 1008 * τ * ell2 / (c.αP ^ 2 * c.αU ^ 2)) * c.β ^ 2 := by
    field_simp
    ring
  rw [mul_assoc, he]
  unfold rateEnvelope
  nlinarith [mul_le_mul_of_nonneg_left hmul (sq_nonneg c.ρ)]

/-- A beta-proportional step satisfies the paper's square-root threshold. -/
theorem rate_step_condition (c : Parameters) {ell2 τ g : ℝ}
    (hell : 0 ≤ ell2) (hτ : 0 ≤ τ) (hg : 0 ≤ g)
    (hγ : c.γ = g * c.β)
    (hsmall : g * (2 * c.L + 2 * Real.sqrt (rateEnvelope c ell2 τ)) ≤ 1) :
    c.γ * (2 * c.L + 2 * Real.sqrt (zeta c ell2 τ)) ≤ 1 := by
  have hb := c.β_pos
  have hZ := rateEnvelope_nonneg c hell hτ
  have hz := zeta_nonneg c hell hτ
  have hs : c.β * Real.sqrt (zeta c ell2 τ) ≤ Real.sqrt (rateEnvelope c ell2 τ) := by
    have h := (le_div_iff₀ (sq_pos_of_pos hb)).mp (zeta_le_rateEnvelope c hell hτ)
    have hsq := Real.sq_sqrt hz
    have hsqZ := Real.sq_sqrt hZ
    nlinarith [Real.sqrt_nonneg (zeta c ell2 τ), Real.sqrt_nonneg (rateEnvelope c ell2 τ)]
  rw [hγ]
  have hL := mul_le_mul_of_nonneg_left c.β_le_one c.L_nonneg
  nlinarith [mul_le_mul_of_nonneg_left hs hg, mul_le_mul_of_nonneg_left hL hg]

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

/-- A horizon-independent upper bound for the zero-initialized potential when
gamma = g * beta. It includes the paper's initial gradient errors. -/
def Problem.rateInitial (P : Problem N m n μ c) (g : ℝ) (x : Model m n) : ℝ :=
  P.f x - P.fmin + g * ∑ i,
    (6 * (c i).ρ ^ 2 * ‖P.grad x i‖ ^ 2 +
      72 * (c i).ρ ^ 2 / (c i).αU ^ 2 *
        average (fun j => ‖P.localGrad j x i‖ ^ 2))

omit [IsProbabilityMeasure μ] in
theorem Problem.initial_potential_le_rateInitial (P : Problem N m n μ c)
    {g : ℝ} (hg : 0 ≤ g) (hγ : ∀ i, (c i).γ = g * (c i).β) (x : Model m n) :
    P.potential (start x) ≤ P.rateInitial g x := by
  rw [P.initial_potential]
  unfold Problem.rateInitial
  rw [Finset.mul_sum]
  apply add_le_add le_rfl
  apply Finset.sum_le_sum
  intro i _
  have hb := (c i).β_pos
  have hb2 : (c i).β ^ 2 ≤ 1 := by nlinarith [(c i).β_le_one]
  have hlocal := average_nonneg (fun j : Fin N => sq_nonneg ‖P.localGrad j x i‖)
  have hcoef : 0 ≤ 72 * (c i).ρ ^ 2 / (c i).αU ^ 2 *
      average (fun j => ‖P.localGrad j x i‖ ^ 2) := by positivity
  have h := mul_le_mul_of_nonneg_left hb2 hcoef
  rw [hγ i]
  unfold globalWeight localWeight
  have he : g * (c i).β *
      (6 * (c i).ρ ^ 2 / (c i).β * ‖P.grad x i‖ ^ 2 +
        72 * (c i).ρ ^ 2 * (c i).β / (c i).αU ^ 2 *
          average (fun j => ‖P.localGrad j x i‖ ^ 2)) =
      g * (6 * (c i).ρ ^ 2 * ‖P.grad x i‖ ^ 2 +
        (72 * (c i).ρ ^ 2 / (c i).αU ^ 2 *
          average (fun j => ‖P.localGrad j x i‖ ^ 2)) * (c i).β ^ 2) := by
    field_simp
  rw [he]
  exact mul_le_mul_of_nonneg_left (add_le_add le_rfl (by simpa only [mul_one] using h)) hg

/-- The smooth O(T^(-1/2)) rate, with an explicit constant, on the actual
stochastic multiworker/multilayer paths. Both EF21 methods instantiate `A`.
The choices beta = 1/sqrt(T), gamma = g/sqrt(T) meet the paper's step restriction. -/
theorem Problem.smooth_rate (P : Problem N m n μ c) (A : Scheme N m n c)
    {g : ℝ} (hg : 0 < g) (T : ℕ) (hT : 0 < T)
    (hβ : ∀ i, (c i).β = 1 / Real.sqrt T)
    (hγ : ∀ i, (c i).γ = g * (c i).β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) :
    P.stationarity A (g / Real.sqrt T) (start x) T ≤
      (4 * P.rateInitial g x / g + 4 * P.noiseScaleTotal) / Real.sqrt T := by
  have hTr : (0 : ℝ) < T := by exact_mod_cast hT
  have hr := Real.sqrt_pos.mpr hTr
  have hrsq := Real.sq_sqrt hTr.le
  have hsteps (i : Fin p) : (c i).γ = g / Real.sqrt T := by rw [hγ i, hβ i]; ring
  have hstep (i : Fin p) := (c i).step_condition_for
    (zeta_nonneg _ (P.ell2_nonneg i) (A.τ_nonneg i))
    (rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i) hg.le (hγ i) (hsmall i))
  have hbound := P.average_stationarity A hstep (div_pos hg hr) (start x) T hT
  have hinit := P.initial_potential_le_rateInitial hg.le hγ x
  have hnoise : P.noiseTotal ≤ (g / Real.sqrt T) *
      ((1 / Real.sqrt T) * P.noiseScaleTotal) := by
    unfold Problem.noiseTotal Problem.noiseScaleTotal
    simp only [hsteps, mul_assoc, ← Finset.mul_sum]
    apply mul_le_mul_of_nonneg_left _ (div_pos hg hr).le
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun i _ => by
      simpa only [hβ i, mul_assoc] using
        mul_le_mul_of_nonneg_right (noise_le (c i) N) (sq_nonneg (P.σ i)))
  have hi := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hinit (by norm_num : (0 : ℝ) ≤ 4))
    (mul_pos (div_pos hg hr) hTr).le
  have hn := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hnoise (by norm_num : (0 : ℝ) ≤ 4))
    (div_pos hg hr).le
  apply hbound.trans
  apply (add_le_add hi hn).trans
  apply le_of_eq
  field_simp
  rw [hrsq]
  ring

/-- The common proportionality constant can be chosen explicitly, independently
of beta. This removes any implicit fixed-point choice of beta and gamma. -/
theorem rate_common_step {p : ℕ} (c : Fin p → Parameters) (ell2 τ : Fin p → ℝ) :
    let g := safeGamma (fun i => 2 * (c i).L +
      2 * Real.sqrt (rateEnvelope (c i) (ell2 i) (τ i)))
    0 < g ∧ ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (rateEnvelope (c i) (ell2 i) (τ i))) ≤ 1 := by
  have hK (i : Fin p) : 0 ≤ 2 * (c i).L +
      2 * Real.sqrt (rateEnvelope (c i) (ell2 i) (τ i)) := by
    have := (c i).L_nonneg
    positivity
  exact ⟨safeGamma_pos hK, safeGamma_valid hK⟩

/-- Horizon-uniform data give a genuine Big-O statement for the family of
stochastic runs. The two uniform bounds concern only the objective, initial
gradients, variances and fixed geometry/compression constants, not the rate. -/
theorem smooth_rate_isBigO
    (controls : ℕ → Fin p → Parameters)
    (P : ∀ T, Problem N m n μ (controls T))
    (A : ∀ T, Scheme N m n (controls T))
    {g B S : ℝ} (hg : 0 < g) (x : Model m n)
    (hβ : ∀ T, 0 < T → ∀ i, (controls T i).β = 1 / Real.sqrt T)
    (hγ : ∀ T, 0 < T → ∀ i, (controls T i).γ = g * (controls T i).β)
    (hsmall : ∀ T, 0 < T → ∀ i, g * (2 * (controls T i).L +
      2 * Real.sqrt (rateEnvelope (controls T i) ((P T).ell2 i) ((A T).τ i))) ≤ 1)
    (hinit : ∀ T, 0 < T → (P T).rateInitial g x ≤ B)
    (hnoise : ∀ T, 0 < T → (P T).noiseScaleTotal ≤ S) :
    (fun T => (P T).stationarity (A T) (g / Real.sqrt T) (start x) T)
      =O[Filter.atTop] (fun T : ℕ => 1 / Real.sqrt T) := by
  apply Asymptotics.IsBigO.of_bound (4 * B / g + 4 * S)
  filter_upwards [Filter.eventually_gt_atTop (0 : ℕ)] with T hT
  have hr := Real.sqrt_pos.mpr (show (0 : ℝ) < T by exact_mod_cast hT)
  have h := (P T).smooth_rate (A T) hg T hT (hβ T hT) (hγ T hT) (hsmall T hT) x
  have hnonneg : 0 ≤ (P T).stationarity (A T) (g / Real.sqrt T) (start x) T := by
    unfold Problem.stationarity
    apply div_nonneg _ (Nat.cast_nonneg _)
    apply integral_nonneg
    intro ξ
    apply List.sum_nonneg
    intro y hy
    obtain ⟨s, _, rfl⟩ := List.mem_map.mp hy
    rw [(P T).weightedCost_eq]
    exact Finset.sum_nonneg (fun i _ => mul_nonneg
      (div_nonneg (controls T i).γ_pos.le (div_pos hg hr).le) (sq_nonneg _))
  rw [Real.norm_eq_abs, abs_of_nonneg hnonneg, Real.norm_eq_abs,
    abs_of_nonneg (one_div_pos.mpr hr).le]
  have hb := div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_left (hinit T hT) (by norm_num : (0 : ℝ) ≤ 4)) hg.le
  have hs := mul_le_mul_of_nonneg_left (hnoise T hT) (by norm_num : (0 : ℝ) ≤ 4)
  exact h.trans (by
    simpa only [div_eq_mul_inv, mul_one, one_mul] using
      div_le_div_of_nonneg_right (add_le_add hb hs) hr.le)

/-- Retuning momentum and the step does not change the stochastic optimization
problem. Only equality of the smoothness constants is required. -/
def Problem.retune (P : Problem N m n μ c) (d : Fin p → Parameters)
    (hL : ∀ i, (d i).L = (c i).L) : Problem N m n μ d where
  workers_pos := P.workers_pos
  f := P.f
  grad := P.grad
  localGrad := P.localGrad
  localL := P.localL
  localL_nonneg := P.localL_nonneg
  fmin := P.fmin
  lower_bound := P.lower_bound
  derivative := P.derivative
  smooth x y i := by rw [hL i]; exact P.smooth x y i
  localSmooth := P.localSmooth
  mean_gradient := P.mean_gradient
  σ := P.σ
  reply := P.reply
  measurable_reply := P.measurable_reply
  noise_memLp := P.noise_memLp
  noise_centered := P.noise_centered
  noise_variance := P.noise_variance
  worker_independence := P.worker_independence

/-- Defined even at horizon zero, using max(1,T); positive horizons use 1/sqrt(T). -/
def horizonBeta (T : ℕ) : ℝ := 1 / Real.sqrt (max 1 T : ℕ)

theorem horizonBeta_pos (T : ℕ) : 0 < horizonBeta T := by
  apply one_div_pos.mpr
  apply Real.sqrt_pos.mpr
  exact_mod_cast (show 0 < max 1 T from lt_of_lt_of_le (by norm_num) (le_max_left _ _))

theorem horizonBeta_le_one (T : ℕ) : horizonBeta T ≤ 1 := by
  have h : (1 : ℝ) ≤ (max 1 T : ℕ) := by exact_mod_cast (le_max_left 1 T)
  have hs := Real.sq_sqrt (show (0 : ℝ) ≤ (max 1 T : ℕ) by positivity)
  have hr : 1 ≤ Real.sqrt (max 1 T : ℕ) := by
    nlinarith [Real.sqrt_nonneg (max 1 T : ℕ)]
  exact (div_le_one (Real.sqrt_pos.mpr (by positivity))).mpr hr

def rateGamma (base : Fin p → Parameters) (ell2 τ : Fin p → ℝ) : ℝ :=
  safeGamma (fun i => 2 * (base i).L + 2 * Real.sqrt (rateEnvelope (base i) (ell2 i) (τ i)))

theorem rateGamma_pos (base : Fin p → Parameters) (ell2 τ : Fin p → ℝ) :
    0 < rateGamma base ell2 τ := (rate_common_step base ell2 τ).1

/-- Fully explicit horizon-dependent controls. Geometry, smoothness and
compression parameters are inherited unchanged from base. -/
def horizonControls (base : Fin p → Parameters) (ell2 τ : Fin p → ℝ)
    (T : ℕ) : Fin p → Parameters :=
  fun i => { base i with
    β := horizonBeta T
    β_pos := horizonBeta_pos T
    β_le_one := horizonBeta_le_one T
    γ := rateGamma base ell2 τ * horizonBeta T
    γ_pos := mul_pos (rateGamma_pos base ell2 τ) (horizonBeta_pos T) }

def Problem.forHorizon (P : Problem N m n μ c) (τ : Fin p → ℝ) (T : ℕ) :
    Problem N m n μ (horizonControls c P.ell2 τ T) :=
  P.retune _ (fun _ => rfl)

/-- The smooth convergence rate for a single fixed objective, oracle, noise law,
initial point and compressor geometry, with the explicit horizon schedule above.
No uniformity of problem-dependent constants is left as a hypothesis. -/
theorem Problem.horizon_rate_isBigO (P : Problem N m n μ c) (τ : Fin p → ℝ)
    (A : ∀ T, Scheme N m n (horizonControls c P.ell2 τ T))
    (hτ : ∀ T i, (A T).τ i = τ i) (x : Model m n) :
    (fun T => (P.forHorizon τ T).stationarity (A T)
      (rateGamma c P.ell2 τ / Real.sqrt T) (start x) T)
      =O[Filter.atTop] (fun T : ℕ => 1 / Real.sqrt T) := by
  apply smooth_rate_isBigO _ (P.forHorizon τ) A
    (rateGamma_pos c P.ell2 τ) x
    (B := P.rateInitial (rateGamma c P.ell2 τ) x) (S := P.noiseScaleTotal)
  · intro T hT i
    change horizonBeta T = _
    rw [horizonBeta, max_eq_right hT]
  · intro T _ i
    rfl
  · intro T _ i
    rw [hτ T i]
    exact (rate_common_step c P.ell2 τ).2 i
  · intro T _
    exact le_rfl
  · intro T _
    exact le_rfl

/-- Downlink norm factor for the concrete MuonUSign/MuonSign methods. -/
def matrixRateTau (compressed : Bool) (m n : Fin p → ℕ) (i : Fin p) : ℝ :=
  if compressed then (Real.sqrt ((min (m i) (n i) : ℕ) : ℝ)) ^ 2 else 0

/-- Concrete scaled-sign uplink and exact/scaled-sign downlink, with the
horizon-dependent parameters used in the smooth convergence theorem. -/
def matrixHorizonScheme (compressed : Bool) {β γ L : Fin p → ℝ}
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (hβ : ∀ i, 0 < β i)
    (hβ1 : ∀ i, β i ≤ 1) (hγ : ∀ i, 0 < γ i) (hL : ∀ i, 0 ≤ L i)
    (ell2 : Fin p → ℝ) (T : ℕ) :
    Scheme N m n (horizonControls
      (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL)
      ell2 (matrixRateTau compressed m n) T) :=
  matrixScheme compressed (fun _ => horizonBeta T)
    (fun _ => rateGamma (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL)
      ell2 (matrixRateTau compressed m n) * horizonBeta T) L hm hn
    (fun _ => horizonBeta_pos T) (fun _ => horizonBeta_le_one T)
    (fun _ => mul_pos (rateGamma_pos _ _ _) (horizonBeta_pos T)) hL

/-- Theorem 5(i): O(T^(-1/2)) for EF21-MuonUSign (compressed=false) and
EF21-MuonSign (compressed=true), with a fixed stochastic optimization problem
and fully explicit horizon-dependent momentum and common sharp step.
The exact optimization of the paper's momentum formula is not needed. -/
theorem matrix_horizon_rate_isBigO (compressed : Bool) {β γ L : Fin p → ℝ}
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (hβ : ∀ i, 0 < β i)
    (hβ1 : ∀ i, β i ≤ 1) (hγ : ∀ i, 0 < γ i) (hL : ∀ i, 0 ≤ L i)
    (P : Problem N m n μ (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL))
    (x : Model m n) :
    (fun T => (P.forHorizon (matrixRateTau compressed m n) T).stationarity
      (matrixHorizonScheme compressed hm hn hβ hβ1 hγ hL P.ell2 T)
      (rateGamma (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL)
        P.ell2 (matrixRateTau compressed m n) / Real.sqrt T) (start x) T)
      =O[Filter.atTop] (fun T : ℕ => 1 / Real.sqrt T) := by
  exact P.horizon_rate_isBigO _ _ (fun _ _ => rfl) x

end EF21Muon.Distributed
