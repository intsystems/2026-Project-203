import EF21Muon.Distributed.Rates

/-! Unequal, fixed layer coefficients with horizon-dependent sharp steps.
The weights are independent of the horizon; the objective and noise law are
transported unchanged. This proves the weighted version of Corollary 1. -/

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def Problem.weightedRateInitial (P : Problem N m n μ c)
    (g : Fin p → ℝ) (x : Model m n) : ℝ :=
  P.f x - P.fmin + ∑ i, g i *
    (6 * (c i).ρ ^ 2 * ‖P.grad x i‖ ^ 2 +
      72 * (c i).ρ ^ 2 / (c i).αU ^ 2 *
        average (fun j => ‖P.localGrad j x i‖ ^ 2))

def Problem.weightedNoiseScale (P : Problem N m n μ c) (g : Fin p → ℝ) : ℝ :=
  ∑ i, g i * noiseScale (c i) N * P.σ i ^ 2

omit [IsProbabilityMeasure μ] in
theorem Problem.initial_potential_le_weightedRateInitial (P : Problem N m n μ c)
    {g : Fin p → ℝ} (hg : ∀ i, 0 ≤ g i)
    (hγ : ∀ i, (c i).γ = g i * (c i).β) (x : Model m n) :
    P.potential (start x) ≤ P.weightedRateInitial g x := by
  rw [P.initial_potential]
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
  have he : g i * (c i).β *
      (6 * (c i).ρ ^ 2 / (c i).β * ‖P.grad x i‖ ^ 2 +
        72 * (c i).ρ ^ 2 * (c i).β / (c i).αU ^ 2 *
          average (fun j => ‖P.localGrad j x i‖ ^ 2)) =
      g i * (6 * (c i).ρ ^ 2 * ‖P.grad x i‖ ^ 2 +
        (72 * (c i).ρ ^ 2 / (c i).αU ^ 2 *
          average (fun j => ‖P.localGrad j x i‖ ^ 2)) * (c i).β ^ 2) := by
    field_simp
  rw [he]
  exact mul_le_mul_of_nonneg_left (add_le_add le_rfl (by simpa using h)) (hg i)

/-- Explicit finite-horizon bound with fixed layer weights g_i / gbar.
Taking gbar to be the arithmetic mean gives exactly the paper's weights. -/
theorem Problem.weighted_smooth_rate (P : Problem N m n μ c) (A : Scheme N m n c)
    {g : Fin p → ℝ} {gbar : ℝ} (hg : ∀ i, 0 < g i) (hbar : 0 < gbar)
    (T : ℕ) (hT : 0 < T)
    (hβ : ∀ i, (c i).β = 1 / Real.sqrt T)
    (hγ : ∀ i, (c i).γ = g i * (c i).β)
    (hsmall : ∀ i, g i * (2 * (c i).L +
      2 * Real.sqrt (rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) :
    P.stationarity A (gbar / Real.sqrt T) (start x) T ≤
      (4 * P.weightedRateInitial g x / gbar +
        4 * P.weightedNoiseScale g / gbar) / Real.sqrt T := by
  have hTr : (0 : ℝ) < T := by exact_mod_cast hT
  have hr := Real.sqrt_pos.mpr hTr
  have hrsq := Real.sq_sqrt hTr.le
  have hstep (i : Fin p) := (c i).step_condition_for
    (zeta_nonneg _ (P.ell2_nonneg i) (A.τ_nonneg i))
    (rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i)
      (hg i).le (hγ i) (hsmall i))
  have hbound := P.average_stationarity A hstep (div_pos hbar hr) (start x) T hT
  have hinit := P.initial_potential_le_weightedRateInitial (fun i => (hg i).le) hγ x
  have hnoise : P.noiseTotal ≤ (1 / Real.sqrt T) ^ 2 * P.weightedNoiseScale g := by
    unfold Problem.noiseTotal Problem.weightedNoiseScale
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    have h := mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_right (noise_le (c i) N) (sq_nonneg (P.σ i)))
      (c i).γ_pos.le
    rw [hγ i, hβ i] at h ⊢
    convert h using 1 <;> ring
  have hi := div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_left hinit (by norm_num : (0 : ℝ) ≤ 4))
    (mul_pos (div_pos hbar hr) hTr).le
  have hn := div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_left hnoise (by norm_num : (0 : ℝ) ≤ 4))
    (div_pos hbar hr).le
  apply hbound.trans
  apply (add_le_add hi hn).trans
  apply le_of_eq
  field_simp
  rw [hrsq]
  ring

/-- Independent layer tuning: any positive g_i below this bound works. -/
def layerRateGamma (base : Fin p → Parameters) (ell2 τ : Fin p → ℝ)
    (i : Fin p) : ℝ :=
  1 / (1 + 2 * (base i).L + 2 * Real.sqrt (rateEnvelope (base i) (ell2 i) (τ i)))

theorem layerRateGamma_pos (base : Fin p → Parameters) (ell2 τ : Fin p → ℝ) (i : Fin p) :
    0 < layerRateGamma base ell2 τ i := by
  have := (base i).L_nonneg
  unfold layerRateGamma
  positivity

theorem layerRateGamma_valid (base : Fin p → Parameters) (ell2 τ : Fin p → ℝ) (i : Fin p) :
    layerRateGamma base ell2 τ i *
      (2 * (base i).L + 2 * Real.sqrt (rateEnvelope (base i) (ell2 i) (τ i))) ≤ 1 := by
  have h : 0 ≤ 2 * (base i).L +
      2 * Real.sqrt (rateEnvelope (base i) (ell2 i) (τ i)) := by
    have := (base i).L_nonneg
    positivity
  unfold layerRateGamma
  rw [one_div_mul_eq_div, div_le_one (by linarith)]
  linarith

def layerHorizonControls (base : Fin p → Parameters) (g : Fin p → ℝ)
    (hg : ∀ i, 0 < g i) (T : ℕ) : Fin p → Parameters :=
  fun i => { base i with
    β := horizonBeta T
    β_pos := horizonBeta_pos T
    β_le_one := horizonBeta_le_one T
    γ := g i * horizonBeta T
    γ_pos := mul_pos (hg i) (horizonBeta_pos T) }

def Problem.forLayerHorizon (P : Problem N m n μ c) (g : Fin p → ℝ)
    (hg : ∀ i, 0 < g i) (T : ℕ) : Problem N m n μ (layerHorizonControls c g hg T) :=
  P.retune _ (fun _ => rfl)

omit [IsProbabilityMeasure μ] in
theorem Problem.layer_horizon_cost (P : Problem N m n μ c)
    (g : Fin p → ℝ) (hg : ∀ i, 0 < g i) {gbar : ℝ} (hbar : 0 < gbar)
    (T : ℕ) (hT : 0 < T) (s : States N m n) :
    (P.forLayerHorizon g hg T).weightedCost (gbar / Real.sqrt T) s =
      ∑ i, (g i / gbar) * nuclearNorm (P.grad (points s) i) ^ 2 := by
  rw [Problem.weightedCost_eq]
  apply Finset.sum_congr rfl
  intro i _
  change (g i * horizonBeta T / (gbar / Real.sqrt T)) * _ = _
  rw [horizonBeta, max_eq_right hT]
  have hr := Real.sqrt_pos.mpr (show (0 : ℝ) < T by exact_mod_cast hT)
  field_simp
  rfl

/-- The weighted asymptotic rate for one fixed stochastic problem. No uniform
data assumptions are needed: only beta and gamma vary with the horizon. -/
theorem Problem.weighted_horizon_rate_isBigO (P : Problem N m n μ c)
    (g : Fin p → ℝ) (hg : ∀ i, 0 < g i) {gbar : ℝ} (hbar : 0 < gbar)
    (τ : Fin p → ℝ)
    (A : ∀ T, Scheme N m n (layerHorizonControls c g hg T))
    (hτ : ∀ T i, (A T).τ i = τ i)
    (hsmall : ∀ i, g i * (2 * (c i).L +
      2 * Real.sqrt (rateEnvelope (c i) (P.ell2 i) (τ i))) ≤ 1)
    (x : Model m n) :
    (fun T => (P.forLayerHorizon g hg T).stationarity (A T)
      (gbar / Real.sqrt T) (start x) T)
      =O[Filter.atTop] (fun T : ℕ => 1 / Real.sqrt T) := by
  apply Asymptotics.IsBigO.of_bound
    (4 * P.weightedRateInitial g x / gbar + 4 * P.weightedNoiseScale g / gbar)
  filter_upwards [Filter.eventually_gt_atTop (0 : ℕ)] with T hT
  have hr := Real.sqrt_pos.mpr (show (0 : ℝ) < T by exact_mod_cast hT)
  have h := (P.forLayerHorizon g hg T).weighted_smooth_rate (A T) hg hbar T hT
    (fun _ => by change horizonBeta T = _; rw [horizonBeta, max_eq_right hT])
    (fun _ => rfl) (fun i => by rw [hτ T i]; exact hsmall i) x
  have hnonneg : 0 ≤ (P.forLayerHorizon g hg T).stationarity (A T)
      (gbar / Real.sqrt T) (start x) T := by
    unfold Problem.stationarity
    apply div_nonneg _ (Nat.cast_nonneg _)
    apply integral_nonneg
    intro ξ
    apply List.sum_nonneg
    intro y hy
    obtain ⟨s, _, rfl⟩ := List.mem_map.mp hy
    rw [P.layer_horizon_cost g hg hbar T hT]
    exact Finset.sum_nonneg (fun i _ => mul_nonneg
      (div_nonneg (hg i).le hbar.le) (sq_nonneg _))
  rw [Real.norm_eq_abs, abs_of_nonneg hnonneg, Real.norm_eq_abs,
    abs_of_nonneg (one_div_pos.mpr hr).le]
  change (P.forLayerHorizon g hg T).stationarity (A T)
    (gbar / Real.sqrt T) (start x) T ≤
      (4 * P.weightedRateInitial g x / gbar +
        4 * P.weightedNoiseScale g / gbar) / Real.sqrt T at h
  simpa only [div_eq_mul_inv, mul_one, one_mul] using h

theorem layer_average_pos (hp : 0 < p) {g : Fin p → ℝ} (hg : ∀ i, 0 < g i) :
    0 < average g := by
  unfold average
  simp only [smul_eq_mul]
  apply mul_pos
  · exact inv_pos.mpr (by exact_mod_cast hp)
  · exact Finset.sum_pos (fun i _ => hg i) (Finset.univ_nonempty_iff.mpr
      ⟨⟨0, hp⟩⟩)

def matrixLayerHorizonScheme (compressed : Bool) {β γ L : Fin p → ℝ}
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (hβ : ∀ i, 0 < β i)
    (hβ1 : ∀ i, β i ≤ 1) (hγ : ∀ i, 0 < γ i) (hL : ∀ i, 0 ≤ L i)
    (g : Fin p → ℝ) (hg : ∀ i, 0 < g i) (T : ℕ) :
    Scheme N m n (layerHorizonControls
      (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL) g hg T) :=
  matrixScheme compressed (fun _ => horizonBeta T)
    (fun i => g i * horizonBeta T) L hm hn
    (fun _ => horizonBeta_pos T) (fun _ => horizonBeta_le_one T)
    (fun i => mul_pos (hg i) (horizonBeta_pos T)) hL

/-- Corollary 1 for the actual two matrix compressors, with unequal layer
coefficients and exactly the paper's arithmetic-mean normalization. -/
theorem matrix_weighted_horizon_rate_isBigO (compressed : Bool)
    {β γ L : Fin p → ℝ} (hp : 0 < p)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (hβ : ∀ i, 0 < β i)
    (hβ1 : ∀ i, β i ≤ 1) (hγ : ∀ i, 0 < γ i) (hL : ∀ i, 0 ≤ L i)
    (P : Problem N m n μ (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL))
    (g : Fin p → ℝ) (hg : ∀ i, 0 < g i)
    (hsmall : ∀ i, g i * (2 * L i + 2 * Real.sqrt
      (rateEnvelope (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL i)
        (P.ell2 i) (matrixRateTau compressed m n i))) ≤ 1)
    (x : Model m n) :
    (fun T => (P.forLayerHorizon g hg T).stationarity
      (matrixLayerHorizonScheme compressed hm hn hβ hβ1 hγ hL g hg T)
      (average g / Real.sqrt T) (start x) T)
      =O[Filter.atTop] (fun T : ℕ => 1 / Real.sqrt T) := by
  exact P.weighted_horizon_rate_isBigO g hg (layer_average_pos hp hg)
    (matrixRateTau compressed m n) _ (fun _ _ => rfl) hsmall x

end EF21Muon.Distributed
