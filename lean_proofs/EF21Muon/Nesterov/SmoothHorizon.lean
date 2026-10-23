import EF21Muon.Nesterov.SmoothRates

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

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
    (hinit : ∀ T, 0 < T → rateInitial (P T) g x ≤ B)
    (hnoise : ∀ T, 0 < T → noiseTotalScale (P T) ≤ S) :
    (fun T => smoothStationarity (P T) (A T) (g / Real.sqrt T) (initial x) T)
      =O[Filter.atTop] (fun T : ℕ => 1 / Real.sqrt T) := by
  apply Asymptotics.IsBigO.of_bound (4 * B / g + 4 * S)
  filter_upwards [Filter.eventually_gt_atTop (0 : ℕ)] with T hT
  have hr := Real.sqrt_pos.mpr (show (0 : ℝ) < T by exact_mod_cast hT)
  have h := smooth_rate (P T) (A T) hg T hT (hβ T hT) (hγ T hT) (hsmall T hT) x
  have hnonneg : 0 ≤ smoothStationarity (P T) (A T) (g / Real.sqrt T) (initial x) T := by
    unfold smoothStationarity
    apply div_nonneg _ (Nat.cast_nonneg _)
    apply integral_nonneg
    intro ξ
    apply List.sum_nonneg
    intro y hy
    obtain ⟨s, _, rfl⟩ := List.mem_map.mp hy
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

def forHorizon (P : Problem N m n μ c) (τ : Fin p → ℝ) (T : ℕ) :
    Problem N m n μ (horizonControls c P.ell2 τ T) :=
  P.retune _ (fun _ => rfl)

/-- The smooth convergence rate for a single fixed objective, oracle, noise law,
initial point and compressor geometry, with the explicit horizon schedule above.
No uniformity of problem-dependent constants is left as a hypothesis. -/
theorem horizon_rate_isBigO (P : Problem N m n μ c) (τ : Fin p → ℝ)
    (A : ∀ T, Scheme N m n (horizonControls c P.ell2 τ T))
    (hτ : ∀ T i, (A T).τ i = τ i) (x : Model m n) :
    (fun T => smoothStationarity (forHorizon P τ T) (A T)
      (rateGamma c P.ell2 τ / Real.sqrt T) (initial x) T)
      =O[Filter.atTop] (fun T : ℕ => 1 / Real.sqrt T) := by
  apply smooth_rate_isBigO _ (forHorizon P τ) A
    (rateGamma_pos c P.ell2 τ) x
    (B := rateInitial P (rateGamma c P.ell2 τ) x) (S := noiseTotalScale P)
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

/-- The Nesterov extension of Theorem 5(i): O(T^(-1/2)) for EF21-MuonUSign (compressed=false) and
EF21-MuonSign (compressed=true), with a fixed stochastic optimization problem
and fully explicit horizon-dependent momentum and common sharp step.
The exact optimization of the paper's momentum formula is not needed. -/
theorem matrix_horizon_rate_isBigO (compressed : Bool) {β γ L : Fin p → ℝ}
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (hβ : ∀ i, 0 < β i)
    (hβ1 : ∀ i, β i ≤ 1) (hγ : ∀ i, 0 < γ i) (hL : ∀ i, 0 ≤ L i)
    (P : Problem N m n μ (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL))
    (x : Model m n) :
    (fun T => smoothStationarity (forHorizon P (matrixRateTau compressed m n) T)
      (matrixHorizonScheme compressed hm hn hβ hβ1 hγ hL P.ell2 T)
      (rateGamma (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL)
        P.ell2 (matrixRateTau compressed m n) / Real.sqrt T) (initial x) T)
      =O[Filter.atTop] (fun T : ℕ => 1 / Real.sqrt T) := by
  exact horizon_rate_isBigO P _ _ (fun _ _ => rfl) x

end EF21Muon.Nesterov
