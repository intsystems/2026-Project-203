import EF21Muon.Distributed.Bounds

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

def noiseScale (c : Parameters) (N : ℕ) : ℝ :=
  6 * c.ρ ^ 2 * ((N : ℝ)⁻¹ + (1 - c.αU) / c.αU + 12 / c.αU ^ 2)

theorem noiseScale_nonneg (c : Parameters) (N : ℕ) : 0 ≤ noiseScale c N := by
  have hU := c.αU_pos
  have hU1 := sub_nonneg.mpr c.αU_le_one
  unfold noiseScale
  positivity

theorem noise_le (c : Parameters) (N : ℕ) : noise c N ≤ c.β * noiseScale c N := by
  have hβ := c.β_pos
  have hU := c.αU_pos
  have hβ1 := c.β_le_one
  have h1 : (1 - c.αU) * c.β / c.αU ≤ (1 - c.αU) / c.αU := by
    exact div_le_div_of_nonneg_right (mul_le_of_le_one_right (sub_nonneg.mpr c.αU_le_one) hβ1) hU.le
  have hsq : c.β ^ 2 ≤ 1 := by nlinarith only [hβ, hβ1]
  have h2 : 12 * c.β ^ 2 / c.αU ^ 2 ≤ 12 / c.αU ^ 2 := by
    exact div_le_div_of_nonneg_right (by linarith only [hsq]) (sq_nonneg _)
  have h := mul_le_mul_of_nonneg_left (add_le_add_left (add_le_add h1 h2) ((N : ℝ)⁻¹))
    (by positivity : 0 ≤ 6 * c.ρ ^ 2 * c.β)
  unfold noise noiseScale
  nlinarith only [h]

def tunedBeta (S ε : ℝ) : ℝ := min 1 (ε ^ 2 / (8 * (S + 1)))

theorem tunedBeta_pos {S ε : ℝ} (hS : 0 ≤ S) (hε : 0 < ε) : 0 < tunedBeta S ε :=
  lt_min (by norm_num) (div_pos (sq_pos_of_pos hε) (by positivity))

theorem tunedBeta_le_one (S ε : ℝ) : tunedBeta S ε ≤ 1 := min_le_left _ _

theorem tunedBeta_noise {S ε : ℝ} (hS : 0 ≤ S) (hε : 0 < ε) :
    4 * tunedBeta S ε * S ≤ ε ^ 2 / 2 := by
  have h := (le_div_iff₀ (show 0 < 8 * (S + 1) by positivity)).mp
    (show tunedBeta S ε ≤ ε ^ 2 / (8 * (S + 1)) from min_le_right _ _)
  have hβ := (tunedBeta_pos hS hε).le
  nlinarith only [h, hβ]

/-- A common step that always satisfies every layer's square-root threshold. -/
def safeGamma {p : ℕ} (K : Fin p → ℝ) : ℝ := 1 / (1 + ∑ i, K i)

theorem safeGamma_pos {p : ℕ} {K : Fin p → ℝ} (hK : ∀ i, 0 ≤ K i) : 0 < safeGamma K := by
  unfold safeGamma
  exact one_div_pos.mpr (by have := Finset.sum_nonneg (fun i (_ : i ∈ Finset.univ) => hK i); linarith)

theorem safeGamma_valid {p : ℕ} {K : Fin p → ℝ} (hK : ∀ i, 0 ≤ K i) (i : Fin p) :
    safeGamma K * K i ≤ 1 := by
  have hs := Finset.sum_nonneg (fun i (_ : i ∈ Finset.univ) => hK i)
  have hi := Finset.single_le_sum (fun j (_ : j ∈ Finset.univ) => hK j) (Finset.mem_univ i)
  unfold safeGamma
  rw [one_div_mul_eq_div]
  apply (div_le_one (by linarith : 0 < 1 + ∑ i, K i)).mpr
  linarith only [hi]

/-- A direct construction: first choose beta, then zeta, then gamma. The original
beta/gamma fields of `base` are discarded, so these choices require no fixed point. -/
def explicitControls {p : ℕ} (base : Fin p → Parameters) (N : ℕ)
    (σ ell2 τ : Fin p → ℝ) (ε : ℝ) (hε : 0 < ε) : Fin p → Parameters :=
  let S := ∑ i, noiseScale (base i) N * σ i ^ 2
  have hS : 0 ≤ S := Finset.sum_nonneg (fun i _ => mul_nonneg (noiseScale_nonneg _ _) (sq_nonneg _))
  let β := tunedBeta S ε
  let cb : Fin p → Parameters := fun i => { base i with
    β := β, β_pos := tunedBeta_pos hS hε, β_le_one := tunedBeta_le_one _ _ }
  let K := fun i => 2 * (cb i).L + 2 * Real.sqrt (zeta (cb i) (ell2 i) (τ i))
  have hK : ∀ i, 0 ≤ K i := fun i => add_nonneg
    (mul_nonneg (by norm_num) (cb i).L_nonneg) (by positivity)
  fun i => { cb i with γ := safeGamma K, γ_pos := safeGamma_pos hK }

theorem explicitControls_beta {p : ℕ} (base : Fin p → Parameters) (N : ℕ)
    (σ ell2 τ : Fin p → ℝ) (ε : ℝ) (hε : 0 < ε) (i : Fin p) :
    (explicitControls base N σ ell2 τ ε hε i).β =
      tunedBeta (∑ j, noiseScale (explicitControls base N σ ell2 τ ε hε j) N * σ j ^ 2) ε := rfl

theorem explicitControls_gamma {p : ℕ} (base : Fin p → Parameters) (N : ℕ)
    (σ ell2 τ : Fin p → ℝ) (ε : ℝ) (hε : 0 < ε) (i : Fin p) :
    (explicitControls base N σ ell2 τ ε hε i).γ = safeGamma (fun j =>
      2 * (explicitControls base N σ ell2 τ ε hε j).L +
        2 * Real.sqrt (zeta (explicitControls base N σ ell2 τ ε hε j) (ell2 j) (τ j))) := rfl

theorem explicitControls_uniform {p : ℕ} (base : Fin p → Parameters) (N : ℕ)
    (σ ell2 τ : Fin p → ℝ) (ε : ℝ) (hε : 0 < ε) (i j : Fin p) :
    (explicitControls base N σ ell2 τ ε hε i).γ =
      (explicitControls base N σ ell2 τ ε hε j).γ := rfl

def stochasticBudget (Ψ γ ε : ℝ) : ℕ := max 1 ⌈8 * Ψ / (γ * ε ^ 2)⌉₊

theorem budget_transient {Ψ γ ε : ℝ} (hγ : 0 < γ) (hε : 0 < ε) :
    4 * Ψ / (γ * stochasticBudget Ψ γ ε) ≤ ε ^ 2 / 2 := by
  have hT : 0 < stochasticBudget Ψ γ ε := lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  have hTr : (0 : ℝ) < stochasticBudget Ψ γ ε := by exact_mod_cast hT
  have hceil : 8 * Ψ / (γ * ε ^ 2) ≤ (stochasticBudget Ψ γ ε : ℝ) := by
    apply (Nat.le_ceil _).trans
    exact_mod_cast (show ⌈8 * Ψ / (γ * ε ^ 2)⌉₊ ≤ stochasticBudget Ψ γ ε from le_max_right _ _)
  have h := (div_le_iff₀ (mul_pos hγ (sq_pos_of_pos hε))).mp hceil
  apply (div_le_iff₀ (mul_pos hγ hTr)).mpr
  nlinarith only [h]

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def Problem.noiseScaleTotal (P : Problem N m n μ c) : ℝ :=
  ∑ i, noiseScale (c i) N * P.σ i ^ 2

omit [IsProbabilityMeasure μ] in
theorem Problem.noiseScaleTotal_nonneg (P : Problem N m n μ c) : 0 ≤ P.noiseScaleTotal :=
  Finset.sum_nonneg (fun _ _ => mul_nonneg (noiseScale_nonneg _ _) (sq_nonneg _))

omit [IsProbabilityMeasure μ] in
theorem Problem.tuned_noise_floor (P : Problem N m n μ c) {γ ε : ℝ} (hγ : 0 < γ) (hε : 0 < ε)
    (hsteps : ∀ i, (c i).γ = γ) (hβ : ∀ i, (c i).β = tunedBeta P.noiseScaleTotal ε) :
    4 * P.noiseTotal / γ ≤ ε ^ 2 / 2 := by
  have h := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_right (noise_le (c i) N) (sq_nonneg (P.σ i)))
  simp only [hβ, mul_assoc, ← Finset.mul_sum] at h
  have he : P.noiseTotal = γ * ∑ i, noise (c i) N * P.σ i ^ 2 := by
    simp only [Problem.noiseTotal, hsteps, mul_assoc, Finset.mul_sum]
  rw [he, mul_left_comm, mul_div_cancel_left₀ _ (ne_of_gt hγ)]
  have hh := mul_le_mul_of_nonneg_left h (by norm_num : (0 : ℝ) ≤ 4)
  have hbudget := tunedBeta_noise P.noiseScaleTotal_nonneg hε
  exact hh.trans (by simpa only [Problem.noiseScaleTotal, mul_assoc] using hbudget)

/-- An explicit accuracy-dependent momentum, common step, and rounded iteration budget.
With uniform steps the left side is E[sum_t sum_i ||grad_i f(X_t)||_*^2] / T. -/
theorem Problem.stationarity_at_budget (P : Problem N m n μ c) (A : Scheme N m n c)
    {γ ε : ℝ} (hγ : 0 < γ) (hε : 0 < ε)
    (hsteps : ∀ i, (c i).γ = γ)
    (hβ : ∀ i, (c i).β = tunedBeta P.noiseScaleTotal ε)
    (hγchoice : γ = safeGamma (fun i => 2 * (c i).L +
      2 * Real.sqrt (zeta (c i) (P.ell2 i) (A.τ i))))
    (s : States N m n) :
    P.stationarity A γ s (stochasticBudget (P.potential s) γ ε) ≤ ε ^ 2 := by
  have hK (i : Fin p) : 0 ≤ 2 * (c i).L + 2 * Real.sqrt (zeta (c i) (P.ell2 i) (A.τ i)) :=
    add_nonneg (mul_nonneg (by norm_num) (c i).L_nonneg) (by positivity)
  have hsmall (i : Fin p) :
      2 * (c i).L * (c i).γ + 4 * zeta (c i) (P.ell2 i) (A.τ i) * (c i).γ ^ 2 ≤ 1 := by
    apply (c i).step_condition_for (zeta_nonneg (c i) (P.ell2_nonneg i) (A.τ_nonneg i))
    rw [hsteps i, hγchoice]
    exact safeGamma_valid hK i
  have hT : 0 < stochasticBudget (P.potential s) γ ε := lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  have h := P.average_stationarity A hsmall hγ s _ hT
  have hn := P.tuned_noise_floor hγ hε hsteps hβ
  have ht := budget_transient (Ψ := P.potential s) hγ hε
  linarith only [h, hn, ht]

end EF21Muon.Distributed
