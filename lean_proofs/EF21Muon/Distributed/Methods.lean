import EF21Muon.Distributed.Tuning

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar

variable {N p : ℕ} {m n : Fin p → ℕ}

def matrixControls (compressed : Bool) (β γ L : Fin p → ℝ)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (hβ : ∀ i, 0 < β i)
    (hβ1 : ∀ i, β i ≤ 1) (hγ : ∀ i, 0 < γ i) (hL : ∀ i, 0 ≤ L i) : Fin p → Parameters :=
  fun i => matrixParameters (m i) (n i) compressed (β i) (γ i) (L i)
    (hm i) (hn i) (hβ i) (hβ1 i) (hγ i) (hL i)

def matrixScheme (compressed : Bool) (β γ L : Fin p → ℝ)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (hβ : ∀ i, 0 < β i)
    (hβ1 : ∀ i, β i ≤ 1) (hγ : ∀ i, 0 < γ i) (hL : ∀ i, 0 ≤ L i) :
    Scheme N m n (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL) where
  CU _ := signCompressor
  CP _ := if compressed then signCompressor else id
  measurable_CU _ := measurable_signCompressor
  measurable_CP _ := by
    cases compressed with
    | false => exact measurable_id
    | true => exact measurable_signCompressor
  contracts_CU i := sign_squared_contracts (hm i) (hn i)
  τ i := if compressed then (Real.sqrt ((min (m i) (n i) : ℕ) : ℝ)) ^ 2 else 0
  τ_nonneg i := by cases compressed <;> simp only [Bool.false_eq_true, ↓reduceIte] <;> positivity
  broadcast i := by
    cases compressed with
    | false => exact identity_broadcast_bound _
    | true => exact downlink_rec _ (sign_squared_contracts (hm i) (hn i)) polar_frobenius_le_sqrt_rank
  dual _ := nuclearNorm_le_sqrt_rank

section Results

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {β γ L : Fin p → ℝ}
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (hβ : ∀ i, 0 < β i)
    (hβ1 : ∀ i, β i ≤ 1) (hγ : ∀ i, 0 < γ i) (hL : ∀ i, 0 ≤ L i)

/-- General noise, arbitrarily many workers and heterogeneous layers; the identity
downlink removes all broadcast terms from the USign step restriction. -/
theorem matrix_stationarity (compressed : Bool)
    (P : Problem N m n μ (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL))
    (hstep : ∀ i, γ i * (2 * L i + 2 * Real.sqrt
      (zeta (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL i) (P.ell2 i)
        ((matrixScheme (N := N) compressed β γ L hm hn hβ hβ1 hγ hL).τ i))) ≤ 1)
    {γbar : ℝ} (hγbar : 0 < γbar) (x : Model m n) (T : ℕ) (hT : 0 < T) :
    P.stationarity (matrixScheme compressed β γ L hm hn hβ hβ1 hγ hL) γbar (start x) T ≤
      4 * P.potential (start x) / (γbar * T) + 4 * P.noiseTotal / γbar := by
  apply P.average_stationarity _ ?_ hγbar _ _ hT
  intro i
  exact (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL i).step_condition_for
    (zeta_nonneg _ (P.ell2_nonneg i) ((matrixScheme compressed β γ L hm hn hβ hβ1 hγ hL).τ_nonneg i))
    (hstep i)

end Results

/-- A finite nuclear second moment, as assumed in the paper, supplies both the
L2 integrability and Frobenius variance fields of the general-noise model. -/
theorem nuclear_variance_suffices {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    {m n : ℕ} {z : Ω → Space m n} {σ : ℝ} (hz : Measurable z)
    (hi : Integrable (fun ω => nuclearNorm (z ω) ^ 2) μ)
    (hv : (∫ ω, nuclearNorm (z ω) ^ 2 ∂μ) ≤ σ ^ 2) :
    MemLp z 2 μ ∧ (∫ ω, ‖z ω‖ ^ 2 ∂μ) ≤ σ ^ 2 := by
  have hb (ω : Ω) : ‖z ω‖ ^ 2 ≤ nuclearNorm (z ω) ^ 2 :=
    pow_le_pow_left₀ (norm_nonneg _) (frobenius_le_nuclear _) 2
  have hf : Integrable (fun ω => ‖z ω‖ ^ 2) μ := hi.mono' (hz.norm.pow_const 2).aestronglyMeasurable
    (Filter.Eventually.of_forall (fun ω => (Real.norm_of_nonneg (sq_nonneg ‖z ω‖)).trans_le (hb ω)))
  exact ⟨(memLp_two_iff_integrable_sq_norm hz.aestronglyMeasurable).mpr hf,
    (integral_mono hf hi hb).trans hv⟩

end EF21Muon.Distributed
