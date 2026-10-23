import EF21Muon.Distributed.RadiusTracking

/-! A first-moment Lyapunov function for exact-downlink constant-radius steps.
The centered noise EMA is kept in squared norm; its weight balances the
`sqrt(beta)` term without assuming bounded gradients. -/

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar

variable {N m n : ℕ}

def radiusQ (c : Parameters) : ℝ := Real.sqrt (1-c.αU)
def radiusDWeight (c : Parameters) : ℝ := 2*c.γ*c.ρ/c.β
def radiusSWeight (c : Parameters) : ℝ := 2*c.γ*c.ρ/(1-radiusQ c)
def radiusLWeight (c : Parameters) : ℝ := radiusSWeight c*radiusQ c
def radiusZWeight (c : Parameters) : ℝ := c.γ*c.ρ/(c.β*Real.sqrt c.β)

theorem radiusQ_nonneg (c : Parameters) : 0 ≤ radiusQ c := Real.sqrt_nonneg _

theorem radiusQ_lt_one (c : Parameters) : radiusQ c < 1 := by
  have hs := Real.sq_sqrt (sub_nonneg.mpr c.αU_le_one)
  have hp := c.αU_pos
  have hn := radiusQ_nonneg c
  dsimp [radiusQ] at *
  nlinarith

theorem radius_weights_nonneg (c : Parameters) :
    0 ≤ radiusDWeight c ∧ 0 ≤ radiusLWeight c ∧
      0 ≤ radiusSWeight c ∧ 0 ≤ radiusZWeight c := by
  have hb := c.β_pos
  have hg := c.γ_pos
  have hr := c.ρ_nonneg
  have hq := radiusQ_nonneg c
  have hq1 := sub_pos.mpr (radiusQ_lt_one c)
  unfold radiusDWeight radiusLWeight radiusSWeight radiusZWeight
  exact ⟨by positivity, by positivity, by positivity, by positivity⟩

theorem radius_weight_identities (c : Parameters) :
    radiusDWeight c*c.β = 2*c.γ*c.ρ ∧
    radiusSWeight c*(1-radiusQ c) = 2*c.γ*c.ρ ∧
    radiusZWeight c*c.β = c.γ*c.ρ/Real.sqrt c.β := by
  have hb := c.β_pos.ne'
  have hq := (sub_pos.mpr (radiusQ_lt_one c)).ne'
  simp only [radiusDWeight, radiusSWeight, radiusZWeight]
  constructor
  · exact div_mul_cancel₀ _ hb
  constructor
  · exact div_mul_cancel₀ _ hq
  · field_simp

def radiusEnergy (c : Parameters) (s : LayerState N m n)
    (g Z : Space m n) (localG : Fin N → Space m n) : ℝ :=
  radiusDWeight c*debiasedError s g Z + radiusLWeight c*firstLocal s localG +
    radiusSWeight c*firstCompression s + radiusZWeight c*‖Z‖^2

theorem radiusEnergy_nonneg (c : Parameters) (s : LayerState N m n)
    (g Z : Space m n) (localG : Fin N → Space m n) :
    0 ≤ radiusEnergy c s g Z localG := by
  rcases radius_weights_nonneg c with ⟨hD, hL, hS, hZ⟩
  exact add_nonneg (add_nonneg (add_nonneg (mul_nonneg hD (norm_nonneg _))
    (mul_nonneg hL (average_nonneg fun _ => norm_nonneg _)))
    (mul_nonneg hS (average_nonneg fun _ => norm_nonneg _)))
    (mul_nonneg hZ (sq_nonneg _))

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

theorem radiusEnergy_integrable (c : Parameters) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU) (s : LayerState N m n)
    (Z new : Space m n) {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    Integrable (fun ω => radiusEnergy c (radiusUpdate c CU s (O.replies ω)) new
      (newMomentum c.β Z 0 O.mean ω) q) μ := by
  have hD := radius_debiased_integrable c CU s Z new O
  have hL := radius_local_integrable c CU s O
  have hS := radius_compression_integrable c hm hc s O
  have hZ := square_integrable (newMomentum_memLp c.β Z 0 O.mean_memLp)
  exact (((hD.const_mul _).add (hL.const_mul _)).add (hS.const_mul _)).add (hZ.const_mul _)

set_option maxHeartbeats 800000 in
/-- The fresh-noise terms cancel from the debiased tracking error. The only
variance term is the centered average-noise EMA, with variance `sigma^2/N`. -/
theorem radiusEnergy_recurrence (c : Parameters) (hN : 0 < N)
    {CU : Space m n → Space m n} (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (s : LayerState N m n) (Z old new : Space m n) (oldJ q : Fin N → Space m n)
    {σ : ℝ} (hσ : 0 ≤ σ) (O : Noise N m n μ q σ) (hmean : average q = new) :
    (∫ ω, radiusEnergy c (radiusUpdate c CU s (O.replies ω)) new
      (newMomentum c.β Z 0 O.mean ω) q ∂μ) +
        2*c.γ*c.ρ*(debiasedError s old Z + firstCompression s) +
          c.γ*c.ρ/Real.sqrt c.β*‖Z‖^2 ≤
      radiusEnergy c s old Z oldJ + radiusDWeight c*(1-c.β)*‖old-new‖ +
        radiusLWeight c*average (fun j => ‖oldJ j-q j‖) +
        2*radiusLWeight c*c.β*σ + radiusZWeight c*c.β^2*σ^2/N := by
  rcases radius_weights_nonneg c with ⟨hD, hL, hS, hZ⟩
  rcases radius_weight_identities c with ⟨hDb, hSq, hZb⟩
  have hd := mul_le_mul_of_nonneg_left
    (radius_debiased_recurrence c CU s Z old new O hmean) hD
  have hl := mul_le_mul_of_nonneg_left (radius_local_recurrence c hN CU s oldJ q hσ O) hL
  have hs := mul_le_mul_of_nonneg_left (radius_compression_recurrence c hN hm hc s oldJ q hσ O) hS
  have hz := mul_le_mul_of_nonneg_left (radius_noise_recurrence c hN Z O) hZ
  have hiD := radius_debiased_integrable c CU s Z new O
  have hiL := radius_local_integrable c CU s O
  have hiS := radius_compression_integrable c hm hc s O
  have hiZ := square_integrable (newMomentum_memLp c.β Z 0 O.mean_memLp)
  simp only [radiusEnergy]
  rw [integral_add4 (hiD.const_mul _) (hiL.const_mul _) (hiS.const_mul _) (hiZ.const_mul _)]
  simp only [integral_const_mul]
  have hLq : radiusSWeight c*radiusQ c = radiusLWeight c := rfl
  change _ ≤ radiusSWeight c *
    (radiusQ c*firstCompression s + radiusQ c*c.β*firstLocal s oldJ +
      radiusQ c*c.β*average (fun j => ‖oldJ j-q j‖) + radiusQ c*c.β*σ) at hs
  have hD' := congrArg (fun a : ℝ => a*debiasedError s old Z) hDb
  have hS' := congrArg (fun a : ℝ => a*firstCompression s) hSq
  have hZ' := congrArg (fun a : ℝ => a*‖Z‖^2) hZb
  have hL' := congrArg (fun a : ℝ => a*c.β*(firstLocal s oldJ +
    average (fun j => ‖oldJ j-q j‖) + σ)) hLq
  simp only [div_eq_mul_inv] at hd hl hs hz hD' hS' hZ' hL' ⊢
  nlinarith only [hd, hl, hs, hz, hD', hS', hZ', hL']

/-- The elementary balance used for the noise EMA. -/
theorem radius_noise_balance (c : Parameters) (Z : Space m n) :
    2*c.γ*c.ρ*‖Z‖ ≤ c.γ*c.ρ/Real.sqrt c.β*‖Z‖^2 + c.γ*c.ρ*Real.sqrt c.β := by
  have hb := Real.sqrt_pos.mpr c.β_pos
  have hy : 2*‖Z‖ ≤ ‖Z‖^2/Real.sqrt c.β + Real.sqrt c.β := by
    have hx : 2*‖Z‖*Real.sqrt c.β ≤ ‖Z‖^2+(Real.sqrt c.β)^2 := by
      nlinarith only [sq_nonneg (‖Z‖-Real.sqrt c.β)]
    have hh := (le_div_iff₀ hb).mpr hx
    have he : (‖Z‖^2+(Real.sqrt c.β)^2)/Real.sqrt c.β =
        ‖Z‖^2/Real.sqrt c.β + Real.sqrt c.β := by field_simp
    rwa [he] at hh
  have hh := mul_le_mul_of_nonneg_left hy (mul_nonneg c.γ_pos.le c.ρ_nonneg)
  simp only [div_eq_mul_inv] at hh ⊢
  convert hh using 1 <;> ring

theorem radius_mean_error_bound (s : LayerState N m n) (g Z : Space m n) :
    ‖average s.M-g‖ + ‖average s.G-average s.M‖ ≤
      debiasedError s g Z + firstCompression s + ‖Z‖ := by
  have he : average s.M-g-Z+Z = average s.M-g := by module
  have hM := norm_add_le (average s.M-g-Z) Z
  rw [he] at hM
  have hS := norm_average_le (fun j => s.M j-s.G j)
  rw [average_sub, norm_sub_rev] at hS
  change _ ≤ firstCompression s at hS
  dsimp [debiasedError]
  linarith only [hM, hS]

theorem radiusEnergy_with_tracking (c : Parameters) (hN : 0 < N)
    {CU : Space m n → Space m n} (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (s : LayerState N m n) (Z old new : Space m n) (oldJ q : Fin N → Space m n)
    {σ : ℝ} (hσ : 0 ≤ σ) (O : Noise N m n μ q σ) (hmean : average q = new) :
    2*c.γ*c.ρ*(‖average s.M-old‖ + ‖average s.G-average s.M‖) +
      (∫ ω, radiusEnergy c (radiusUpdate c CU s (O.replies ω)) new
        (newMomentum c.β Z 0 O.mean ω) q ∂μ) ≤
      radiusEnergy c s old Z oldJ + radiusDWeight c*(1-c.β)*‖old-new‖ +
        radiusLWeight c*average (fun j => ‖oldJ j-q j‖) +
        2*radiusLWeight c*c.β*σ + radiusZWeight c*c.β^2*σ^2/N +
          c.γ*c.ρ*Real.sqrt c.β := by
  have hr := radiusEnergy_recurrence c hN hm hc s Z old new oldJ q hσ O hmean
  have hn := radius_noise_balance c Z
  have he := mul_le_mul_of_nonneg_left (radius_mean_error_bound s old Z)
    (show 0 ≤ 2*c.γ*c.ρ from
      mul_nonneg (mul_nonneg (by norm_num) c.γ_pos.le) c.ρ_nonneg)
  linarith only [hr, hn, he]

end EF21Muon.Distributed
