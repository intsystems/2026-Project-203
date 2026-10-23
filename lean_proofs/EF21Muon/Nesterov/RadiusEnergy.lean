import EF21Muon.Nesterov.FirstTracking
import EF21Muon.Distributed.RadiusEnergy

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar

variable {N m n : ℕ}

def radiusUpdate (c : Parameters) (CU : Space m n → Space m n)
    (s : Layer N m n) (reply : Fin N → Space m n) : Layer N m n :=
  let M := fun j => (1-c.β) • s.1.M j+c.β • reply j
  let V := fun j => (1-c.β) • M j+c.β • reply j
  ((radiusPoint c s.1, radiusPoint c s.1, M, fun j => s.1.G j+CU (V j-s.1.G j)),s.1.M)

def radiusHWeight (c : Parameters) : ℝ := 4*c.γ*c.ρ/(1-radiusQ c)
def radiusJWeight (c : Parameters) : ℝ := 2*c.γ*c.ρ+radiusQ c*radiusHWeight c
def radiusRWeight (c : Parameters) : ℝ := 2*radiusQ c*radiusHWeight c+radiusJWeight c

theorem radius_weights_nonneg (c : Parameters) :
    0 ≤ radiusDWeight c ∧ 0 ≤ radiusHWeight c ∧ 0 ≤ radiusJWeight c ∧
    0 ≤ radiusRWeight c ∧ 0 ≤ radiusZWeight c := by
  have hb := c.β_pos
  have hg := c.γ_pos
  have hr := c.ρ_nonneg
  have hq := radiusQ_nonneg c
  have hq1 := sub_pos.mpr (radiusQ_lt_one c)
  dsimp [radiusDWeight,radiusHWeight,radiusJWeight,radiusRWeight,radiusZWeight]
  exact ⟨by positivity,by positivity,by positivity,by positivity,by positivity⟩

def radiusEnergy (c : Parameters) (s : Layer N m n) (g Z : Space m n)
    (localG : Fin N → Space m n) : ℝ :=
  radiusDWeight c*debiasedError s.1 g Z+radiusRWeight c*firstLocal s.1 localG+
    radiusHWeight c*firstCompression c s+radiusJWeight c*firstIncrement s+radiusZWeight c*‖Z‖^2

theorem radiusEnergy_nonneg (c : Parameters) (s : Layer N m n) (g Z : Space m n)
    (localG : Fin N → Space m n) : 0 ≤ radiusEnergy c s g Z localG := by
  rcases radius_weights_nonneg c with ⟨hd,hh,hj,hr,hz⟩
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (mul_nonneg hd (norm_nonneg _))
    (mul_nonneg hr (average_nonneg fun _ => norm_nonneg _)))
    (mul_nonneg hh (average_nonneg fun _ => norm_nonneg _)))
    (mul_nonneg hj (average_nonneg fun _ => norm_nonneg _))) (mul_nonneg hz (sq_nonneg _))

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

theorem radiusEnergy_integrable (c : Parameters) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU) (s : Layer N m n)
    (Z new : Space m n) {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n μ q σ) :
    Integrable (fun ω => radiusEnergy c (radiusUpdate c CU s (O.replies ω)) new
      (newMomentum c.β Z 0 O.mean ω) q) μ := by
  have hiD := radius_debiased_integrable c CU s.1 Z new O
  have hiF := first_local_integrable c CU id s O
  have hiH := first_compression_integrable c hm hc id s O
  have hiJ := first_increment_integrable c CU id s O
  have hiZ := square_integrable (newMomentum_memLp c.β Z 0 O.mean_memLp)
  exact ((((hiD.const_mul _).add (hiF.const_mul _)).add (hiH.const_mul _)).add
    (hiJ.const_mul _)).add (hiZ.const_mul _)

set_option maxHeartbeats 1600000 in
theorem radiusEnergy_recurrence (c : Parameters) (hN : 0 < N)
    {CU : Space m n → Space m n} (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (s : Layer N m n) (Z old new : Space m n) (oldJ q : Fin N → Space m n)
    {σ : ℝ} (hσ : 0 ≤ σ) (O : Noise N m n μ q σ) (hmean : average q = new) :
    (∫ ω, radiusEnergy c (radiusUpdate c CU s (O.replies ω)) new
      (newMomentum c.β Z 0 O.mean ω) q ∂μ)+
      2*c.γ*c.ρ*(debiasedError s.1 old Z+firstCompression c s+firstIncrement s)+
      c.γ*c.ρ/Real.sqrt c.β*‖Z‖^2 ≤
    radiusEnergy c s old Z oldJ+radiusDWeight c*(1-c.β)*‖old-new‖+
      radiusRWeight c*average (fun j => ‖oldJ j-q j‖)+
      2*radiusRWeight c*c.β*σ+radiusZWeight c*c.β^2*σ^2/N := by
  rcases radius_weights_nonneg c with ⟨hD,hH,hJ,hR,hZ⟩
  have hd := mul_le_mul_of_nonneg_left (radius_debiased_recurrence c CU s.1 Z old new O hmean) hD
  have hf := mul_le_mul_of_nonneg_left (first_local_recurrence c hN CU id s oldJ q hσ O) hR
  have hh := mul_le_mul_of_nonneg_left (first_compression_recurrence c hm hc id s O) hH
  have hj := mul_le_mul_of_nonneg_left (first_increment_recurrence c hN CU id s oldJ q hσ O) hR
  have hz := mul_le_mul_of_nonneg_left (radius_noise_recurrence c hN Z O) hZ
  have hiD := radius_debiased_integrable c CU s.1 Z new O
  have hiF := first_local_integrable c CU id s O
  have hiH := first_compression_integrable c hm hc id s O
  have hiJ := first_increment_integrable c CU id s O
  have hiZ := square_integrable (newMomentum_memLp c.β Z 0 O.mean_memLp)
  change Integrable (fun ω => debiasedError (radiusUpdate c CU s (O.replies ω)).1 new
    (newMomentum c.β Z 0 O.mean ω)) μ at hiD
  change Integrable (fun ω => firstLocal (radiusUpdate c CU s (O.replies ω)).1 q) μ at hiF
  change Integrable (fun ω => firstCompression c (radiusUpdate c CU s (O.replies ω))) μ at hiH
  change Integrable (fun ω => firstIncrement (radiusUpdate c CU s (O.replies ω))) μ at hiJ
  have hifour : Integrable (fun ω =>
      radiusDWeight c*debiasedError (radiusUpdate c CU s (O.replies ω)).1 new
        (newMomentum c.β Z 0 O.mean ω)+
      radiusRWeight c*firstLocal (radiusUpdate c CU s (O.replies ω)).1 q+
      radiusHWeight c*firstCompression c (radiusUpdate c CU s (O.replies ω))+
      radiusJWeight c*firstIncrement (radiusUpdate c CU s (O.replies ω))) μ :=
    (((hiD.const_mul _).add (hiF.const_mul _)).add (hiH.const_mul _)).add (hiJ.const_mul _)
  simp only [radiusEnergy]
  rw [integral_add hifour (hiZ.const_mul _)]
  rw [integral_add4 (hiD.const_mul _) (hiF.const_mul _) (hiH.const_mul _) (hiJ.const_mul _)]
  simp only [integral_const_mul]
  rcases radius_weight_identities c with ⟨hDb,_,hZb⟩
  have hHq : radiusHWeight c*(1-radiusQ c) = 4*c.γ*c.ρ := by
    dsimp [radiusHWeight]
    exact div_mul_cancel₀ _ (ne_of_gt (sub_pos.mpr (radiusQ_lt_one c)))
  have hJq : radiusJWeight c-radiusHWeight c*radiusQ c = 2*c.γ*c.ρ := by
    dsimp [radiusJWeight]; ring
  have hReq : radiusRWeight c = 2*radiusQ c*radiusHWeight c+radiusJWeight c := rfl
  have hD' := congrArg (fun a : ℝ => a*debiasedError s.1 old Z) hDb
  have hH' := congrArg (fun a : ℝ => a*firstCompression c s) hHq
  have hJ' := congrArg (fun a : ℝ => a*firstIncrement s) hJq
  have hZ' := congrArg (fun a : ℝ => a*‖Z‖^2) hZb
  have hR' := congrArg (fun a : ℝ => a*
    (∫ ω, firstIncrement (update c CU id s (O.replies ω)) ∂μ)) hReq
  have hdrop : 0 ≤ 2*c.γ*c.ρ*firstCompression c s := mul_nonneg
    (mul_nonneg (mul_nonneg (by norm_num) c.γ_pos.le) c.ρ_nonneg)
    (average_nonneg fun _ => norm_nonneg _)
  change _ ≤ radiusHWeight c*(radiusQ c*firstCompression c s+
    2*radiusQ c*(∫ ω, firstIncrement (update c CU id s (O.replies ω)) ∂μ)+
    radiusQ c*firstIncrement s) at hh
  change radiusDWeight c*(∫ ω, debiasedError (radiusUpdate c CU s (O.replies ω)).1 new
    (newMomentum c.β Z 0 O.mean ω) ∂μ) ≤ _ at hd
  change radiusRWeight c*(∫ ω, firstLocal (radiusUpdate c CU s (O.replies ω)).1 q ∂μ) ≤ _ at hf
  change radiusHWeight c*(∫ ω, firstCompression c (radiusUpdate c CU s (O.replies ω)) ∂μ) ≤ _ at hh
  change radiusRWeight c*(∫ ω, firstIncrement (radiusUpdate c CU s (O.replies ω)) ∂μ) ≤ _ at hj
  change radiusRWeight c*(∫ ω, firstIncrement (radiusUpdate c CU s (O.replies ω)) ∂μ) = _ at hR'
  have hJeq (ω : Ω) : firstIncrement (update c CU id s (O.replies ω)) =
      firstIncrement (radiusUpdate c CU s (O.replies ω)) := rfl
  simp only [hJeq] at hh hR'
  simp only [div_eq_mul_inv] at hd hf hh hj hz hD' hH' hJ' hZ' hR' ⊢
  nlinarith only [hd,hf,hh,hj,hz,hD',hH',hJ',hZ',hR',hdrop]

theorem radiusEnergy_with_tracking (c : Parameters) (hN : 0 < N)
    {CU : Space m n → Space m n} (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (s : Layer N m n) (Z old new : Space m n) (oldJ q : Fin N → Space m n)
    {σ : ℝ} (hσ : 0 ≤ σ) (O : Noise N m n μ q σ) (hmean : average q = new) :
    2*c.γ*c.ρ*‖average s.1.G-old‖+
      (∫ ω, radiusEnergy c (radiusUpdate c CU s (O.replies ω)) new
        (newMomentum c.β Z 0 O.mean ω) q ∂μ) ≤
    radiusEnergy c s old Z oldJ+radiusDWeight c*(1-c.β)*‖old-new‖+
      radiusRWeight c*average (fun j => ‖oldJ j-q j‖)+
      2*radiusRWeight c*c.β*σ+radiusZWeight c*c.β^2*σ^2/N+c.γ*c.ρ*Real.sqrt c.β := by
  have hr := radiusEnergy_recurrence c hN hm hc s Z old new oldJ q hσ O hmean
  have hn := radius_noise_balance c Z
  have he := mul_le_mul_of_nonneg_left (first_mean_error c s old Z)
    (show 0 ≤ 2*c.γ*c.ρ from
      mul_nonneg (mul_nonneg (by norm_num) c.γ_pos.le) c.ρ_nonneg)
  linarith only [hr,hn,he]

end EF21Muon.Nesterov
