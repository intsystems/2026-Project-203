import EF21Muon.Distributed.GeneralizedDrift

/-! Horizon-independent coefficient envelopes for the generalized rate. -/

noncomputable section

namespace EF21Muon.Distributed

open EF21Muon.Stochastic

def radiusGlobalEnvelope (c : Parameters) : ℝ := 1/2+2*c.ρ
def radiusLocalEnvelope (c : Parameters) : ℝ := 2*c.ρ*radiusQ c/(1-radiusQ c)
def radiusNoiseEnvelope (c : Parameters) (N : ℕ) (σ : ℝ) : ℝ :=
  2*radiusLocalEnvelope c*σ+c.ρ*σ^2/N+c.ρ

theorem radiusEnvelopes_nonneg (c : Parameters) {σ : ℝ} (hσ : 0 ≤ σ) (N : ℕ) :
    0 ≤ radiusGlobalEnvelope c ∧ 0 ≤ radiusLocalEnvelope c ∧
      0 ≤ radiusNoiseEnvelope c N σ := by
  have hr := c.ρ_nonneg
  have hq := radiusQ_nonneg c
  have hq1 := sub_pos.mpr (radiusQ_lt_one c)
  have hL : 0 ≤ radiusLocalEnvelope c := by unfold radiusLocalEnvelope; positivity
  unfold radiusGlobalEnvelope radiusNoiseEnvelope
  exact ⟨by positivity, hL, by positivity⟩

theorem radiusLWeight_eq (c : Parameters) :
    radiusLWeight c = c.γ*radiusLocalEnvelope c := by
  unfold radiusLWeight radiusSWeight radiusLocalEnvelope
  ring

theorem radiusZWeight_variance (c : Parameters) :
    radiusZWeight c*c.β^2 = c.γ*c.ρ*Real.sqrt c.β := by
  have hb := c.β_pos.ne'
  have hs := (Real.sqrt_pos.mpr c.β_pos).ne'
  have he : c.β/Real.sqrt c.β = Real.sqrt c.β :=
    (div_eq_iff hs).mpr (by nlinarith only [Real.sq_sqrt c.β_pos.le])
  calc
    _ = c.γ*c.ρ*(c.β/Real.sqrt c.β) := by unfold radiusZWeight; field_simp
    _ = _ := by rw [he]

theorem radiusDrift_le_envelope (c : Parameters) :
    radiusGlobalDrift c ≤ c.γ^2/c.β*radiusGlobalEnvelope c ∧
      radiusLocalDrift c ≤ c.γ^2/c.β*radiusLocalEnvelope c := by
  have hb := c.β_pos
  have hr := c.ρ_nonneg
  constructor
  · apply (mul_le_mul_iff_of_pos_right hb).mp
    have he : radiusGlobalDrift c*c.β = c.γ^2*c.β/2+2*c.γ^2*c.ρ*(1-c.β) := by
      unfold radiusGlobalDrift radiusDWeight
      field_simp
    rw [he]
    have he' : (c.γ^2/c.β*radiusGlobalEnvelope c)*c.β =
        c.γ^2*(1/2+2*c.ρ) := by unfold radiusGlobalEnvelope; field_simp
    rw [he']
    have h1 := mul_le_mul_of_nonneg_left c.β_le_one (sq_nonneg c.γ)
    have h2 := mul_nonneg (mul_nonneg (sq_nonneg c.γ) hr) hb.le
    nlinarith only [h1, h2]
  · have hL := (radiusEnvelopes_nonneg c (le_refl (0:ℝ)) 0).2.1
    have hg : c.γ^2 ≤ c.γ^2/c.β :=
      (le_div_iff₀ hb).mpr (mul_le_of_le_one_right (sq_nonneg _) c.β_le_one)
    have he : radiusLocalDrift c = c.γ^2*radiusLocalEnvelope c := by
      rw [radiusLocalDrift, radiusLWeight_eq]; ring
    rw [he]
    exact mul_le_mul_of_nonneg_right hg hL

theorem radiusNoise_le_envelope (c : Parameters) {σ : ℝ} (hσ : 0 ≤ σ) (N : ℕ) :
    radiusNoise c N σ ≤ c.γ*Real.sqrt c.β*radiusNoiseEnvelope c N σ := by
  have hs := Real.sqrt_nonneg c.β
  have hs2 := Real.sq_sqrt c.β_pos.le
  have hs1 : Real.sqrt c.β ≤ 1 := by nlinarith only [hs, hs2, c.β_le_one]
  have hb : c.β ≤ Real.sqrt c.β := by nlinarith only [hs, hs2, hs1]
  have hL := (radiusEnvelopes_nonneg c hσ N).2.1
  have hm := mul_le_mul_of_nonneg_left hb
    (show 0 ≤ 2*c.γ*radiusLocalEnvelope c*σ by
      exact mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) c.γ_pos.le) hL) hσ)
  unfold radiusNoise radiusNoiseEnvelope
  rw [radiusLWeight_eq]
  have hz : radiusZWeight c*c.β^2*σ^2/N = c.γ*c.ρ*Real.sqrt c.β*σ^2/N := by
    rw [radiusZWeight_variance]
  rw [hz]
  simp only [div_eq_mul_inv] at hm ⊢
  nlinarith only [hm]

end EF21Muon.Distributed
