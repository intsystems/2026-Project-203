import EF21Muon.Nesterov.GeneralizedSetup
import EF21Muon.Probability.GeometricDrift

/-! Generalized-smooth drift and finite-horizon stationarity. The drift
coefficient is allowed to multiply the objective gap; no bounded-gradient
assumption is made. -/

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω}

def radiusGlobalDrift (c : Parameters) : ℝ := c.γ^2/2+radiusDWeight c*(1-c.β)*c.γ
def radiusLocalDrift (c : Parameters) : ℝ := radiusRWeight c*c.γ
def radiusNoise (c : Parameters) (N : ℕ) (σ : ℝ) : ℝ :=
  2*radiusRWeight c*c.β*σ + radiusZWeight c*c.β^2*σ^2/N + c.γ*c.ρ*Real.sqrt c.β

theorem radiusDrift_nonneg (c : Parameters) {σ : ℝ} (hσ : 0 ≤ σ) (N : ℕ) :
    0 ≤ radiusGlobalDrift c ∧ 0 ≤ radiusLocalDrift c ∧ 0 ≤ radiusNoise c N σ := by
  rcases radius_weights_nonneg c with ⟨hD, _, _, hL, hZ⟩
  have hg := c.γ_pos.le
  have hb := c.β_pos.le
  have hr := c.ρ_nonneg
  have hb1 := sub_nonneg.mpr c.β_le_one
  unfold radiusGlobalDrift radiusLocalDrift radiusNoise
  exact ⟨by positivity, by positivity, by positivity⟩

set_option maxHeartbeats 800000 in
theorem radius_drift_raw [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters} (A : Scheme N m n c)
    (z : RadiusStates N m n) :
    (∫ ω, generalizedPotential P c (radiusNext P A z ω) ∂μ) + generalizedCost P c z ≤
      generalizedPotential P c z + ∑ i,
        (radiusGlobalDrift (c i)*P.globalCoefficient (points z.1) i +
          radiusLocalDrift (c i)*P.localCoefficient (points z.1) i +
            radiusNoise (c i) N (P.σ i)) := by
  have hd := generalized_tracking_descent P.derivative P.smooth P.L₀_nonneg P.L₁_nonneg
    (fun i => (c i).ρ_nonneg) A.dual (fun i => (c i).γ) (fun i => (c i).γ_pos.le)
    (points z.1) (fun i => average (z.1 i).1.G) (fun i => average (z.1 i).1.G)
  simp only [sub_self, norm_zero, add_zero] at hd
  have he (i : Fin p) := radiusEnergy_with_tracking (c i) P.workers_pos
    (A.measurable_CU i) (A.contracts_CU i) (z.1 i) (z.2 i) (P.grad (points z.1) i)
    (P.grad (radiusPoints c z.1) i) (fun j => P.localGrad j (points z.1) i)
    (fun j => P.localGrad j (radiusPoints c z.1) i) (P.σ_nonneg i)
    (radiusFrozen P c z.1 i) (P.mean_gradient _ i)
  have hE (i : Fin p) :
      (∫ ω, radiusEnergy (c i) ((radiusNext P A z ω).1 i) (P.grad (radiusPoints c z.1) i)
        ((radiusNext P A z ω).2 i) (fun j => P.localGrad j (radiusPoints c z.1) i) ∂μ) ≤
      radiusEnergy (c i) (z.1 i) (P.grad (points z.1) i) (z.2 i)
        (fun j => P.localGrad j (points z.1) i) -
        2*(c i).γ*(c i).ρ*‖average (z.1 i).1.G-P.grad (points z.1) i‖ +
        radiusDWeight (c i)*(1-(c i).β)*(c i).γ*P.globalCoefficient (points z.1) i +
        radiusLocalDrift (c i)*P.localCoefficient (points z.1) i + radiusNoise (c i) N (P.σ i) := by
    simp only [(radiusNext_frozen P A z _ i).1, (radiusNext_frozen P A z _ i).2]
    rcases P.radius_gradient_movement c (core z.1) i with ⟨hg, hl⟩
    change ‖P.grad (points z.1) i-P.grad (radiusPoints c z.1) i‖ ≤
      (c i).γ*P.globalCoefficient (points z.1) i at hg
    change average (fun j => ‖P.localGrad j (points z.1) i-P.localGrad j (radiusPoints c z.1) i‖) ≤
      (c i).γ*P.localCoefficient (points z.1) i at hl
    have hg' := mul_le_mul_of_nonneg_left hg (mul_nonneg (radius_weights_nonneg (c i)).1
      (sub_nonneg.mpr (c i).β_le_one))
    have hl' := mul_le_mul_of_nonneg_left hl (radius_weights_nonneg (c i)).2.2.2.1
    have he' := he i
    dsimp [radiusLocalDrift, radiusNoise]
    dsimp only [points,radiusPoints,radiusFrozen,core] at he' hg' hl' ⊢
    simp only [div_eq_mul_inv] at he' hg' hl' ⊢
    nlinarith only [he', hg', hl']
  have hi (i : Fin p) : Integrable (fun ω =>
      radiusEnergy (c i) ((radiusNext P A z ω).1 i) (P.grad (radiusPoints c z.1) i)
        ((radiusNext P A z ω).2 i) (fun j => P.localGrad j (radiusPoints c z.1) i)) μ := by
    simp only [(radiusNext_frozen P A z _ i).1, (radiusNext_frozen P A z _ i).2]
    exact radiusEnergy_integrable (c i) (A.measurable_CU i) (A.contracts_CU i)
      (z.1 i) (z.2 i) _ (radiusFrozen P c z.1 i)
  simp only [generalizedPotential, points_radiusNext]
  rw [integral_add (integrable_const _) (integrable_finsetSum _ fun i _ => hi i),
    integral_finsetSum _ (fun i _ => hi i), integral_const]
  simp only [probReal_univ, one_smul]
  have hs := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hE i)
  change P.f (radiusPoints c z.1) ≤ _ at hd
  dsimp [generalizedCost, generalizedPotential,
    GeneralizedProblem.globalCoefficient, radiusGlobalDrift] at *
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_neg_distrib,
    mul_add, add_mul, neg_mul] at hd hs ⊢
  linarith only [hd, hs]

def radiusKappa (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) : ℝ := ∑ i,
  (radiusGlobalDrift (c i)*P.globalB i+radiusLocalDrift (c i)*P.localB)

def radiusBias (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) : ℝ := ∑ i,
  (radiusGlobalDrift (c i)*P.globalA i+radiusLocalDrift (c i)*P.localA i+radiusNoise (c i) N (P.σ i))

theorem radius_constants_nonneg (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) : 0 ≤ radiusKappa P c ∧ 0 ≤ radiusBias P c := by
  constructor
  · apply Finset.sum_nonneg
    intro i _
    rcases radiusDrift_nonneg (c i) (P.σ_nonneg i) N with ⟨hg, hl, _⟩
    rcases P.envelopes_nonneg i with ⟨_, hB, _, hL⟩
    exact add_nonneg (mul_nonneg hg hB) (mul_nonneg hl hL)
  · apply Finset.sum_nonneg
    intro i _
    rcases radiusDrift_nonneg (c i) (P.σ_nonneg i) N with ⟨hg, hl, hn⟩
    rcases P.envelopes_nonneg i with ⟨hA, _, hL, _⟩
    exact add_nonneg (add_nonneg (mul_nonneg hg hA) (mul_nonneg hl hL)) hn

theorem radius_drift [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters} (A : Scheme N m n c)
    (z : RadiusStates N m n) :
    (∫ ω, generalizedPotential P c (radiusNext P A z ω) ∂μ) + generalizedCost P c z ≤
      (1+radiusKappa P c)*generalizedPotential P c z + radiusBias P c := by
  have hr := radius_drift_raw P A z
  have hterm (i : Fin p) :
      radiusGlobalDrift (c i)*P.globalCoefficient (points z.1) i +
        radiusLocalDrift (c i)*P.localCoefficient (points z.1) i + radiusNoise (c i) N (P.σ i) ≤
      (radiusGlobalDrift (c i)*P.globalB i+radiusLocalDrift (c i)*P.localB)*
        (P.f (points z.1)-P.floor) +
      (radiusGlobalDrift (c i)*P.globalA i+radiusLocalDrift (c i)*P.localA i+radiusNoise (c i) N (P.σ i)) := by
    rcases P.coefficient_envelopes (points z.1) i with ⟨hg, hl⟩
    rcases radiusDrift_nonneg (c i) (P.σ_nonneg i) N with ⟨hG, hL, _⟩
    have hg' := mul_le_mul_of_nonneg_left hg hG
    have hl' := mul_le_mul_of_nonneg_left hl hL
    nlinarith only [hg', hl']
  have hs := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hterm i)
  have hsum : (∑ i, ((radiusGlobalDrift (c i)*P.globalB i+radiusLocalDrift (c i)*P.localB)*
        (P.f (points z.1)-P.floor) +
      (radiusGlobalDrift (c i)*P.globalA i+radiusLocalDrift (c i)*P.localA i+radiusNoise (c i) N (P.σ i)))) =
      radiusKappa P c*(P.f (points z.1)-P.floor)+radiusBias P c := by
    rw [Finset.sum_add_distrib, ← Finset.sum_mul]
    rfl
  rw [hsum] at hs
  have hgap : P.f (points z.1)-P.floor ≤ generalizedPotential P c z :=
    le_add_of_nonneg_right (Finset.sum_nonneg fun _ _ => radiusEnergy_nonneg _ _ _ _ _)
  have hk := mul_le_mul_of_nonneg_left hgap (radius_constants_nonneg P c).1
  linarith only [hr, hs, hk]

/-- Finite-horizon bound on the actual stochastic constant-radius updates,
including integrability of the accumulated nuclear-gradient cost. -/
theorem generalized_finite_time [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters} (A : Scheme N m n c)
    (T : ℕ) (z : RadiusStates N m n) :
    Integrable (fun ξ => ((run (radiusNext P A) T z ξ).map (generalizedCost P c)).sum)
      (pathMeasure μ T) ∧
    (∫ ξ, ((run (radiusNext P A) T z ξ).map (generalizedCost P c)).sum ∂pathMeasure μ T) ≤
      (1+radiusKappa P c)^T*(generalizedPotential P c z+(T:ℝ)*radiusBias P c) :=
  accumulated_cost_geometric_bound μ (by linarith [(radius_constants_nonneg P c).1])
    (radius_constants_nonneg P c).2 (measurable_radiusNext P A)
    (generalizedCost_measurable P c) (generalizedPotential_measurable P c) (generalizedCost_nonneg P c)
    (generalizedPotential_nonneg P c) (generalizedPotential_integrable P A) (radius_drift P A) T z

end EF21Muon.Nesterov
