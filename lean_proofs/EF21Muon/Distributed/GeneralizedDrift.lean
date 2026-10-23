import EF21Muon.Distributed.GeneralizedSetup
import EF21Muon.Probability.GeometricDrift

/-! Generalized-smooth drift and finite-horizon stationarity. The drift
coefficient is allowed to multiply the objective gap; no bounded-gradient
assumption is made. -/

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω}

def GeneralizedProblem.globalCoefficient (P : GeneralizedProblem N m n μ)
    (x : Model m n) (i : Fin p) : ℝ := P.L₀ i+P.L₁ i*nuclearNorm (P.grad x i)

def GeneralizedProblem.localCoefficient (P : GeneralizedProblem N m n μ)
    (x : Model m n) (i : Fin p) : ℝ :=
  average (fun j => P.localL₀ j i+P.localL₁ j i*nuclearNorm (P.localGrad j x i))

def radiusGlobalDrift (c : Parameters) : ℝ := c.γ^2/2+radiusDWeight c*(1-c.β)*c.γ
def radiusLocalDrift (c : Parameters) : ℝ := radiusLWeight c*c.γ
def radiusNoise (c : Parameters) (N : ℕ) (σ : ℝ) : ℝ :=
  2*radiusLWeight c*c.β*σ + radiusZWeight c*c.β^2*σ^2/N + c.γ*c.ρ*Real.sqrt c.β

theorem radiusDrift_nonneg (c : Parameters) {σ : ℝ} (hσ : 0 ≤ σ) (N : ℕ) :
    0 ≤ radiusGlobalDrift c ∧ 0 ≤ radiusLocalDrift c ∧ 0 ≤ radiusNoise c N σ := by
  rcases radius_weights_nonneg c with ⟨hD, hL, _, hZ⟩
  have hg := c.γ_pos.le
  have hb := c.β_pos.le
  have hr := c.ρ_nonneg
  have hb1 := sub_nonneg.mpr c.β_le_one
  unfold radiusGlobalDrift radiusLocalDrift radiusNoise
  exact ⟨by positivity, by positivity, by positivity⟩

theorem GeneralizedProblem.radius_gradient_movement (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) (s : States N m n) (i : Fin p) :
    ‖P.grad (points s) i-P.grad (radiusPoints c s) i‖ ≤
      (c i).γ*P.globalCoefficient (points s) i ∧
    average (fun j => ‖P.localGrad j (points s) i-P.localGrad j (radiusPoints c s) i‖) ≤
      (c i).γ*P.localCoefficient (points s) i := by
  have hd : spectralNorm (points s i-radiusPoints c s i) ≤ (c i).γ := by
    have h := radiusStep_distance (fun i => (c i).γ) (fun i => (c i).γ_pos.le)
      (points s) (fun i => average (s i).G) i
    rw [← neg_sub (radiusPoints c s i) (points s i)]
    simpa only [radiusStep, radiusPoints, radiusPoint, points, spectralNorm,
      map_neg, norm_neg] using h
  constructor
  · have hC : 0 ≤ P.globalCoefficient (points s) i :=
      add_nonneg (P.L₀_nonneg i) (mul_nonneg (P.L₁_nonneg i) (nuclearNorm_nonneg _))
    have h := (frobenius_le_nuclear _).trans ((P.smooth _ _ i).trans
      (mul_le_mul_of_nonneg_left hd hC))
    simpa only [GeneralizedProblem.globalCoefficient, mul_comm] using h
  · have h (j : Fin N) : ‖P.localGrad j (points s) i-P.localGrad j (radiusPoints c s) i‖ ≤
        (c i).γ*(P.localL₀ j i+P.localL₁ j i*nuclearNorm (P.localGrad j (points s) i)) := by
      have hC := add_nonneg (P.localL₀_nonneg j i)
        (mul_nonneg (P.localL₁_nonneg j i) (nuclearNorm_nonneg (P.localGrad j (points s) i)))
      have hb := (frobenius_le_nuclear _).trans (((P.localSmooth j) _ _ i).trans
        (mul_le_mul_of_nonneg_left hd hC))
      simpa only [mul_comm] using hb
    simpa only [average_mul, GeneralizedProblem.localCoefficient] using average_mono h

set_option maxHeartbeats 800000 in
theorem GeneralizedProblem.radius_drift_raw [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters} (A : Scheme N m n c)
    (z : RadiusStates N m n) :
    (∫ ω, P.radiusPotential c (radiusNext P A z ω) ∂μ) + P.radiusCost c z ≤
      P.radiusPotential c z + ∑ i,
        (radiusGlobalDrift (c i)*P.globalCoefficient (points z.1) i +
          radiusLocalDrift (c i)*P.localCoefficient (points z.1) i +
            radiusNoise (c i) N (P.σ i)) := by
  have hd := generalized_tracking_descent P.derivative P.smooth P.L₀_nonneg P.L₁_nonneg
    (fun i => (c i).ρ_nonneg) A.dual (fun i => (c i).γ) (fun i => (c i).γ_pos.le)
    (points z.1) (fun i => average (z.1 i).M) (fun i => average (z.1 i).G)
  have he (i : Fin p) := radiusEnergy_with_tracking (c i) P.workers_pos
    (A.measurable_CU i) (A.contracts_CU i) (z.1 i) (z.2 i) (P.grad (points z.1) i)
    (P.grad (radiusPoints c z.1) i) (fun j => P.localGrad j (points z.1) i)
    (fun j => P.localGrad j (radiusPoints c z.1) i) (P.σ_nonneg i)
    (P.radiusFrozen c z.1 i) (P.mean_gradient _ i)
  have hE (i : Fin p) :
      (∫ ω, radiusEnergy (c i) ((radiusNext P A z ω).1 i) (P.grad (radiusPoints c z.1) i)
        ((radiusNext P A z ω).2 i) (fun j => P.localGrad j (radiusPoints c z.1) i) ∂μ) ≤
      radiusEnergy (c i) (z.1 i) (P.grad (points z.1) i) (z.2 i)
        (fun j => P.localGrad j (points z.1) i) -
        2*(c i).γ*(c i).ρ*(‖average (z.1 i).M-P.grad (points z.1) i‖ +
          ‖average (z.1 i).G-average (z.1 i).M‖) +
        radiusDWeight (c i)*(1-(c i).β)*(c i).γ*P.globalCoefficient (points z.1) i +
        radiusLocalDrift (c i)*P.localCoefficient (points z.1) i + radiusNoise (c i) N (P.σ i) := by
    simp only [(radiusNext_frozen P A z _ i).1, (radiusNext_frozen P A z _ i).2]
    rcases P.radius_gradient_movement c z.1 i with ⟨hg, hl⟩
    have hg' := mul_le_mul_of_nonneg_left hg (mul_nonneg (radius_weights_nonneg (c i)).1
      (sub_nonneg.mpr (c i).β_le_one))
    have hl' := mul_le_mul_of_nonneg_left hl (radius_weights_nonneg (c i)).2.1
    dsimp [radiusLocalDrift, radiusNoise]
    linarith only [he i, hg', hl']
  have hi (i : Fin p) : Integrable (fun ω =>
      radiusEnergy (c i) ((radiusNext P A z ω).1 i) (P.grad (radiusPoints c z.1) i)
        ((radiusNext P A z ω).2 i) (fun j => P.localGrad j (radiusPoints c z.1) i)) μ := by
    simp only [(radiusNext_frozen P A z _ i).1, (radiusNext_frozen P A z _ i).2]
    exact radiusEnergy_integrable (c i) (A.measurable_CU i) (A.contracts_CU i)
      (z.1 i) (z.2 i) _ (P.radiusFrozen c z.1 i)
  simp only [GeneralizedProblem.radiusPotential, points_radiusNext]
  rw [integral_add (integrable_const _) (integrable_finsetSum _ fun i _ => hi i),
    integral_finsetSum _ (fun i _ => hi i), integral_const]
  simp only [probReal_univ, one_smul]
  have hs := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hE i)
  change P.f (radiusPoints c z.1) ≤ _ at hd
  dsimp [GeneralizedProblem.radiusCost, GeneralizedProblem.radiusPotential,
    GeneralizedProblem.globalCoefficient, radiusGlobalDrift] at *
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_neg_distrib,
    mul_add, add_mul, neg_mul] at hd hs ⊢
  linarith only [hd, hs]

def GeneralizedProblem.gradientScale (P : GeneralizedProblem N m n μ) : ℝ :=
  1 + (∑ i, P.L₁ i) + ∑ j, ∑ i, P.localL₁ j i

theorem GeneralizedProblem.gradientScale_bounds (P : GeneralizedProblem N m n μ) :
    0 < P.gradientScale ∧ (∀ i, P.L₁ i ≤ P.gradientScale) ∧
      (∀ j i, P.localL₁ j i ≤ P.gradientScale) := by
  have hG := Finset.sum_nonneg (fun i (_ : i ∈ Finset.univ) => P.L₁_nonneg i)
  have hL := Finset.sum_nonneg (fun j (_ : j ∈ Finset.univ) =>
    Finset.sum_nonneg (fun i (_ : i ∈ Finset.univ) => P.localL₁_nonneg j i))
  constructor
  · unfold gradientScale; linarith only [hG, hL]
  constructor
  · intro i
    have h := Finset.single_le_sum (fun k (_ : k ∈ Finset.univ) => P.L₁_nonneg k)
      (Finset.mem_univ i)
    unfold gradientScale; linarith only [h, hL]
  · intro j i
    have h := Finset.single_le_sum (fun k (_ : k ∈ Finset.univ) => P.localL₁_nonneg j k)
      (Finset.mem_univ i)
    have h' := Finset.single_le_sum (fun k (_ : k ∈ Finset.univ) =>
      Finset.sum_nonneg (fun l (_ : l ∈ Finset.univ) => P.localL₁_nonneg k l))
      (Finset.mem_univ j)
    unfold gradientScale; linarith only [h, h', hG]

def GeneralizedProblem.globalOffset (P : GeneralizedProblem N m n μ) : ℝ :=
  (∑ i, P.L₀ i)/P.gradientScale

def GeneralizedProblem.localOffset (P : GeneralizedProblem N m n μ) : ℝ :=
  average (fun j => (∑ i, P.localL₀ j i)/P.gradientScale)

theorem GeneralizedProblem.gradient_first_bound (P : GeneralizedProblem N m n μ)
    (x : Model m n) (i : Fin p) :
    nuclearNorm (P.grad x i) ≤ 2*P.gradientScale*(P.f x-P.floor)+P.globalOffset ∧
    average (fun j => nuclearNorm (P.localGrad j x i)) ≤
      2*P.gradientScale*(P.f x-P.floor)+P.localOffset := by
  rcases P.gradientScale_bounds with ⟨hM, hG, hL⟩
  have hg := generalized_weighted_gradient_self_bound P.derivative P.smooth P.L₀_nonneg
    P.L₁_nonneg P.lower_bound (fun _ => 1) (fun _ => by norm_num) hM
    (fun k => by simpa only [one_mul] using hG k) x
  simp only [one_mul, one_pow] at hg
  have hsingle := Finset.single_le_sum (fun k (_ : k ∈ Finset.univ) =>
    nuclearNorm_nonneg (P.grad x k)) (Finset.mem_univ i)
  constructor
  · exact hsingle.trans hg
  · have hj (j : Fin N) : nuclearNorm (P.localGrad j x i) ≤
        2*P.gradientScale*(P.localF j x-P.localMin j) +
          (∑ k, P.localL₀ j k)/P.gradientScale := by
      have h := generalized_weighted_gradient_self_bound (P.local_derivative j)
        (P.localSmooth j) (P.localL₀_nonneg j) (P.localL₁_nonneg j) (P.local_lower j)
        (fun _ => 1) (fun _ => by norm_num) hM
        (fun k => by simpa only [one_mul] using hL j k) x
      simp only [one_mul, one_pow] at h
      exact (Finset.single_le_sum (fun k (_ : k ∈ Finset.univ) =>
        nuclearNorm_nonneg (P.localGrad j x k)) (Finset.mem_univ i)).trans h
    have h := average_mono hj
    simpa only [average_add, average_mul, average_sub, P.mean_objective,
      GeneralizedProblem.floor, GeneralizedProblem.localOffset] using h

def GeneralizedProblem.globalA (P : GeneralizedProblem N m n μ) (i : Fin p) : ℝ :=
  P.L₀ i+P.L₁ i*P.globalOffset
def GeneralizedProblem.globalB (P : GeneralizedProblem N m n μ) (i : Fin p) : ℝ :=
  2*P.gradientScale*P.L₁ i
def GeneralizedProblem.localA (P : GeneralizedProblem N m n μ) (i : Fin p) : ℝ :=
  average (fun j => P.localL₀ j i)+P.gradientScale*P.localOffset
def GeneralizedProblem.localB (P : GeneralizedProblem N m n μ) : ℝ :=
  2*P.gradientScale^2

theorem GeneralizedProblem.coefficient_envelopes (P : GeneralizedProblem N m n μ)
    (x : Model m n) (i : Fin p) :
    P.globalCoefficient x i ≤ P.globalA i+P.globalB i*(P.f x-P.floor) ∧
    P.localCoefficient x i ≤ P.localA i+P.localB*(P.f x-P.floor) := by
  rcases P.gradient_first_bound x i with ⟨hg, hl⟩
  rcases P.gradientScale_bounds with ⟨hM, _, hL⟩
  constructor
  · have h := mul_le_mul_of_nonneg_left hg (P.L₁_nonneg i)
    dsimp [globalCoefficient, globalA, globalB]
    nlinarith only [h]
  · have hj (j : Fin N) :
        P.localL₀ j i+P.localL₁ j i*nuclearNorm (P.localGrad j x i) ≤
          P.localL₀ j i+P.gradientScale*nuclearNorm (P.localGrad j x i) :=
      add_le_add le_rfl (mul_le_mul_of_nonneg_right (hL j i) (nuclearNorm_nonneg _))
    have h := average_mono hj
    simp only [average_add, average_mul] at h
    have h' := mul_le_mul_of_nonneg_left hl hM.le
    dsimp [localCoefficient, localA, localB]
    rw [average_add]
    nlinarith only [h, h']

theorem GeneralizedProblem.envelopes_nonneg (P : GeneralizedProblem N m n μ) (i : Fin p) :
    0 ≤ P.globalA i ∧ 0 ≤ P.globalB i ∧ 0 ≤ P.localA i ∧ 0 ≤ P.localB := by
  have hM := P.gradientScale_bounds.1.le
  have hg : 0 ≤ P.globalOffset := div_nonneg
    (Finset.sum_nonneg fun i _ => P.L₀_nonneg i) hM
  have hl : 0 ≤ P.localOffset := average_nonneg fun j => div_nonneg
    (Finset.sum_nonneg fun i _ => P.localL₀_nonneg j i) hM
  have h0 := P.L₀_nonneg i
  have h1 := P.L₁_nonneg i
  have hj := average_nonneg (fun j => P.localL₀_nonneg j i)
  unfold globalA globalB localA localB
  exact ⟨by positivity, by positivity, by positivity, by positivity⟩

def GeneralizedProblem.radiusKappa (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) : ℝ := ∑ i,
  (radiusGlobalDrift (c i)*P.globalB i+radiusLocalDrift (c i)*P.localB)

def GeneralizedProblem.radiusBias (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) : ℝ := ∑ i,
  (radiusGlobalDrift (c i)*P.globalA i+radiusLocalDrift (c i)*P.localA i+radiusNoise (c i) N (P.σ i))

theorem GeneralizedProblem.radius_constants_nonneg (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) : 0 ≤ P.radiusKappa c ∧ 0 ≤ P.radiusBias c := by
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

theorem GeneralizedProblem.radius_drift [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters} (A : Scheme N m n c)
    (z : RadiusStates N m n) :
    (∫ ω, P.radiusPotential c (radiusNext P A z ω) ∂μ) + P.radiusCost c z ≤
      (1+P.radiusKappa c)*P.radiusPotential c z + P.radiusBias c := by
  have hr := P.radius_drift_raw A z
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
      P.radiusKappa c*(P.f (points z.1)-P.floor)+P.radiusBias c := by
    rw [Finset.sum_add_distrib, ← Finset.sum_mul]
    rfl
  rw [hsum] at hs
  have hgap : P.f (points z.1)-P.floor ≤ P.radiusPotential c z :=
    le_add_of_nonneg_right (Finset.sum_nonneg fun _ _ => radiusEnergy_nonneg _ _ _ _ _)
  have hk := mul_le_mul_of_nonneg_left hgap (P.radius_constants_nonneg c).1
  linarith only [hr, hs, hk]

/-- Finite-horizon bound on the actual stochastic constant-radius updates,
including integrability of the accumulated nuclear-gradient cost. -/
theorem GeneralizedProblem.radius_finite_time [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters} (A : Scheme N m n c)
    (T : ℕ) (z : RadiusStates N m n) :
    Integrable (fun ξ => ((run (radiusNext P A) T z ξ).map (P.radiusCost c)).sum)
      (pathMeasure μ T) ∧
    (∫ ξ, ((run (radiusNext P A) T z ξ).map (P.radiusCost c)).sum ∂pathMeasure μ T) ≤
      (1+P.radiusKappa c)^T*(P.radiusPotential c z+(T:ℝ)*P.radiusBias c) :=
  accumulated_cost_geometric_bound μ (by linarith [(P.radius_constants_nonneg c).1])
    (P.radius_constants_nonneg c).2 (measurable_radiusNext P A)
    (P.measurable_radiusCost c) (P.measurable_radiusPotential c) (P.radiusCost_nonneg c)
    (P.radiusPotential_nonneg c) (P.radiusPotential_integrable A) (P.radius_drift A) T z

end EF21Muon.Distributed
