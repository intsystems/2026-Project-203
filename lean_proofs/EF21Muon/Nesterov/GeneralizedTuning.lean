import EF21Muon.Nesterov.RadiusEnvelopes
import EF21Muon.Distributed.GeneralizedTuning
import EF21Muon.Distributed.GeneralizedSteps

/-! Constant-within-run radii and momentum for one fixed generalized-smooth
problem. All envelope constants below are independent of the horizon. -/

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

def generalizedK (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) : ℝ := ∑ i, η i^2*
  (radiusGlobalEnvelope (base i)*P.globalB i+radiusLocalEnvelope (base i)*P.localB)

def generalizedB (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) : ℝ := ∑ i,
  (η i^2*(radiusGlobalEnvelope (base i)*P.globalA i+radiusLocalEnvelope (base i)*P.localA i) +
    η i*radiusNoiseEnvelope (base i) N (P.σ i))

def generalizedInitial (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) (x : Model m n) : ℝ :=
  P.f x-P.floor + ∑ i, (2*η i*(base i).ρ*‖P.grad x i‖ +
    η i*radiusLocalEnvelope (base i)*average (fun j => ‖P.localGrad j x i‖))

theorem generalized_constants_nonneg (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) {η : Fin p → ℝ} (hη : ∀ i, 0 ≤ η i) (x : Model m n) :
    0 ≤ generalizedK P base η ∧ 0 ≤ generalizedB P base η ∧
      0 ≤ generalizedInitial P base η x := by
  have hK : 0 ≤ generalizedK P base η := by
    apply Finset.sum_nonneg
    intro i _
    rcases radiusEnvelopes_nonneg (base i) (P.σ_nonneg i) N with ⟨hg, hl, _⟩
    rcases P.envelopes_nonneg i with ⟨_, hG, _, hL⟩
    exact mul_nonneg (sq_nonneg _) (add_nonneg (mul_nonneg hg hG) (mul_nonneg hl hL))
  have hB : 0 ≤ generalizedB P base η := by
    apply Finset.sum_nonneg
    intro i _
    rcases radiusEnvelopes_nonneg (base i) (P.σ_nonneg i) N with ⟨hg, hl, hn⟩
    rcases P.envelopes_nonneg i with ⟨hG, _, hL, _⟩
    exact add_nonneg (mul_nonneg (sq_nonneg _) (add_nonneg (mul_nonneg hg hG) (mul_nonneg hl hL)))
      (mul_nonneg (hη i) hn)
  refine ⟨hK, hB, add_nonneg (sub_nonneg.mpr (P.lower_bound x)) ?_⟩
  apply Finset.sum_nonneg
  intro i _
  have hr := (base i).ρ_nonneg
  have hL := (radiusEnvelopes_nonneg (base i) (P.σ_nonneg i) N).2.1
  have hj := average_nonneg (fun j => norm_nonneg (P.localGrad j x i))
  have hi := hη i
  positivity

theorem generalized_kappa_bias (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T : ℕ) :
    radiusKappa P (generalizedHorizonControls base η hη T) ≤ generalizedK P base η/((T:ℝ)+2) ∧
    radiusBias P (generalizedHorizonControls base η hη T) ≤ generalizedB P base η/((T:ℝ)+2) := by
  let c := generalizedHorizonControls base η hη T
  have hd i := radiusDrift_le_envelope (c i)
  have hn i := radiusNoise_le_envelope (c i) (P.σ_nonneg i) N
  have he i : radiusGlobalEnvelope (c i) = radiusGlobalEnvelope (base i) ∧
      radiusLocalEnvelope (c i) = radiusLocalEnvelope (base i) ∧
      radiusNoiseEnvelope (c i) N (P.σ i) = radiusNoiseEnvelope (base i) N (P.σ i) :=
    ⟨rfl, rfl, rfl⟩
  constructor
  · unfold radiusKappa generalizedK
    rw [Finset.sum_div]
    apply Finset.sum_le_sum
    intro i _
    rcases P.envelopes_nonneg i with ⟨_, hG, _, hL⟩
    have hg := mul_le_mul_of_nonneg_right (hd i).1 hG
    have hl := mul_le_mul_of_nonneg_right (hd i).2 hL
    rw [(generalized_tuning_identities base η hη T i).1, (he i).1] at hg
    rw [(generalized_tuning_identities base η hη T i).1, (he i).2.1] at hl
    simp only [div_eq_mul_inv] at hg hl ⊢
    nlinarith only [hg, hl]
  · unfold radiusBias generalizedB
    rw [Finset.sum_div]
    apply Finset.sum_le_sum
    intro i _
    rcases P.envelopes_nonneg i with ⟨hG, _, hL, _⟩
    have hg := mul_le_mul_of_nonneg_right (hd i).1 hG
    have hl := mul_le_mul_of_nonneg_right (hd i).2 hL
    rw [(generalized_tuning_identities base η hη T i).1, (he i).1] at hg
    rw [(generalized_tuning_identities base η hη T i).1, (he i).2.1] at hl
    have hnoise := hn i
    rw [(generalized_tuning_identities base η hη T i).2, (he i).2.2] at hnoise
    simp only [div_eq_mul_inv] at hg hl hnoise ⊢
    nlinarith only [hg, hl, hnoise]

theorem radius_initial_potential (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) (x : Model m n) :
    generalizedPotential P c (radiusStart x) = P.f x-P.floor + ∑ i,
      (radiusDWeight (c i)*‖P.grad x i‖ +
        radiusRWeight (c i)*average (fun j => ‖P.localGrad j x i‖)) := by
  have hx : points (initial x : States N m n) = x := rfl
  simp [generalizedPotential, radiusStart, hx, radiusEnergy, debiasedError,
    firstLocal, firstCompression, firstIncrement, target, initial, start, LayerState.start, LayerState.M,
    LayerState.G, average]

theorem generalized_initial_weights (base : Fin p → Parameters) (η : Fin p → ℝ)
    (hη : ∀ i, 0 < η i) (T : ℕ) (i : Fin p) :
    radiusDWeight (generalizedHorizonControls base η hη T i) ≤ 2*η i*(base i).ρ ∧
    radiusRWeight (generalizedHorizonControls base η hη T i) ≤ η i*radiusLocalEnvelope (base i) := by
  have hu := (generalizedRoot_pos T).ne'
  have hη0 := (hη i).le
  have hρ := (base i).ρ_nonneg
  have hL := (radiusEnvelopes_nonneg (base i) (le_refl (0:ℝ)) 0).2.1
  constructor
  · have he : radiusDWeight (generalizedHorizonControls base η hη T i) =
        2*η i*(base i).ρ/generalizedRoot T := by
      change 2*(η i/generalizedRoot T^3)*(base i).ρ/(1/generalizedRoot T^2) = _
      field_simp
    rw [he]
    exact div_le_self (by positivity) (generalizedRoot_ge_one T)
  · rw [radiusRWeight_eq]
    change (η i/generalizedRoot T^3)*radiusLocalEnvelope (base i) ≤ _
    exact mul_le_mul_of_nonneg_right (div_le_self hη0
      (one_le_pow₀ (generalizedRoot_ge_one T))) hL

theorem generalized_initial_bound (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) (hη : ∀ i, 0 < η i)
    (T : ℕ) (x : Model m n) :
    generalizedPotential P (generalizedHorizonControls base η hη T) (radiusStart x) ≤
      generalizedInitial P base η x := by
  rw [radius_initial_potential P]
  apply add_le_add le_rfl
  apply Finset.sum_le_sum
  intro i _
  rcases generalized_initial_weights base η hη T i with ⟨hD, hL⟩
  exact add_le_add (mul_le_mul_of_nonneg_right hD (norm_nonneg _))
    (mul_le_mul_of_nonneg_right hL (average_nonneg fun _ => norm_nonneg _))

/-- A horizon-independent bound on the total radius-weighted gradient cost.
The objective and the noise law are fixed; only beta and the radii change. -/
theorem generalized_total_cost [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T : ℕ)
    (A : Scheme N m n (generalizedHorizonControls base η hη T)) (x : Model m n) :
    (∫ ξ, ((run (radiusNext P A) T (radiusStart x) ξ).map
      (generalizedCost P (generalizedHorizonControls base η hη T))).sum ∂pathMeasure μ T) ≤
        Real.exp (generalizedK P base η)*(generalizedInitial P base η x+generalizedB P base η) := by
  let c := generalizedHorizonControls base η hη T
  have hfinite := (generalized_finite_time P A T (radiusStart x)).2
  rcases generalized_kappa_bias P base η hη T with ⟨hk, hb⟩
  rcases radius_constants_nonneg P c with ⟨hk0, hb0⟩
  rcases generalized_constants_nonneg P base (fun i => (hη i).le) x with ⟨_, hB0, hI0⟩
  have hS : 0 < (T:ℝ)+2 := by positivity
  have hb' := (le_div_iff₀ hS).mp hb
  have hbT : (T:ℝ)*radiusBias P c ≤ generalizedB P base η := by
    nlinarith only [hb', hb0]
  have hinit := generalized_initial_bound P base η hη T x
  have hsum := add_le_add hinit hbT
  have h1 := mul_le_mul_of_nonneg_left hsum
    (pow_nonneg (show 0 ≤ 1+radiusKappa P c by linarith) T)
  have h2 := mul_le_mul_of_nonneg_right (generalized_exponent_bound hk0 T hk) (add_nonneg hI0 hB0)
  exact hfinite.trans (h1.trans h2)

theorem generalized_total_cost_for [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T k : ℕ) (hk : (k:ℝ) ≤ (T:ℝ)+2)
    (A : Scheme N m n (generalizedHorizonControls base η hη T)) (x : Model m n) :
    (∫ ξ, ((run (radiusNext P A) k (radiusStart x) ξ).map
      (generalizedCost P (generalizedHorizonControls base η hη T))).sum ∂pathMeasure μ k) ≤
        Real.exp (generalizedK P base η)*(generalizedInitial P base η x+generalizedB P base η) := by
  let c := generalizedHorizonControls base η hη T
  have hfinite := (generalized_finite_time P A k (radiusStart x)).2
  rcases generalized_kappa_bias P base η hη T with ⟨hκ, hb⟩
  rcases radius_constants_nonneg P c with ⟨hκ0, hb0⟩
  rcases generalized_constants_nonneg P base (fun i => (hη i).le) x with ⟨_, hB0, hI0⟩
  have hS : 0 < (T:ℝ)+2 := by positivity
  have hb' := (le_div_iff₀ hS).mp hb
  have hbK : (k:ℝ)*radiusBias P c ≤ generalizedB P base η :=
    (mul_le_mul_of_nonneg_right hk hb0).trans (by simpa only [mul_comm] using hb')
  have hinit := generalized_initial_bound P base η hη T x
  have hsum := add_le_add hinit hbK
  have h1 := mul_le_mul_of_nonneg_left hsum
    (pow_nonneg (show 0 ≤ 1+radiusKappa P c by linarith) k)
  have h2 := mul_le_mul_of_nonneg_right (generalized_exponent_bound_for hκ0 hS k hk hκ)
    (add_nonneg hI0 hB0)
  exact hfinite.trans (h1.trans h2)

end EF21Muon.Nesterov
