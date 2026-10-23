import EF21Muon.Distributed.RadiusEnvelopes
import EF21Muon.Distributed.GeneralizedSteps

/-! Constant-within-run radii and momentum for one fixed generalized-smooth
problem. All envelope constants below are independent of the horizon. -/

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

def generalizedRoot (T : ℕ) : ℝ := Real.sqrt (Real.sqrt ((T:ℝ)+2))

theorem generalizedRoot_ge_one (T : ℕ) : 1 ≤ generalizedRoot T :=
  Real.one_le_sqrt.mpr (Real.one_le_sqrt.mpr (by have h : (0:ℝ) ≤ T := Nat.cast_nonneg T; linarith))

theorem generalizedRoot_pos (T : ℕ) : 0 < generalizedRoot T :=
  lt_of_lt_of_le (by norm_num) (generalizedRoot_ge_one T)

theorem generalizedRoot_sq (T : ℕ) : generalizedRoot T^2 = Real.sqrt ((T:ℝ)+2) :=
  Real.sq_sqrt (Real.sqrt_nonneg _)

theorem generalizedRoot_fourth (T : ℕ) : generalizedRoot T^4 = (T:ℝ)+2 := by
  calc
    _ = (generalizedRoot T^2)^2 := by ring
    _ = _ := by rw [generalizedRoot_sq, Real.sq_sqrt (by positivity)]

def generalizedHorizonControls {p : ℕ} (base : Fin p → Parameters) (η : Fin p → ℝ)
    (hη : ∀ i, 0 < η i) (T : ℕ) : Fin p → Parameters :=
  fun i => { base i with
    β := 1/generalizedRoot T^2
    β_pos := div_pos (by norm_num) (pow_pos (generalizedRoot_pos T) 2)
    β_le_one := (div_le_one (pow_pos (generalizedRoot_pos T) 2)).mpr
      (one_le_pow₀ (generalizedRoot_ge_one T))
    γ := η i/generalizedRoot T^3
    γ_pos := div_pos (hη i) (pow_pos (generalizedRoot_pos T) 3) }

theorem generalizedHorizonControls_paper {p : ℕ} (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T : ℕ) (i : Fin p) :
    (generalizedHorizonControls base η hη T i).β = 1/Real.sqrt ((T:ℝ)+2) ∧
    (generalizedHorizonControls base η hη T i).γ = generalizedRadius ((T:ℝ)+2) (η i) := by
  constructor
  · exact congrArg (fun r : ℝ => 1/r) (generalizedRoot_sq T)
  · change η i/generalizedRoot T^3 = _
    unfold generalizedRadius
    have he : generalizedRoot T^3 = Real.sqrt ((T:ℝ)+2)*generalizedRoot T := by
      calc
        _ = generalizedRoot T^2*generalizedRoot T := by ring
        _ = _ := by rw [generalizedRoot_sq]
    rw [he]
    rfl

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

def GeneralizedProblem.generalizedK (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) : ℝ := ∑ i, η i^2*
  (radiusGlobalEnvelope (base i)*P.globalB i+radiusLocalEnvelope (base i)*P.localB)

def GeneralizedProblem.generalizedB (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) : ℝ := ∑ i,
  (η i^2*(radiusGlobalEnvelope (base i)*P.globalA i+radiusLocalEnvelope (base i)*P.localA i) +
    η i*radiusNoiseEnvelope (base i) N (P.σ i))

def GeneralizedProblem.generalizedInitial (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) (x : Model m n) : ℝ :=
  P.f x-P.floor + ∑ i, (2*η i*(base i).ρ*‖P.grad x i‖ +
    η i*radiusLocalEnvelope (base i)*average (fun j => ‖P.localGrad j x i‖))

theorem GeneralizedProblem.generalized_constants_nonneg (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) {η : Fin p → ℝ} (hη : ∀ i, 0 ≤ η i) (x : Model m n) :
    0 ≤ P.generalizedK base η ∧ 0 ≤ P.generalizedB base η ∧
      0 ≤ P.generalizedInitial base η x := by
  have hK : 0 ≤ P.generalizedK base η := by
    apply Finset.sum_nonneg
    intro i _
    rcases radiusEnvelopes_nonneg (base i) (P.σ_nonneg i) N with ⟨hg, hl, _⟩
    rcases P.envelopes_nonneg i with ⟨_, hG, _, hL⟩
    exact mul_nonneg (sq_nonneg _) (add_nonneg (mul_nonneg hg hG) (mul_nonneg hl hL))
  have hB : 0 ≤ P.generalizedB base η := by
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

theorem generalized_tuning_identities (base : Fin p → Parameters) (η : Fin p → ℝ)
    (hη : ∀ i, 0 < η i) (T : ℕ) (i : Fin p) :
    (generalizedHorizonControls base η hη T i).γ^2 /
        (generalizedHorizonControls base η hη T i).β = η i^2/((T:ℝ)+2) ∧
    (generalizedHorizonControls base η hη T i).γ *
      Real.sqrt (generalizedHorizonControls base η hη T i).β = η i/((T:ℝ)+2) := by
  have hu := (generalizedRoot_pos T).ne'
  have h4 := generalizedRoot_fourth T
  have hs : Real.sqrt (1/generalizedRoot T^2) = 1/generalizedRoot T := by
    have he : 1/generalizedRoot T^2 = (1/generalizedRoot T)^2 := by field_simp
    rw [he, Real.sqrt_sq_eq_abs, abs_of_pos (div_pos (by norm_num) (generalizedRoot_pos T))]
  change (η i/generalizedRoot T^3)^2/(1/generalizedRoot T^2) = _ ∧
    (η i/generalizedRoot T^3)*Real.sqrt (1/generalizedRoot T^2) = _
  rw [hs, ← h4]
  constructor <;> field_simp

theorem GeneralizedProblem.generalized_kappa_bias (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T : ℕ) :
    P.radiusKappa (generalizedHorizonControls base η hη T) ≤ P.generalizedK base η/((T:ℝ)+2) ∧
    P.radiusBias (generalizedHorizonControls base η hη T) ≤ P.generalizedB base η/((T:ℝ)+2) := by
  let c := generalizedHorizonControls base η hη T
  have hd i := radiusDrift_le_envelope (c i)
  have hn i := radiusNoise_le_envelope (c i) (P.σ_nonneg i) N
  have he i : radiusGlobalEnvelope (c i) = radiusGlobalEnvelope (base i) ∧
      radiusLocalEnvelope (c i) = radiusLocalEnvelope (base i) ∧
      radiusNoiseEnvelope (c i) N (P.σ i) = radiusNoiseEnvelope (base i) N (P.σ i) :=
    ⟨rfl, rfl, rfl⟩
  constructor
  · unfold GeneralizedProblem.radiusKappa GeneralizedProblem.generalizedK
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
  · unfold GeneralizedProblem.radiusBias GeneralizedProblem.generalizedB
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

def radiusStart (x : Model m n) : RadiusStates N m n := (start x, 0)

theorem GeneralizedProblem.radius_initial_potential (P : GeneralizedProblem N m n μ)
    (c : Fin p → Parameters) (x : Model m n) :
    P.radiusPotential c (radiusStart x) = P.f x-P.floor + ∑ i,
      (radiusDWeight (c i)*‖P.grad x i‖ +
        radiusLWeight (c i)*average (fun j => ‖P.localGrad j x i‖)) := by
  have hx : points (start x : States N m n) = x := rfl
  simp [GeneralizedProblem.radiusPotential, radiusStart, hx, radiusEnergy, debiasedError,
    firstLocal, firstCompression, start, LayerState.start, LayerState.M,
    LayerState.G, average]

theorem generalized_initial_weights (base : Fin p → Parameters) (η : Fin p → ℝ)
    (hη : ∀ i, 0 < η i) (T : ℕ) (i : Fin p) :
    radiusDWeight (generalizedHorizonControls base η hη T i) ≤ 2*η i*(base i).ρ ∧
    radiusLWeight (generalizedHorizonControls base η hη T i) ≤ η i*radiusLocalEnvelope (base i) := by
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
  · rw [radiusLWeight_eq]
    change (η i/generalizedRoot T^3)*radiusLocalEnvelope (base i) ≤ _
    exact mul_le_mul_of_nonneg_right (div_le_self hη0
      (one_le_pow₀ (generalizedRoot_ge_one T))) hL

theorem GeneralizedProblem.generalized_initial_bound (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) (hη : ∀ i, 0 < η i)
    (T : ℕ) (x : Model m n) :
    P.radiusPotential (generalizedHorizonControls base η hη T) (radiusStart x) ≤
      P.generalizedInitial base η x := by
  rw [P.radius_initial_potential]
  apply add_le_add le_rfl
  apply Finset.sum_le_sum
  intro i _
  rcases generalized_initial_weights base η hη T i with ⟨hD, hL⟩
  exact add_le_add (mul_le_mul_of_nonneg_right hD (norm_nonneg _))
    (mul_le_mul_of_nonneg_right hL (average_nonneg fun _ => norm_nonneg _))

theorem generalized_exponent_bound {κ K : ℝ} (hκ : 0 ≤ κ) (T : ℕ)
    (hbound : κ ≤ K/((T:ℝ)+2)) : (1+κ)^T ≤ Real.exp K := by
  have hS : 0 < (T:ℝ)+2 := by positivity
  have hκ' := (le_div_iff₀ hS).mp hbound
  have hpow := pow_le_pow_left₀ (show 0 ≤ 1+κ by linarith)
    (show 1+κ ≤ Real.exp κ by linarith only [Real.add_one_le_exp κ]) T
  rw [← Real.exp_nat_mul] at hpow
  exact hpow.trans (Real.exp_le_exp.mpr (by nlinarith only [hκ', hκ]))

theorem generalized_exponent_bound_for {κ K S : ℝ} (hκ : 0 ≤ κ) (hS : 0 < S)
    (k : ℕ) (hk : (k:ℝ) ≤ S) (hbound : κ ≤ K/S) : (1+κ)^k ≤ Real.exp K := by
  have hκ' := (le_div_iff₀ hS).mp hbound
  have hpow := pow_le_pow_left₀ (show 0 ≤ 1+κ by linarith)
    (show 1+κ ≤ Real.exp κ by linarith only [Real.add_one_le_exp κ]) k
  rw [← Real.exp_nat_mul] at hpow
  exact hpow.trans (Real.exp_le_exp.mpr
    ((mul_le_mul_of_nonneg_right hk hκ).trans (by simpa only [mul_comm] using hκ')))

/-- A horizon-independent bound on the total radius-weighted gradient cost.
The objective and the noise law are fixed; only beta and the radii change. -/
theorem GeneralizedProblem.generalized_total_cost [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T : ℕ)
    (A : Scheme N m n (generalizedHorizonControls base η hη T)) (x : Model m n) :
    (∫ ξ, ((run (radiusNext P A) T (radiusStart x) ξ).map
      (P.radiusCost (generalizedHorizonControls base η hη T))).sum ∂pathMeasure μ T) ≤
        Real.exp (P.generalizedK base η)*(P.generalizedInitial base η x+P.generalizedB base η) := by
  let c := generalizedHorizonControls base η hη T
  have hfinite := (P.radius_finite_time A T (radiusStart x)).2
  rcases P.generalized_kappa_bias base η hη T with ⟨hk, hb⟩
  rcases P.radius_constants_nonneg c with ⟨hk0, hb0⟩
  rcases P.generalized_constants_nonneg base (fun i => (hη i).le) x with ⟨_, hB0, hI0⟩
  have hS : 0 < (T:ℝ)+2 := by positivity
  have hb' := (le_div_iff₀ hS).mp hb
  have hbT : (T:ℝ)*P.radiusBias c ≤ P.generalizedB base η := by
    nlinarith only [hb', hb0]
  have hinit := P.generalized_initial_bound base η hη T x
  have hsum := add_le_add hinit hbT
  have h1 := mul_le_mul_of_nonneg_left hsum
    (pow_nonneg (show 0 ≤ 1+P.radiusKappa c by linarith) T)
  have h2 := mul_le_mul_of_nonneg_right (generalized_exponent_bound hk0 T hk) (add_nonneg hI0 hB0)
  exact hfinite.trans (h1.trans h2)

theorem GeneralizedProblem.generalized_total_cost_for [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T k : ℕ) (hk : (k:ℝ) ≤ (T:ℝ)+2)
    (A : Scheme N m n (generalizedHorizonControls base η hη T)) (x : Model m n) :
    (∫ ξ, ((run (radiusNext P A) k (radiusStart x) ξ).map
      (P.radiusCost (generalizedHorizonControls base η hη T))).sum ∂pathMeasure μ k) ≤
        Real.exp (P.generalizedK base η)*(P.generalizedInitial base η x+P.generalizedB base η) := by
  let c := generalizedHorizonControls base η hη T
  have hfinite := (P.radius_finite_time A k (radiusStart x)).2
  rcases P.generalized_kappa_bias base η hη T with ⟨hκ, hb⟩
  rcases P.radius_constants_nonneg c with ⟨hκ0, hb0⟩
  rcases P.generalized_constants_nonneg base (fun i => (hη i).le) x with ⟨_, hB0, hI0⟩
  have hS : 0 < (T:ℝ)+2 := by positivity
  have hb' := (le_div_iff₀ hS).mp hb
  have hbK : (k:ℝ)*P.radiusBias c ≤ P.generalizedB base η :=
    (mul_le_mul_of_nonneg_right hk hb0).trans (by simpa only [mul_comm] using hb')
  have hinit := P.generalized_initial_bound base η hη T x
  have hsum := add_le_add hinit hbK
  have h1 := mul_le_mul_of_nonneg_left hsum
    (pow_nonneg (show 0 ≤ 1+P.radiusKappa c by linarith) k)
  have h2 := mul_le_mul_of_nonneg_right (generalized_exponent_bound_for hκ0 hS k hk hκ)
    (add_nonneg hI0 hB0)
  exact hfinite.trans (h1.trans h2)

end EF21Muon.Distributed
