import EF21Muon.RandomTies.NesterovPaper
import EF21Muon.Distributed.GeneralizedPaper
import EF21Muon.Nesterov.GeneralizedPaper

/-! Generalized-smooth paper-order rates with fresh fair random ties.
The same horizon-independent envelopes apply to the random-compressor paths. -/

noncomputable section
namespace EF21Muon.RandomTies
set_option maxHeartbeats 800000
set_option backward.isDefEq.respectTransparency false
open MeasureTheory hiding average
open EF21Muon.Distributed EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators
variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def emaRadiusAverage (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (w : Fin p → ℝ)
    (x : Model m n) (T : ℕ) : ℝ :=
  (∫ ξ, ((run (emaRadiusNext P A) (T+1) (radiusStart x) ξ).map
    (P.weightedNuclear w)).sum ∂pathMeasure ((coinLaw N m n).prod μ) (T+1))/(T+1:ℕ)

def emaRadiusMinimum (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (w : Fin p → ℝ)
    (x : Model m n) (T : ℕ) : ℝ :=
  expectedMinimum ((coinLaw N m n).prod μ) (emaRadiusNext P A) (P.weightedNuclear w) (radiusStart x) T

theorem ema_generalized_total_cost_for
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T k : ℕ) (hk : (k:ℝ) ≤ (T:ℝ)+2)
    (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (x : Model m n) :
    (∫ ξ, ((run (emaRadiusNext P A) k (radiusStart x) ξ).map
      (P.radiusCost (generalizedHorizonControls base η hη T))).sum ∂pathMeasure ((coinLaw N m n).prod μ) k) ≤
        Real.exp (P.generalizedK base η)*(P.generalizedInitial base η x+P.generalizedB base η) := by
  let c := generalizedHorizonControls base η hη T
  have hfinite := (ema_generalized_finite_time P A hCU hm hn k (radiusStart x)).2
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

theorem ema_generalized_average_rate
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (x : Model m n) :
    emaRadiusAverage P A (fun i => η i/ηbar) x T ≤
      (2*Real.exp (P.generalizedK base η)*
        (P.generalizedInitial base η x+P.generalizedB base η)/ηbar)/generalizedRoot T := by
  let C := Real.exp (P.generalizedK base η)*(P.generalizedInitial base η x+P.generalizedB base η)
  rcases P.generalized_constants_nonneg base (fun i => (hη i).le) x with ⟨_, hB, hI⟩
  have hC : 0 ≤ C := mul_nonneg (Real.exp_nonneg _) (add_nonneg hI hB)
  have htotal := ema_generalized_total_cost_for P base η hη T (T+1)
    (by push_cast; linarith) A hCU hm hn x
  have hu := generalizedRoot_pos T
  have hscale : 0 ≤ (generalizedRoot T^3/ηbar)/(T+1:ℕ) := by positivity
  have hs := mul_le_mul_of_nonneg_left htotal hscale
  have hf := mul_le_mul_of_nonneg_right (generalized_average_factor hbar T) hC
  unfold emaRadiusAverage
  have he : P.weightedNuclear (fun i => η i/ηbar) = fun z => generalizedRoot T^3/ηbar *
      P.radiusCost (generalizedHorizonControls base η hη T) z :=
    funext (fun z => P.radius_weighted_cost_scale base η hη hbar T z)
  rw [he]
  simp_rw [sum_map_scale, integral_const_mul]
  apply le_trans _ (le_trans hs ?_)
  · exact le_of_eq (by ring)
  · apply hf.trans
    exact le_of_eq (by dsimp [C]; ring)

theorem ema_generalized_weighted_sum_integrable
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T k : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (x : Model m n) :
    Integrable (fun ξ => ((run (emaRadiusNext P A) k (radiusStart x) ξ).map
      (P.weightedNuclear (fun i => η i/ηbar))).sum) (pathMeasure ((coinLaw N m n).prod μ) k) := by
  have he : P.weightedNuclear (fun i => η i/ηbar) = fun z => generalizedRoot T^3/ηbar *
      P.radiusCost (generalizedHorizonControls base η hη T) z :=
    funext (fun z => P.radius_weighted_cost_scale base η hη hbar T z)
  rw [he]
  simp_rw [sum_map_scale]
  exact (ema_generalized_finite_time P A hCU hm hn k (radiusStart x)).1.const_mul _

theorem ema_generalized_minimum_rate
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (x : Model m n) :
    emaRadiusMinimum P A (fun i => η i/ηbar) x T ≤
      (2*Real.exp (P.generalizedK base η)*
        (P.generalizedInitial base η x+P.generalizedB base η)/ηbar)/generalizedRoot T := by
  have hi := ema_generalized_weighted_sum_integrable P base η hη hbar T (T+1) A hCU hm hn x
  have hmin := expectedMinimum_le_average ((coinLaw N m n).prod μ) (measurable_emaRadiusNext P A)
    (P.measurable_weightedNuclear _) (P.weightedNuclear_nonneg
      (fun i => div_nonneg (hη i).le hbar.le)) (radiusStart x) T hi
  exact hmin.trans (ema_generalized_average_rate P base η hη hbar T A hCU hm hn x)

section NesterovRates
open EF21Muon.Nesterov

def nestRadiusAverage (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (w : Fin p → ℝ)
    (x : Model m n) (T : ℕ) : ℝ :=
  (∫ ξ, ((run (nestRadiusNext P A) (T+1) (radiusStart x) ξ).map
    (weightedNuclear P w)).sum ∂pathMeasure ((coinLaw N m n).prod μ) (T+1))/(T+1:ℕ)

def nestRadiusMinimum (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (w : Fin p → ℝ)
    (x : Model m n) (T : ℕ) : ℝ :=
  expectedMinimum ((coinLaw N m n).prod μ) (nestRadiusNext P A) (weightedNuclear P w) (radiusStart x) T


theorem nest_generalized_total_cost_for
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T k : ℕ) (hk : (k:ℝ) ≤ (T:ℝ)+2)
    (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (x : Model m n) :
    (∫ ξ, ((run (nestRadiusNext P A) k (radiusStart x) ξ).map
      (generalizedCost P (generalizedHorizonControls base η hη T))).sum ∂pathMeasure ((coinLaw N m n).prod μ) k) ≤
        Real.exp (generalizedK P base η)*(generalizedInitial P base η x+generalizedB P base η) := by
  let c := generalizedHorizonControls base η hη T
  have hfinite := (nest_generalized_finite_time P A hCU hm hn k (radiusStart x)).2
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

theorem nest_generalized_average_rate
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (x : Model m n) :
    nestRadiusAverage P A (fun i => η i/ηbar) x T ≤
      (2*Real.exp (generalizedK P base η)*
        (generalizedInitial P base η x+generalizedB P base η)/ηbar)/generalizedRoot T := by
  let C := Real.exp (generalizedK P base η)*(generalizedInitial P base η x+generalizedB P base η)
  rcases generalized_constants_nonneg P base (fun i => (hη i).le) x with ⟨_, hB, hI⟩
  have hC : 0 ≤ C := mul_nonneg (Real.exp_nonneg _) (add_nonneg hI hB)
  have htotal := nest_generalized_total_cost_for P base η hη T (T+1)
    (by push_cast; linarith) A hCU hm hn x
  have hu := generalizedRoot_pos T
  have hscale : 0 ≤ (generalizedRoot T^3/ηbar)/(T+1:ℕ) := by positivity
  have hs := mul_le_mul_of_nonneg_left htotal hscale
  have hf := mul_le_mul_of_nonneg_right (generalized_average_factor hbar T) hC
  unfold nestRadiusAverage
  have he : weightedNuclear P (fun i => η i/ηbar) = fun z => generalizedRoot T^3/ηbar *
      generalizedCost P (generalizedHorizonControls base η hη T) z :=
    funext (fun z => radius_weighted_cost_scale P base η hη hbar T z)
  rw [he]
  simp_rw [sum_map_scale, integral_const_mul]
  apply le_trans _ (le_trans hs ?_)
  · exact le_of_eq (by ring)
  · apply hf.trans
    exact le_of_eq (by dsimp [C]; ring)

theorem nest_generalized_weighted_sum_integrable
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T k : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (x : Model m n) :
    Integrable (fun ξ => ((run (nestRadiusNext P A) k (radiusStart x) ξ).map
      (weightedNuclear P (fun i => η i/ηbar))).sum) (pathMeasure ((coinLaw N m n).prod μ) k) := by
  have he : weightedNuclear P (fun i => η i/ηbar) = fun z => generalizedRoot T^3/ηbar *
      generalizedCost P (generalizedHorizonControls base η hη T) z :=
    funext (fun z => radius_weighted_cost_scale P base η hη hbar T z)
  rw [he]
  simp_rw [sum_map_scale]
  exact (nest_generalized_finite_time P A hCU hm hn k (radiusStart x)).1.const_mul _

theorem nest_generalized_minimum_rate
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T))
    (hCU : ∀ i, A.CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (x : Model m n) :
    nestRadiusMinimum P A (fun i => η i/ηbar) x T ≤
      (2*Real.exp (generalizedK P base η)*
        (generalizedInitial P base η x+generalizedB P base η)/ηbar)/generalizedRoot T := by
  have hi := nest_generalized_weighted_sum_integrable P base η hη hbar T (T+1) A hCU hm hn x
  have hmin := expectedMinimum_le_average ((coinLaw N m n).prod μ) (measurable_nestRadiusNext P A)
    (measurable_weightedNuclear P _) (weightedNuclear_nonneg P
      (fun i => div_nonneg (hη i).le hbar.le)) (radiusStart x) T hi
  exact hmin.trans (nest_generalized_average_rate P base η hη hbar T A hCU hm hn x)

end NesterovRates
end EF21Muon.RandomTies
