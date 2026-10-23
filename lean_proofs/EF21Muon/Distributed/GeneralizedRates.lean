import EF21Muon.Distributed.GeneralizedTuning
import EF21Muon.Distributed.WeightedRates
import EF21Muon.Probability.PathStates

/-! The generalized-smooth nuclear first-moment rate, for one fixed problem,
with unequal layer coefficients and exactly the article's horizon schedule. -/

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

def GeneralizedProblem.weightedNuclear (P : GeneralizedProblem N m n μ)
    (w : Fin p → ℝ) (z : RadiusStates N m n) : ℝ :=
  ∑ i, w i*nuclearNorm (P.grad (points z.1) i)

theorem GeneralizedProblem.weightedNuclear_nonneg (P : GeneralizedProblem N m n μ)
    {w : Fin p → ℝ} (hw : ∀ i, 0 ≤ w i) (z : RadiusStates N m n) :
    0 ≤ P.weightedNuclear w z :=
  Finset.sum_nonneg fun i _ => mul_nonneg (hw i) (nuclearNorm_nonneg _)

theorem GeneralizedProblem.measurable_weightedNuclear (P : GeneralizedProblem N m n μ)
    (w : Fin p → ℝ) : Measurable (P.weightedNuclear w) := by
  apply Finset.measurable_sum
  intro i _
  exact (continuous_nuclearNorm.measurable.comp ((measurable_pi_apply i).comp
    (P.gradient_continuous.measurable.comp (measurable_points.comp measurable_fst)))).const_mul _

/-- The observable is exactly the sum of weighted expectations used in the
article, whenever its nonnegative total is integrable. -/
theorem GeneralizedProblem.integral_weightedNuclear (P : GeneralizedProblem N m n μ)
    {Ξ : Type*} [MeasurableSpace Ξ] {ν : Measure Ξ} {X : Ξ → RadiusStates N m n}
    (hX : Measurable X) {w : Fin p → ℝ} (hw : ∀ i, 0 ≤ w i)
    (hi : Integrable (fun ξ => P.weightedNuclear w (X ξ)) ν) :
    (∫ ξ, P.weightedNuclear w (X ξ) ∂ν) =
      ∑ i, w i*(∫ ξ, nuclearNorm (P.grad (points (X ξ).1) i) ∂ν) := by
  have hm i : Measurable (fun ξ => nuclearNorm (P.grad (points (X ξ).1) i)) :=
    continuous_nuclearNorm.measurable.comp ((measurable_pi_apply i).comp
      (P.gradient_continuous.measurable.comp (measurable_points.comp hX.fst)))
  have hint (i : Fin p) : Integrable (fun ξ => w i*nuclearNorm (P.grad (points (X ξ).1) i)) ν := by
    apply hi.mono' ((hm i).const_mul (w i)).aestronglyMeasurable
    apply Filter.Eventually.of_forall
    intro ξ
    change ‖w i*nuclearNorm (P.grad (points (X ξ).1) i)‖ ≤ _
    rw [Real.norm_of_nonneg (mul_nonneg (hw i) (nuclearNorm_nonneg _))]
    exact Finset.single_le_sum (fun j (_ : j ∈ Finset.univ) =>
      mul_nonneg (hw j) (nuclearNorm_nonneg _)) (Finset.mem_univ i)
  unfold GeneralizedProblem.weightedNuclear
  rw [integral_finsetSum _ (fun i _ => hint i)]
  simp only [integral_const_mul]

def GeneralizedProblem.radiusAverage (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (w : Fin p → ℝ)
    (x : Model m n) (T : ℕ) : ℝ :=
  (∫ ξ, ((run (radiusNext P A) (T+1) (radiusStart x) ξ).map
    (P.weightedNuclear w)).sum ∂pathMeasure μ (T+1))/(T+1:ℕ)

def GeneralizedProblem.radiusMinimum (P : GeneralizedProblem N m n μ)
    {c : Fin p → Parameters} (A : Scheme N m n c) (w : Fin p → ℝ)
    (x : Model m n) (T : ℕ) : ℝ :=
  expectedMinimum μ (radiusNext P A) (P.weightedNuclear w) (radiusStart x) T

theorem GeneralizedProblem.radius_weighted_cost_scale (P : GeneralizedProblem N m n μ)
    (base : Fin p → Parameters) (η : Fin p → ℝ) (hη : ∀ i, 0 < η i)
    {ηbar : ℝ} (hbar : 0 < ηbar) (T : ℕ) (z : RadiusStates N m n) :
    P.weightedNuclear (fun i => η i/ηbar) z = generalizedRoot T^3/ηbar *
      P.radiusCost (generalizedHorizonControls base η hη T) z := by
  unfold GeneralizedProblem.weightedNuclear GeneralizedProblem.radiusCost
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  change η i/ηbar * _ = generalizedRoot T^3/ηbar * (η i/generalizedRoot T^3 * _)
  have hu := (generalizedRoot_pos T).ne'
  field_simp

theorem generalized_average_factor {ηbar : ℝ} (hbar : 0 < ηbar) (T : ℕ) :
    (generalizedRoot T^3/ηbar)/(T+1:ℕ) ≤ 2/(ηbar*generalizedRoot T) := by
  have hu := generalizedRoot_pos T
  have hk : (0:ℝ) < (T+1:ℕ) := by positivity
  apply (mul_le_mul_iff_of_pos_right (mul_pos (mul_pos hbar hu) hk)).mp
  have he₁ : ((generalizedRoot T^3/ηbar)/(T+1:ℕ)) *
      (ηbar*generalizedRoot T*(T+1:ℕ)) = generalizedRoot T^4 := by field_simp
  have he₂ : (2/(ηbar*generalizedRoot T))*(ηbar*generalizedRoot T*(T+1:ℕ)) =
      2*(T+1:ℕ) := by field_simp
  rw [he₁, he₂, generalizedRoot_fourth]
  push_cast
  have ht : (0:ℝ) ≤ T := Nat.cast_nonneg T
  linarith only [ht]

/-- The arithmetic average is bounded at the same rate as the paper's minimum.
The constant is explicit and independent of the horizon. -/
theorem GeneralizedProblem.generalized_average_rate [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T)) (x : Model m n) :
    P.radiusAverage A (fun i => η i/ηbar) x T ≤
      (2*Real.exp (P.generalizedK base η)*
        (P.generalizedInitial base η x+P.generalizedB base η)/ηbar)/generalizedRoot T := by
  let C := Real.exp (P.generalizedK base η)*(P.generalizedInitial base η x+P.generalizedB base η)
  rcases P.generalized_constants_nonneg base (fun i => (hη i).le) x with ⟨_, hB, hI⟩
  have hC : 0 ≤ C := mul_nonneg (Real.exp_nonneg _) (add_nonneg hI hB)
  have htotal := P.generalized_total_cost_for base η hη T (T+1)
    (by push_cast; linarith) A x
  have hu := generalizedRoot_pos T
  have hscale : 0 ≤ (generalizedRoot T^3/ηbar)/(T+1:ℕ) := by positivity
  have hs := mul_le_mul_of_nonneg_left htotal hscale
  have hf := mul_le_mul_of_nonneg_right (generalized_average_factor hbar T) hC
  unfold GeneralizedProblem.radiusAverage
  have he : P.weightedNuclear (fun i => η i/ηbar) = fun z => generalizedRoot T^3/ηbar *
      P.radiusCost (generalizedHorizonControls base η hη T) z :=
    funext (fun z => P.radius_weighted_cost_scale base η hη hbar T z)
  rw [he]
  simp_rw [sum_map_scale, integral_const_mul]
  apply le_trans _ (le_trans hs ?_)
  · exact le_of_eq (by ring)
  · apply hf.trans
    exact le_of_eq (by dsimp [C]; ring)

theorem GeneralizedProblem.generalized_weighted_sum_integrable [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T k : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T)) (x : Model m n) :
    Integrable (fun ξ => ((run (radiusNext P A) k (radiusStart x) ξ).map
      (P.weightedNuclear (fun i => η i/ηbar))).sum) (pathMeasure μ k) := by
  have he : P.weightedNuclear (fun i => η i/ηbar) = fun z => generalizedRoot T^3/ηbar *
      P.radiusCost (generalizedHorizonControls base η hη T) z :=
    funext (fun z => P.radius_weighted_cost_scale base η hη hbar T z)
  rw [he]
  simp_rw [sum_map_scale]
  exact (P.radius_finite_time A k (radiusStart x)).1.const_mul _

theorem GeneralizedProblem.generalized_minimum_rate [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (T : ℕ) (A : Scheme N m n (generalizedHorizonControls base η hη T)) (x : Model m n) :
    P.radiusMinimum A (fun i => η i/ηbar) x T ≤
      (2*Real.exp (P.generalizedK base η)*
        (P.generalizedInitial base η x+P.generalizedB base η)/ηbar)/generalizedRoot T := by
  have hi := P.generalized_weighted_sum_integrable base η hη hbar T (T+1) A x
  have hm := expectedMinimum_le_average μ (measurable_radiusNext P A)
    (P.measurable_weightedNuclear _) (P.weightedNuclear_nonneg
      (fun i => div_nonneg (hη i).le hbar.le)) (radiusStart x) T hi
  exact hm.trans (P.generalized_average_rate base η hη hbar T A x)

/-- The minimum is over the expectations at iterates 0,...,T, with fixed
weights eta_i/etaBar. The right side is `(T+2)^(-1/4)`. -/
theorem GeneralizedProblem.generalized_horizon_rate_isBigO [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (base : Fin p → Parameters)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) {ηbar : ℝ} (hbar : 0 < ηbar)
    (A : ∀ T, Scheme N m n (generalizedHorizonControls base η hη T)) (x : Model m n) :
    (fun T => P.radiusMinimum (A T) (fun i => η i/ηbar) x T)
      =O[Filter.atTop] (fun T : ℕ => 1/generalizedRoot T) := by
  apply Asymptotics.IsBigO.of_bound (2*Real.exp (P.generalizedK base η)*
    (P.generalizedInitial base η x+P.generalizedB base η)/ηbar)
  apply Filter.Eventually.of_forall
  intro T
  have h := P.generalized_minimum_rate base η hη hbar T (A T) x
  have h0 := expectedMinimum_nonneg μ (radiusNext P (A T))
    (P.weightedNuclear (fun i => η i/ηbar))
    (P.weightedNuclear_nonneg (fun i => div_nonneg (hη i).le hbar.le)) (radiusStart x) T
  change 0 ≤ P.radiusMinimum (A T) (fun i => η i/ηbar) x T at h0
  rw [Real.norm_eq_abs, abs_of_nonneg h0, Real.norm_eq_abs,
    abs_of_pos (div_pos (by norm_num) (generalizedRoot_pos T))]
  simpa only [div_eq_mul_inv, mul_one, one_mul] using h

/-- Reuse a fixed uplink with the identity downlink and retuned beta/radii. -/
def generalizedHorizonScheme {base : Fin p → Parameters} (A : Scheme N m n base)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T : ℕ) :
    Scheme N m n (generalizedHorizonControls base η hη T) where
  CU := A.CU
  CP _ := id
  measurable_CU := A.measurable_CU
  measurable_CP _ := measurable_id
  contracts_CU := A.contracts_CU
  τ _ := 0
  τ_nonneg _ := le_refl _
  broadcast _ := identity_broadcast_bound _
  dual := A.dual

def generalizedMatrixBase (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) : Fin p → Parameters :=
  matrixControls false (fun _ => 1) (fun _ => 1) (fun _ => 0) hm hn
    (fun _ => by norm_num) (fun _ => le_refl _) (fun _ => by norm_num) (fun _ => le_refl _)

def generalizedMatrixScheme (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (T : ℕ) :
    Scheme N m n (generalizedHorizonControls (generalizedMatrixBase hm hn) η hη T) :=
  generalizedHorizonScheme
    (matrixScheme false (fun _ => 1) (fun _ => 1) (fun _ => 0) hm hn
      (fun _ => by norm_num) (fun _ => le_refl _) (fun _ => by norm_num) (fun _ => le_refl _)) η hη T

/-- Corollary 2 for the actual scaled-sign uplink and exact downlink, with the
article's arithmetic-mean layer weights and one fixed stochastic problem. -/
theorem matrix_generalized_horizon_rate_isBigO [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (hp : 0 < p)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (x : Model m n) :
    (fun T => P.radiusMinimum (generalizedMatrixScheme hm hn η hη T)
      (fun i => η i/average η) x T)
      =O[Filter.atTop] (fun T : ℕ => 1/generalizedRoot T) :=
  P.generalized_horizon_rate_isBigO (generalizedMatrixBase hm hn) η hη
    (layer_average_pos hp hη) (generalizedMatrixScheme hm hn η hη) x

theorem generalizedRoot_rpow (T : ℕ) : generalizedRoot T = ((T:ℝ)+2)^((1:ℝ)/4) := by
  unfold generalizedRoot
  simp_rw [Real.sqrt_eq_rpow]
  rw [← Real.rpow_mul (by positivity)]
  norm_num

theorem generalizedRoot_inv_rpow (T : ℕ) :
    1/generalizedRoot T = ((T:ℝ)+2)^(-(1:ℝ)/4) := by
  rw [generalizedRoot_rpow, one_div]
  have he : -(1:ℝ)/4 = -((1:ℝ)/4) := by ring
  rw [he, Real.rpow_neg (by positivity)]

/-- The same theorem in the conventional unshifted `T^(-1/4)` notation. -/
theorem matrix_generalized_horizon_rate_rpow [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (hp : 0 < p)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (η : Fin p → ℝ) (hη : ∀ i, 0 < η i) (x : Model m n) :
    (fun T => P.radiusMinimum (generalizedMatrixScheme hm hn η hη T)
      (fun i => η i/average η) x T)
      =O[Filter.atTop] (fun T : ℕ => (T:ℝ)^(-(1:ℝ)/4)) := by
  apply (matrix_generalized_horizon_rate_isBigO P hp hm hn η hη x).trans
  apply Asymptotics.IsBigO.of_bound 1
  filter_upwards [Filter.eventually_gt_atTop (0:ℕ)] with T hT
  have ht : (0:ℝ) < T := by exact_mod_cast hT
  rw [generalizedRoot_inv_rpow, Real.norm_eq_abs,
    abs_of_pos (Real.rpow_pos_of_pos (by positivity) _), Real.norm_eq_abs,
    abs_of_pos (Real.rpow_pos_of_pos ht _), one_mul]
  exact Real.rpow_le_rpow_of_nonpos ht (by linarith) (by norm_num)

/-- The unweighted first-moment statement in Theorem 5(ii). -/
theorem matrix_generalized_unweighted_rate [IsProbabilityMeasure μ]
    (P : GeneralizedProblem N m n μ) (hp : 0 < p)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) {η : ℝ} (hη : 0 < η) (x : Model m n) :
    (fun T => P.radiusMinimum (generalizedMatrixScheme hm hn (fun _ => η) (fun _ => hη) T)
      (fun _ => 1) x T)
      =O[Filter.atTop] (fun T : ℕ => (T:ℝ)^(-(1:ℝ)/4)) := by
  have h := matrix_generalized_horizon_rate_rpow P hp hm hn (fun _ => η) (fun _ => hη) x
  have hw : (fun _ : Fin p => η/average (fun _ : Fin p => η)) = (fun _ => 1) := by
    funext i
    rw [average_const hp, div_self hη.ne']
  rwa [hw] at h

end EF21Muon.Distributed
