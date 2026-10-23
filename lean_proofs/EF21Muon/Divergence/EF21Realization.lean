import EF21Muon.Divergence.EF21FieldTail
import EF21Muon.Divergence.Clipping

/-! A smooth realization of the prescribed gradients. Three disjoint bumps
realize the transient gradients. Smooth clipping makes the objective globally
nonnegative while preserving the complete orbit and its objective increments. -/

noncomputable section

namespace EF21Muon.Divergence.EF21SignMuon

open LongTails.Algorithms.MatrixPolar InnerProductSpace Filter
open scoped ContDiff NNReal

theorem coordinate_dist_le (X Y : Space 2 2) (p : Fin 2 × Fin 2) :
    |X p - Y p| ≤ dist X Y := by
  simpa only [dist_eq_norm, PiLp.sub_apply, Real.norm_eq_abs] using
    PiLp.norm_apply_le (X-Y) p

theorem centers_far (i j : Fin 3) (hij : i ≠ j) :
    (1/50 : ℝ) < dist (position i) (position j) := by
  have hc : (1/50 : ℝ) < |position i (0,1) - position j (0,1)| := by
    fin_cases i <;> fin_cases j <;> first
    | exact False.elim (hij rfl)
    | norm_num [position_one, position_two, show position 0 = 0 from rfl, ofMatrix, matrixEquiv]
  exact hc.trans_le (coordinate_dist_le _ _ _)

theorem tail_far (i : Fin 3) (t : ℕ) (ht : 3 ≤ t) :
    (1/50 : ℝ) < dist (position t) (position i) := by
  have hc : position i (0,1) ≤ 0 := by
    fin_cases i <;> norm_num [position_one, position_two, show position 0 = 0 from rfl,
      ofMatrix, matrixEquiv]
  have hl := tail_coord_lower t ht
  have hn := coordinate_dist_le (position t) (position i) (0,1)
  rw [abs_of_nonneg (by linarith)] at hn
  linarith

def correctionTerm (variant : MomentumVariant) (μ : ℝ) (i : Fin 3) : Space 2 2 → ℝ :=
  correction (requiredGradient variant μ i - gradient (baseField (amplitude variant μ)) (position i))
    (position i)

def corrections (variant : MomentumVariant) (μ : ℝ) (X : Space 2 2) : ℝ :=
  ∑ i : Fin 3, correctionTerm variant μ i X

theorem corrections_contDiff (variant : MomentumVariant) (μ : ℝ) :
    ContDiff ℝ ∞ (corrections variant μ) :=
  ContDiff.sum (fun _ _ => correction_contDiff _ _)

theorem corrections_compact (variant : MomentumVariant) (μ : ℝ) :
    HasCompactSupport (corrections variant μ) := by
  have he : corrections variant μ = ∑ i : Fin 3, correctionTerm variant μ i := by
    funext X
    simp only [corrections, Finset.sum_apply]
  rw [he]
  exact HasCompactSupport.finset_sum (fun _ _ => correction_compact _ _)

theorem corrections_gradient (variant : MomentumVariant) (μ : ℝ) (X : Space 2 2) :
    gradient (corrections variant μ) X =
      ∑ i : Fin 3, gradient (correctionTerm variant μ i) X :=
  gradient_sum (fun _ => (correction_contDiff _ _).differentiable (by simp)) X

theorem corrections_gradient_center (variant : MomentumVariant) (μ : ℝ) (j : Fin 3) :
    gradient (corrections variant μ) (position j) =
      requiredGradient variant μ j - gradient (baseField (amplitude variant μ)) (position j) := by
  rw [corrections_gradient, Finset.sum_eq_single j]
  · exact correction_gradient_center _ _
  · intro i _ hij
    exact correction_gradient_zero (centers_far j i hij.symm)
  · simp

theorem corrections_gradient_tail (variant : MomentumVariant) (μ : ℝ) (t : ℕ) (ht : 3 ≤ t) :
    gradient (corrections variant μ) (position t) = 0 := by
  rw [corrections_gradient]
  exact Finset.sum_eq_zero (fun i _ => correction_gradient_zero (tail_far i t ht))

theorem corrections_tail (variant : MomentumVariant) (μ : ℝ) (t : ℕ) (ht : 3 ≤ t) :
    corrections variant μ (position t) = 0 := by
  exact Finset.sum_eq_zero (fun i _ => correction_zero (tail_far i t ht).le)

def rawObjective (variant : MomentumVariant) (μ : ℝ) (X : Space 2 2) : ℝ :=
  baseField (amplitude variant μ) X + corrections variant μ X

theorem rawObjective_contDiff (variant : MomentumVariant) (μ : ℝ) :
    ContDiff ℝ ∞ (rawObjective variant μ) :=
  (baseField_contDiff _).add (corrections_contDiff _ _)

theorem rawObjective_lipschitz_gradient (variant : MomentumVariant) (μ : ℝ) :
    ∃ K, LipschitzWith K (gradient (rawObjective variant μ)) := by
  obtain ⟨K,hK⟩ := baseField_lipschitz_gradient (amplitude variant μ)
  obtain ⟨J,hJ⟩ := compact_smooth_lipschitz_gradient (corrections_contDiff variant μ)
    (corrections_compact variant μ)
  refine ⟨K+J, ?_⟩
  have he : gradient (rawObjective variant μ) = fun X =>
      gradient (baseField (amplitude variant μ)) X + gradient (corrections variant μ) X :=
    funext (gradient_add ((baseField_contDiff _).differentiable (by simp))
      ((corrections_contDiff _ _).differentiable (by simp)))
  rw [he]
  exact hK.add hJ

theorem rawObjective_bounded_gradient (variant : MomentumVariant) (μ : ℝ) :
    ∃ G : ℝ≥0, ∀ X, ‖gradient (rawObjective variant μ) X‖ ≤ G := by
  obtain ⟨G,hG⟩ := baseField_bounded_gradient (amplitude variant μ)
  obtain ⟨H,hH⟩ := compact_smooth_bounded_gradient (corrections_contDiff variant μ)
    (corrections_compact variant μ)
  refine ⟨G+H, fun X => ?_⟩
  rw [show gradient (rawObjective variant μ) X = _ from
    gradient_add ((baseField_contDiff _).differentiable (by simp))
      ((corrections_contDiff _ _).differentiable (by simp)) X]
  exact (norm_add_le _ _).trans (add_le_add (hG X) (hH X))

theorem rawObjective_gradient (variant : MomentumVariant) {μ : ℝ} (hμ1 : μ < 1) (t : ℕ) :
    gradient (rawObjective variant μ) (position t) = requiredGradient variant μ t := by
  rw [show gradient (rawObjective variant μ) (position t) = _ from
    gradient_add ((baseField_contDiff _).differentiable (by simp))
      ((corrections_contDiff _ _).differentiable (by simp)) (position t)]
  by_cases ht : t < 3
  · have he := corrections_gradient_center variant μ ⟨t,ht⟩
    change gradient (corrections variant μ) (position t) = _ at he
    rw [he]
    abel
  · rw [corrections_gradient_tail variant μ t (by omega), add_zero,
      baseField_gradient_tail variant hμ1 t (by omega)]

theorem rawObjective_tail (variant : MomentumVariant) (μ : ℝ) (t : ℕ) (ht : 3 ≤ t) :
    rawObjective variant μ (position t) = field (amplitude variant μ) potential₁ potential₂
      (position t) := by
  simp only [rawObjective, corrections_tail variant μ t ht, add_zero, baseField]

theorem normalized_objective (variant : MomentumVariant) {μ : ℝ} (hμ1 : μ < 1) :
    ∃ (f : Space 2 2 → ℝ) (K : ℝ≥0),
      ContDiff ℝ ∞ f ∧ (∀ X, 0 ≤ f X) ∧ LipschitzWith K (gradient f) ∧
      (∀ t, gradient f (position t) = requiredGradient variant μ t) ∧
      (∀ t, 3 ≤ t → f (position (t+2)) - f (position t) = 49/240) := by
  let f := rawObjective variant μ
  have hf := rawObjective_contDiff variant μ
  have hd := hf.differentiable (by simp)
  have htail := rawObjective_tail variant μ
  have hdiv : Tendsto (fun t => f (position t)) atTop atTop :=
    objective_divergence _ _ _ potential₁_periodic potential₂_periodic htail
  obtain ⟨B,hB⟩ := bddBelow_range_of_tendsto_atTop_atTop hdiv
  have hbound (t : ℕ) : (B-1)+1 ≤ f (position t) := by
    have := hB (Set.mem_range_self t)
    linarith
  obtain ⟨K,hK⟩ := rawObjective_lipschitz_gradient variant μ
  obtain ⟨G,hG⟩ := rawObjective_bounded_gradient variant μ
  obtain ⟨J,hJ⟩ := clipped_lipschitz_gradient hd hK hG (B-1)
  refine ⟨clipped f (B-1), J, clipped_contDiff hf _, ?_, hJ, ?_, ?_⟩
  · intro X
    exact clippingPrimitive_nonneg _
  · intro t
    rw [clipped_gradient_eq hd (B-1) (hbound t)]
    exact rawObjective_gradient variant hμ1 t
  · intro t ht
    rw [clipped_affine (hbound (t+2)), clipped_affine (hbound t)]
    have hp := objective_period _ _ _ potential₁_periodic potential₂_periodic htail t ht
    dsimp [f]
    linarith

end EF21Muon.Divergence.EF21SignMuon
