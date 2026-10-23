import EF21Muon.Divergence.SignMuon
import EF21Muon.Divergence.MuonSign
import EF21Muon.Divergence.EF21Realization
import EF21Muon.Divergence.Scaling
import EF21Muon.Divergence.HarmlessCycle
import EF21Muon.Divergence.LinearObjective

/-! The article divergence theorems. The EF21 counterexample below has a
globally nonnegative, infinitely differentiable objective, the article's
spectral/nuclear smoothness, and the actual zero-initialized algorithm run. -/

noncomputable section

namespace EF21Muon.Divergence.EF21SignMuon

open LongTails.Algorithms.MatrixPolar Filter
open scoped ContDiff NNReal

theorem normalized_run (variant : MomentumVariant) {μ : ℝ} (hμ : 0 ≤ μ) (hμ1 : μ < 1)
    {f : Space 2 2 → ℝ} (hgrad : ∀ t, gradient f (position t) = requiredGradient variant μ t)
    (t : ℕ) : run variant μ 1 (gradient f) t =
      ⟨position t, momentumSeq variant μ t, estimate t⟩ := by
  apply run_of_prescribed variant μ (gradient f) (momentumSeq variant μ) rfl
  · intro s
    rw [hgrad s]
    exact requiredGradient_rec variant hμ1 s
  · exact prescribed_polar variant hμ hμ1

/-- Lemma `lem:efsm_realize` for the article's actual plateau-ramp field
and its three localized corrections, before the bounded-below modification. -/
theorem plateau_realization (variant : MomentumVariant) {μ : ℝ}
    (hμ : 0 ≤ μ) (hμ1 : μ < 1) :
    ContDiff ℝ ∞ (rawObjective variant μ) ∧
      (∃ K, LipschitzWith K (gradient (rawObjective variant μ))) ∧
      (∀ t, run variant μ 1 (gradient (rawObjective variant μ)) t =
        ⟨position t, momentumSeq variant μ t, estimate t⟩) ∧
      (∀ t, 3 ≤ t → rawObjective variant μ (position (t+2)) -
        rawObjective variant μ (position t) = 49/240) := by
  refine ⟨rawObjective_contDiff variant μ, rawObjective_lipschitz_gradient variant μ,
    normalized_run variant hμ hμ1 (rawObjective_gradient variant hμ1), ?_⟩
  exact objective_period _ _ _ potential₁_periodic potential₂_periodic
    (rawObjective_tail variant μ)

def smoothnessScale (K : ℝ≥0) : ℝ := geometryConstant 2 2 ^ 2 * ((K : ℝ)+1)

theorem smoothnessScale_pos (K : ℝ≥0) : 0 < smoothnessScale K := by
  have hκ : 0 < geometryConstant 2 2 := geometryConstant_pos
  unfold smoothnessScale
  positivity

def divergenceRate (K : ℝ≥0) : ℝ := 49 / (240*smoothnessScale K)

theorem divergenceRate_pos (K : ℝ≥0) : 0 < divergenceRate K := by
  have h := smoothnessScale_pos K
  unfold divergenceRate
  positivity

/-- Theorem `th:ef_div`, for both momentum variants and every `0 ≤ μ < 1`.
The objective is nonnegative everywhere; in particular it is bounded below. -/
theorem divergence (variant : MomentumVariant) {μ L η : ℝ}
    (hμ : 0 ≤ μ) (hμ1 : μ < 1) (hL : 0 < L) (hη : 0 < η) :
    ∃ (F : Space 2 2 → ℝ) (c : ℝ),
      ContDiff ℝ ∞ F ∧ (∀ X, 0 ≤ F X) ∧ SpectralSmooth F L ∧ 0 < c ∧
      (∀ t, 3 ≤ t →
        F (run variant μ η (gradient F) (t+2)).x -
          F (run variant μ η (gradient F) t).x = c*L*η^2) ∧
      Tendsto (fun t => F (run variant μ η (gradient F) t).x) atTop atTop := by
  obtain ⟨f,K,hf,hfnonneg,hK,hgrad,hperiod⟩ := normalized_objective variant hμ1
  let L₀ := smoothnessScale K
  have hL₀ : 0 < L₀ := smoothnessScale_pos K
  have hs : SpectralSmooth f L₀ := by
    apply (spectralSmooth_of_lipschitz_gradient hK).mono
    dsimp [L₀, smoothnessScale]
    nlinarith [sq_nonneg (geometryConstant 2 2)]
  let F := dilate f (L*η^2/L₀) η⁻¹
  let c := divergenceRate K
  have hscale := scale_reduction variant μ (hf.differentiable (by simp)) hL hη hL₀ hs
  have hrun (t : ℕ) : f (run variant μ 1 (gradient f) t).x = f (position t) := by
    rw [normalized_run variant hμ hμ1 hgrad]
  have hc : 0 < c := divergenceRate_pos K
  have hvalues (t : ℕ) : F (run variant μ η (gradient F) t).x =
      (L*η^2/L₀)*f (position t) := (hscale.2.2 t).trans (congrArg _ (hrun t))
  have hperiodF (t : ℕ) (ht : 3 ≤ t) :
      F (run variant μ η (gradient F) (t+2)).x -
        F (run variant μ η (gradient F) t).x = c*L*η^2 := by
    rw [hvalues, hvalues, ← mul_sub, hperiod t ht]
    dsimp [c, divergenceRate, L₀]
    ring
  refine ⟨F,c,dilate_contDiff hf _ _, ?_, hscale.1, hc, hperiodF, ?_⟩
  · intro X
    exact mul_nonneg (by positivity) (hfnonneg _)
  · apply tendsto_atTop_of_twoStep (mul_pos (mul_pos hc hL) (sq_pos_of_pos hη)) 3
    intro t ht
    have h := hperiodF t ht
    linarith

end EF21Muon.Divergence.EF21SignMuon
