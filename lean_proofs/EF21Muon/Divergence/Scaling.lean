import EF21Muon.Divergence.Smooth
import EF21Muon.Divergence.EF21Algorithm
import EF21Muon.Geometry
import LongTails.Proofs.MatrixSmooth

/-! Spectral/nuclear smoothness and the rescaling used in the divergence theorem. -/

noncomputable section

namespace EF21Muon.Divergence

open LongTails.Algorithms.MatrixPolar InnerProductSpace
open scoped ContDiff NNReal

def SpectralSmooth {m n : ℕ} (f : Space m n → ℝ) (L : ℝ) : Prop :=
  ∀ X Y, nuclearNorm (gradient f X - gradient f Y) ≤ L * spectralNorm (X-Y)

theorem SpectralSmooth.mono {m n : ℕ} {f : Space m n → ℝ} {L J : ℝ}
    (hL : SpectralSmooth f L) (hLJ : L ≤ J) : SpectralSmooth f J := by
  intro X Y
  exact (hL X Y).trans (mul_le_mul_of_nonneg_right hLJ (norm_nonneg _))

theorem norm_le_geometry_spectral {m n : ℕ} (X : Space m n) :
    ‖X‖ ≤ geometryConstant m n * spectralNorm X := by
  have h := (operatorEquiv m n).symm.toContinuousLinearMap.le_opNorm (operatorEquiv m n X)
  simp only [ContinuousLinearEquiv.coe_coe, ContinuousLinearEquiv.symm_apply_apply] at h
  change ‖X‖ ≤ ‖(operatorEquiv m n).symm.toContinuousLinearMap‖ * spectralNorm X at h
  exact h.trans (by
    unfold geometryConstant
    have hs : 0 ≤ spectralNorm X := norm_nonneg _
    nlinarith)

theorem spectralSmooth_of_lipschitz_gradient {m n : ℕ} {f : Space m n → ℝ}
    {K : ℝ≥0} (hK : LipschitzWith K (gradient f)) :
    SpectralSmooth f (geometryConstant m n ^ 2 * (K : ℝ)) := by
  intro X Y
  have hκ : 0 ≤ geometryConstant m n := geometryConstant_pos.le
  calc
    _ ≤ geometryConstant m n * ‖gradient f X-gradient f Y‖ := nuclearNorm_le_geometry _
    _ ≤ geometryConstant m n * ((K : ℝ)*‖X-Y‖) := by
      exact mul_le_mul_of_nonneg_left (hK.norm_sub_le X Y) hκ
    _ ≤ geometryConstant m n * ((K : ℝ)*(geometryConstant m n*spectralNorm (X-Y))) := by
      gcongr
      exact norm_le_geometry_spectral _
    _ = _ := by ring

def dilate {E : Type*} [SMul ℝ E] (f : E → ℝ) (a b : ℝ) (X : E) : ℝ := a*f (b • X)

theorem dilate_contDiff {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {f : E → ℝ} (hf : ContDiff ℝ ∞ f) (a b : ℝ) : ContDiff ℝ ∞ (dilate f a b) :=
  contDiff_const.mul (hf.comp (contDiff_const.smul contDiff_id))

theorem dilate_gradient {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [CompleteSpace E] {f : E → ℝ} (hf : Differentiable ℝ f) (a b : ℝ) (X : E) :
    gradient (dilate f a b) X = (a*b) • gradient f (b • X) := by
  have h := (((hf (b • X)).hasFDerivAt.comp X
    ((hasFDerivAt_id X).const_smul b)).const_mul a)
  have hd : fderiv ℝ (dilate f a b) X =
      a • ((fderiv ℝ f (b • X)).comp (b • ContinuousLinearMap.id ℝ E)) := h.fderiv
  simp only [gradient, hd, ContinuousLinearMap.comp_smul, ContinuousLinearMap.comp_id,
    map_smul, smul_smul]

theorem nuclearNorm_smul_le {m n : ℕ} (c : ℝ) (X : Space m n) :
    nuclearNorm (c • X) ≤ |c| * nuclearNorm X := by
  have he : (innerSL ℝ (c • X)).comp (operatorEquiv m n).symm.toContinuousLinearMap =
      c • ((innerSL ℝ X).comp (operatorEquiv m n).symm.toContinuousLinearMap) := by
    ext Y
    simp
  unfold nuclearNorm
  rw [he]
  simpa only [Real.norm_eq_abs] using ContinuousLinearMap.opNorm_smul_le c
    ((innerSL ℝ X).comp (operatorEquiv m n).symm.toContinuousLinearMap)

theorem dilate_spectralSmooth {m n : ℕ} {f : Space m n → ℝ} (hf : Differentiable ℝ f)
    {L a b : ℝ} (hL : SpectralSmooth f L) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    SpectralSmooth (dilate f a b) (a*b^2*L) := by
  intro X Y
  rw [dilate_gradient hf, dilate_gradient hf, ← smul_sub]
  calc
    _ ≤ |a*b| * nuclearNorm (gradient f (b • X)-gradient f (b • Y)) := nuclearNorm_smul_le _ _
    _ = (a*b)*nuclearNorm (gradient f (b • X)-gradient f (b • Y)) := by
      rw [abs_of_nonneg (mul_nonneg ha hb)]
    _ ≤ (a*b)*(L*spectralNorm (b • X-b • Y)) :=
      mul_le_mul_of_nonneg_left (hL _ _) (mul_nonneg ha hb)
    _ = _ := by
      rw [← smul_sub, LongTails.Proofs.MatrixSmooth.spectralNorm_smul, abs_of_nonneg hb]
      ring

namespace EF21SignMuon

theorem scale_reduction (variant : MomentumVariant) (μ : ℝ)
    {f : Space 2 2 → ℝ} (hf : Differentiable ℝ f) {L η L₀ : ℝ}
    (hL : 0 < L) (hη : 0 < η) (hL₀ : 0 < L₀) (hs : SpectralSmooth f L₀) :
    let F := dilate f (L*η^2/L₀) η⁻¹
    SpectralSmooth F L ∧
      (∀ t, (run variant μ η (gradient F) t).x = η • (run variant μ 1 (gradient f) t).x) ∧
      (∀ t, F (run variant μ η (gradient F) t).x =
        (L*η^2/L₀)*f (run variant μ 1 (gradient f) t).x) := by
  let a := L*η^2/L₀
  let c := L*η/L₀
  have ha : 0 < a := by dsimp [a]; positivity
  have hc : 0 < c := by dsimp [c]; positivity
  have hgrad : gradient (dilate f a η⁻¹) = fun X => c • gradient f (η⁻¹ • X) := by
    funext X
    rw [dilate_gradient hf]
    congr 1
    dsimp [a,c]
    field_simp
  have hrun (t : ℕ) : run variant μ η (gradient (dilate f a η⁻¹)) t =
      rescale η c (run variant μ 1 (gradient f) t) := by
    rw [hgrad]
    exact run_scale variant μ hη.ne' hc _ t
  have hcoeff : a*(η⁻¹)^2*L₀ = L := by dsimp [a]; field_simp
  refine ⟨?_, ?_, ?_⟩
  · simpa only [hcoeff] using dilate_spectralSmooth hf hs ha.le (inv_nonneg.mpr hη.le)
  · intro t
    exact congrArg State.x (hrun t)
  · intro t
    rw [hrun]
    simp [dilate, rescale, smul_smul, inv_mul_cancel₀ hη.ne']

end EF21SignMuon

end EF21Muon.Divergence
