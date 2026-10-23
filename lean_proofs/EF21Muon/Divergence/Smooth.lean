import EF21Muon.Divergence.Polar
import Mathlib.Analysis.Calculus.BumpFunction.InnerProduct
import Mathlib.Analysis.Calculus.ContDiff.RCLike
import Mathlib.Analysis.Calculus.Gradient.Basic

/-! Smooth localized corrections with certified gradients and global
Lipschitz gradients. Used to realize finitely many transient gradients. -/

noncomputable section

namespace EF21Muon.Divergence

open InnerProductSpace Filter
open scoped Topology ContDiff NNReal

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E] [CompleteSpace E]

omit [FiniteDimensional ℝ E] in
theorem hasGradientAt_of_differentiable {f : E → ℝ} (hf : Differentiable ℝ f) (x : E) :
    HasGradientAt f (gradient f x) x := (hf x).hasFDerivAt.hasGradientAt

omit [FiniteDimensional ℝ E] in
theorem gradient_add {f g : E → ℝ} (hf : Differentiable ℝ f)
    (hg : Differentiable ℝ g) (x : E) :
    gradient (fun x => f x + g x) x = gradient f x + gradient g x := by
  simp only [gradient, fderiv_fun_add (hf x) (hg x), map_add]

omit [FiniteDimensional ℝ E] in
theorem lipschitz_gradient_of_fderiv {f : E → ℝ} {K : ℝ≥0}
    (hK : LipschitzWith K (fderiv ℝ f)) : LipschitzWith K (gradient f) := by
  apply (lipschitzWith_iff_norm_sub_le).mpr
  intro x y
  simp only [gradient, ← map_sub, LinearIsometryEquiv.norm_map]
  exact hK.norm_sub_le x y

omit [FiniteDimensional ℝ E] in
theorem compact_smooth_lipschitz_gradient {f : E → ℝ}
    (hf : ContDiff ℝ ∞ f) (hcompact : HasCompactSupport f) :
    ∃ K, LipschitzWith K (gradient f) := by
  have hd : ContDiff ℝ 1 (fderiv ℝ f) := hf.fderiv_right (by simp)
  obtain ⟨K,hK⟩ := hd.lipschitzWith_of_hasCompactSupport (hcompact.fderiv ℝ) (by norm_num)
  refine ⟨K, (lipschitzWith_iff_norm_sub_le).mpr ?_⟩
  intro x y
  simp only [gradient, ← map_sub, LinearIsometryEquiv.norm_map]
  exact hK.norm_sub_le x y

omit [FiniteDimensional ℝ E] in
theorem compact_smooth_bounded_gradient {f : E → ℝ}
    (hf : ContDiff ℝ ∞ f) (hcompact : HasCompactSupport f) :
    ∃ G : ℝ≥0, ∀ x, ‖gradient f x‖ ≤ G := by
  have hd : ContDiff ℝ 1 (fderiv ℝ f) := hf.fderiv_right (by simp)
  obtain ⟨C,hC⟩ := (hcompact.fderiv ℝ).exists_bound_of_continuous hd.continuous
  refine ⟨⟨max C 0, le_max_right _ _⟩, ?_⟩
  intro x
  change ‖gradient f x‖ ≤ max C 0
  simpa only [gradient, LinearIsometryEquiv.norm_map] using (hC x).trans (le_max_left C 0)

omit [FiniteDimensional ℝ E] in
theorem gradient_sum {ι : Type*} [Fintype ι] {f : ι → E → ℝ}
    (hf : ∀ i, Differentiable ℝ (f i)) (x : E) :
    gradient (fun x => ∑ i, f i x) x = ∑ i, gradient (f i) x := by
  simp only [gradient, fderiv_fun_sum (fun i _ => hf i x), map_sum]

omit [FiniteDimensional ℝ E] in
theorem scalar_comp_lipschitz_gradient {f df : ℝ → ℝ} (P : E →L[ℝ] ℝ)
    (hf : ∀ x, HasDerivAt f (df x) x) {K : ℝ≥0} (hd : LipschitzWith K df) :
    ∃ L, LipschitzWith L (gradient (f ∘ P)) := by
  have hD (x : E) : fderiv ℝ (f ∘ P) x = df (P x) • P :=
    ((hf (P x)).comp_hasFDerivAt x P.hasFDerivAt).fderiv
  refine ⟨K*‖P‖₊*‖P‖₊, lipschitz_gradient_of_fderiv ?_⟩
  apply (lipschitzWith_iff_norm_sub_le).mpr
  intro x y
  rw [hD, hD, ← sub_smul, norm_smul, Real.norm_eq_abs]
  have h₁ := hd.norm_sub_le (P x) (P y)
  have h₂ : ‖P x-P y‖ ≤ ‖P‖*‖x-y‖ := by
    simpa only [coe_nnnorm] using P.lipschitzWith.norm_sub_le x y
  change |df (P x) - df (P y)| ≤ (K : ℝ)*‖P x-P y‖ at h₁
  calc
    _ ≤ ((K : ℝ)*‖P x-P y‖)*‖P‖ := mul_le_mul_of_nonneg_right h₁ (norm_nonneg _)
    _ ≤ ((K : ℝ)*(‖P‖*‖x-y‖))*‖P‖ := by gcongr
    _ = _ := by simp only [NNReal.coe_mul, coe_nnnorm]; ring

def correctionBump (Z : E) : ContDiffBump Z :=
  ⟨1/100, 1/50, by norm_num, by norm_num⟩

def correction (C Z : E) (x : E) : ℝ :=
  inner ℝ C (x-Z) * correctionBump Z x

omit [FiniteDimensional ℝ E] [CompleteSpace E] in
theorem correction_contDiff (C Z : E) : ContDiff ℝ ∞ (correction C Z) := by
  apply ContDiff.mul
  · exact contDiff_const.inner ℝ (contDiff_id.sub contDiff_const)
  · exact (correctionBump Z).contDiff

omit [CompleteSpace E] in
theorem correction_compact (C Z : E) : HasCompactSupport (correction C Z) := by
  exact (correctionBump Z).hasCompactSupport.mul_left

omit [FiniteDimensional ℝ E] [CompleteSpace E] in
theorem correction_center (C Z : E) : correction C Z Z = 0 := by
  simp [correction]

omit [FiniteDimensional ℝ E] in
theorem correction_gradient_center (C Z : E) : gradient (correction C Z) Z = C := by
  have hg : HasGradientAt (fun x : E => inner ℝ C (x-Z)) C Z := by
    apply hasGradientAt_iff_hasFDerivAt.mpr
    convert ((innerSL ℝ C).hasFDerivAt.comp Z ((hasFDerivAt_id Z).sub_const Z)) using 1 <;>
      ext y <;> simp
  have he : correction C Z =ᶠ[𝓝 Z] (fun x : E => inner ℝ C (x-Z)) := by
    filter_upwards [(correctionBump Z).eventuallyEq_one] with x hx
    simp [correction, hx]
  exact (hg.congr_of_eventuallyEq he).gradient

omit [FiniteDimensional ℝ E] [CompleteSpace E] in
theorem correction_zero {C Z x : E} (hx : (1/50 : ℝ) ≤ dist x Z) : correction C Z x = 0 := by
  simp [correction, (correctionBump Z).zero_of_le_dist hx]

omit [FiniteDimensional ℝ E] in
theorem correction_gradient_zero {C Z x : E} (hx : (1/50 : ℝ) < dist x Z) :
    gradient (correction C Z) x = 0 := by
  have he : correction C Z =ᶠ[𝓝 x] (fun _ : E => (0 : ℝ)) := by
    have hev : ∀ᶠ y in 𝓝 x, (1/50 : ℝ) < dist y Z :=
      (isOpen_lt continuous_const (continuous_id.dist continuous_const)).mem_nhds hx
    filter_upwards [hev] with y hy
    exact correction_zero hy.le
  exact ((hasGradientAt_const x (0 : ℝ)).congr_of_eventuallyEq he).gradient

end EF21Muon.Divergence
