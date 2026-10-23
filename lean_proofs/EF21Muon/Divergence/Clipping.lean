import EF21Muon.Divergence.Smooth
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-! A smooth, bounded-below modification that preserves the objective and
gradient above a threshold, up to a single additive objective constant. -/

noncomputable section

namespace EF21Muon.Divergence

open Filter Set InnerProductSpace
open scoped Topology ContDiff NNReal

theorem smoothTransition_lipschitz : ∃ K, LipschitzWith K Real.smoothTransition := by
  have hσ : ContDiff ℝ ∞ Real.smoothTransition := Real.smoothTransition.contDiff
  have hs : HasCompactSupport (fderiv ℝ Real.smoothTransition) := by
    apply HasCompactSupport.of_support_subset_isCompact (isCompact_Icc : IsCompact (Icc (0:ℝ) 1))
    intro x hx
    by_contra hn
    have hcases : x < 0 ∨ 1 < x := by simpa only [mem_Icc, not_and_or, not_le] using hn
    have he : ∃ c : ℝ, Real.smoothTransition =ᶠ[𝓝 x] (fun _ => c) := by
      rcases hcases with hl | hr
      · refine ⟨0, ?_⟩
        filter_upwards [Iio_mem_nhds hl] with y hy
        exact Real.smoothTransition.zero_of_nonpos (le_of_lt hy)
      · refine ⟨1, ?_⟩
        filter_upwards [Ioi_mem_nhds hr] with y hy
        exact Real.smoothTransition.one_of_one_le (le_of_lt hy)
    obtain ⟨c,hc⟩ := he
    have hz : fderiv ℝ Real.smoothTransition x = 0 :=
      (HasFDerivAt.congr_of_eventuallyEq (hasFDerivAt_const c x) hc).fderiv
    exact hx hz
  obtain ⟨C,hC⟩ := hs.exists_bound_of_continuous
    (hσ.continuous_fderiv (by simp))
  refine ⟨⟨max C 0, le_max_right _ _⟩, ?_⟩
  apply lipschitzWith_of_nnnorm_fderiv_le (hσ.differentiable (by simp))
  intro x
  change ‖fderiv ℝ Real.smoothTransition x‖ ≤ max C 0
  exact (hC x).trans (le_max_left _ _)

def clippingPrimitive (x : ℝ) : ℝ := ∫ t in (0:ℝ)..x, Real.smoothTransition t

theorem clippingPrimitive_hasDeriv (x : ℝ) :
    HasDerivAt clippingPrimitive (Real.smoothTransition x) x := by
  exact intervalIntegral.integral_hasDerivAt_right
    (Real.smoothTransition.continuous.intervalIntegrable 0 x)
    Real.smoothTransition.continuous.aestronglyMeasurable.stronglyMeasurableAtFilter
    Real.smoothTransition.continuousAt

theorem clippingPrimitive_contDiff : ContDiff ℝ ∞ clippingPrimitive := by
  apply contDiff_infty_iff_deriv.mpr
  constructor
  · intro x; exact (clippingPrimitive_hasDeriv x).differentiableAt
  · have he : deriv clippingPrimitive = Real.smoothTransition :=
      funext fun x => (clippingPrimitive_hasDeriv x).deriv
    rw [he]
    exact Real.smoothTransition.contDiff

theorem clippingPrimitive_nonneg (x : ℝ) : 0 ≤ clippingPrimitive x := by
  by_cases hx : 0 ≤ x
  · exact intervalIntegral.integral_nonneg_of_forall hx Real.smoothTransition.nonneg
  · have hx0 : x ≤ 0 := le_of_not_ge hx
    have he : clippingPrimitive x = 0 := by
      unfold clippingPrimitive
      calc
        _ = ∫ _t in (0:ℝ)..x, (0:ℝ) := by
          apply intervalIntegral.integral_congr
          intro y hy
          have hh : y ∈ Icc x 0 := by simpa only [uIcc_of_ge hx0] using hy
          exact Real.smoothTransition.zero_of_nonpos hh.2
        _ = 0 := by simp
    rw [he]

theorem clippingPrimitive_affine {x : ℝ} (hx : 1 ≤ x) :
    clippingPrimitive x = x + (clippingPrimitive 1 - 1) := by
  have hadd : (∫ t in (0:ℝ)..1, Real.smoothTransition t) +
      (∫ t in (1:ℝ)..x, Real.smoothTransition t) =
      (∫ t in (0:ℝ)..x, Real.smoothTransition t) := intervalIntegral.integral_add_adjacent_intervals
    (Real.smoothTransition.continuous.intervalIntegrable 0 1)
    (Real.smoothTransition.continuous.intervalIntegrable 1 x)
  have hi : (∫ t in (1:ℝ)..x, Real.smoothTransition t) = x-1 := by
    calc
      _ = ∫ _t in (1:ℝ)..x, (1:ℝ) := by
        apply intervalIntegral.integral_congr
        intro y hy
        exact Real.smoothTransition.one_of_one_le (by
          have hh : y ∈ Icc 1 x := by simpa only [uIcc_of_le hx] using hy
          exact hh.1)
      _ = _ := by simp
  change (∫ t in (0:ℝ)..x, Real.smoothTransition t) = _
  rw [← hadd, hi]
  unfold clippingPrimitive
  ring

section Composition

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

def clipped (f : E → ℝ) (a : ℝ) (x : E) : ℝ := clippingPrimitive (f x-a)

omit [CompleteSpace E] in
theorem clipped_contDiff {f : E → ℝ} (hf : ContDiff ℝ ∞ f) (a : ℝ) :
    ContDiff ℝ ∞ (clipped f a) := clippingPrimitive_contDiff.comp (hf.sub contDiff_const)

theorem clipped_gradient {f : E → ℝ} (hf : Differentiable ℝ f) (a : ℝ) (x : E) :
    gradient (clipped f a) x = Real.smoothTransition (f x-a) • gradient f x := by
  have h := (clippingPrimitive_hasDeriv (f x-a)).comp_hasFDerivAt x
    ((hf x).hasFDerivAt.sub_const a)
  change gradient (clippingPrimitive ∘ fun x => f x-a) x = _
  simp only [gradient, h.fderiv, map_smul]

theorem clipped_gradient_eq {f : E → ℝ} (hf : Differentiable ℝ f) (a : ℝ)
    {x : E} (hx : a+1 ≤ f x) : gradient (clipped f a) x = gradient f x := by
  rw [clipped_gradient hf, Real.smoothTransition.one_of_one_le (by linarith), one_smul]

omit [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem clipped_affine {f : E → ℝ} {a : ℝ} {x : E} (hx : a+1 ≤ f x) :
    clipped f a x = f x + (clippingPrimitive 1-1-a) := by
  unfold clipped
  rw [clippingPrimitive_affine (by linarith)]
  ring

theorem clipped_lipschitz_gradient {f : E → ℝ} (hf : Differentiable ℝ f)
    {K G : ℝ≥0} (hK : LipschitzWith K (gradient f))
    (hG : ∀ x, ‖gradient f x‖ ≤ G) (a : ℝ) :
    ∃ L, LipschitzWith L (gradient (clipped f a)) := by
  have hfLip : LipschitzWith G f := by
    apply lipschitzWith_of_nnnorm_fderiv_le hf
    intro x
    change ‖fderiv ℝ f x‖ ≤ (G : ℝ)
    have he : ‖gradient f x‖ = ‖fderiv ℝ f x‖ := by
      simp only [gradient, LinearIsometryEquiv.norm_map]
    rw [← he]
    exact hG x
  obtain ⟨S,hS⟩ := smoothTransition_lipschitz
  refine ⟨K+S*G*G, (lipschitzWith_iff_norm_sub_le).mpr ?_⟩
  intro x y
  rw [clipped_gradient hf, clipped_gradient hf]
  let sx := Real.smoothTransition (f x-a)
  let sy := Real.smoothTransition (f y-a)
  have hsx : |sx| ≤ 1 := by
    dsimp [sx]
    rw [abs_of_nonneg (Real.smoothTransition.nonneg _)]
    exact Real.smoothTransition.le_one _
  have hsxy : |sx-sy| ≤ (S : ℝ)*(G : ℝ)*‖x-y‖ := by
    have h₁ := hS.norm_sub_le (f x-a) (f y-a)
    have h₂ := hfLip.norm_sub_le x y
    simp only [Real.norm_eq_abs] at h₂
    simp only [sub_sub_sub_cancel_right, Real.norm_eq_abs] at h₁
    change |sx-sy| ≤ _ at h₁
    calc
      _ ≤ (S : ℝ)*|f x-f y| := h₁
      _ ≤ (S : ℝ)*((G : ℝ)*‖x-y‖) := by gcongr
      _ = _ := by ring
  have he : sx • gradient f x - sy • gradient f y =
      sx • (gradient f x-gradient f y) + (sx-sy) • gradient f y := by module
  change ‖sx • gradient f x - sy • gradient f y‖ ≤ _
  rw [he]
  calc
    _ ≤ ‖sx • (gradient f x-gradient f y)‖ + ‖(sx-sy) • gradient f y‖ := norm_add_le _ _
    _ = |sx| * ‖gradient f x-gradient f y‖ + |sx-sy| * ‖gradient f y‖ := by
      simp only [norm_smul, Real.norm_eq_abs]
    _ ≤ 1*((K : ℝ)*‖x-y‖) + ((S : ℝ)*(G : ℝ)*‖x-y‖)*(G : ℝ) := by
      gcongr
      · exact hK.norm_sub_le x y
      · exact hG y
    _ = _ := by simp only [NNReal.coe_add, NNReal.coe_mul]; ring

end Composition

end EF21Muon.Divergence
