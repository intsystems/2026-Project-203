import EF21Muon.Divergence.Clipping
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Periodic

/-! Explicit smooth periodic ramps for the article's realization lemma.
Two translated copies of the same periodic bump have equal integrals, so
their difference has zero mean without any unspecified balancing term. -/

noncomputable section

namespace EF21Muon.Divergence

open scoped ContDiff NNReal

def periodicBump (p x : ℝ) : ℝ :=
  Real.smoothTransition ((1/2500 - (p/Real.pi * Real.sin (Real.pi*x/p))^2) / (3/10000))

theorem periodicBump_contDiff (p : ℝ) : ContDiff ℝ ∞ (periodicBump p) := by
  apply Real.smoothTransition.contDiff.comp
  fun_prop

theorem periodicBump_periodic {p : ℝ} (hp : p ≠ 0) :
    Function.Periodic (periodicBump p) p := by
  intro x
  have he : Real.pi*(x+p)/p = Real.pi*x/p + Real.pi := by field_simp
  simp only [periodicBump, he, Real.sin_add_pi]
  congr 1
  ring

theorem periodicBump_bounds (p x : ℝ) :
    0 ≤ periodicBump p x ∧ periodicBump p x ≤ 1 :=
  ⟨Real.smoothTransition.nonneg _, Real.smoothTransition.le_one _⟩

theorem sineSquare_sub_le (x y : ℝ) :
    |Real.sin x ^ 2 - Real.sin y ^ 2| ≤ 2*|x-y| := by
  rw [show Real.sin x ^ 2 - Real.sin y ^ 2 =
    (Real.sin x - Real.sin y)*(Real.sin x + Real.sin y) by ring, abs_mul]
  have hsum : |Real.sin x + Real.sin y| ≤ 2 :=
    (abs_add_le _ _).trans (by linarith [Real.abs_sin_le_one x, Real.abs_sin_le_one y])
  calc
    _ ≤ |x-y| * 2 := mul_le_mul (Real.abs_sin_sub_sin_le x y) hsum
      (abs_nonneg _) (abs_nonneg _)
    _ = _ := by ring

theorem periodicBump_lipschitz (p : ℝ) : ∃ K, LipschitzWith K (periodicBump p) := by
  obtain ⟨S,hS⟩ := smoothTransition_lipschitz
  let K : ℝ≥0 := ⟨(p/Real.pi)^2 * (2*|Real.pi/p|) / (3/10000), by positivity⟩
  have hK : LipschitzWith K (fun x : ℝ =>
      (1/2500 - (p/Real.pi * Real.sin (Real.pi*x/p))^2)/(3/10000)) := by
    apply lipschitzWith_iff_norm_sub_le.mpr
    intro x y
    have hs := sineSquare_sub_le (Real.pi*x/p) (Real.pi*y/p)
    have he : |Real.pi*x/p - Real.pi*y/p| = |Real.pi/p| * |x-y| := by
      rw [← abs_mul]
      congr 1
      ring
    rw [he] at hs
    change |(1/2500 - (p/Real.pi * Real.sin (Real.pi*x/p))^2)/(3/10000) -
      (1/2500 - (p/Real.pi * Real.sin (Real.pi*y/p))^2)/(3/10000)| ≤ _
    have he' : (1/2500 - (p/Real.pi * Real.sin (Real.pi*x/p))^2)/(3/10000) -
      (1/2500 - (p/Real.pi * Real.sin (Real.pi*y/p))^2)/(3/10000) =
      -((p/Real.pi)^2/(3/10000)) *
        (Real.sin (Real.pi*x/p)^2 - Real.sin (Real.pi*y/p)^2) := by ring
    rw [he', abs_mul, abs_neg, abs_of_nonneg (by positivity)]
    have hm := mul_le_mul_of_nonneg_left hs
      (show 0 ≤ (p/Real.pi)^2/(3/10000) by positivity)
    change _ ≤ ((p/Real.pi)^2 * (2*|Real.pi/p|) / (3/10000)) * |x-y|
    calc
      _ ≤ (p/Real.pi)^2/(3/10000) * (2 * (|Real.pi/p| * |x-y|)) := hm
      _ = _ := by ring
  exact ⟨S*K, hS.comp hK⟩

theorem periodicBump_one {p x : ℝ} (hp : 0 < p) (hx : |x| ≤ 1/100) :
    periodicBump p x = 1 := by
  have he : |p/Real.pi * (Real.pi*x/p)| = |x| := by
    congr 1
    field_simp
  have hs : |p/Real.pi * Real.sin (Real.pi*x/p)| ≤ 1/100 := by
    calc
      _ = |p/Real.pi| * |Real.sin (Real.pi*x/p)| := abs_mul _ _
      _ ≤ |p/Real.pi| * |Real.pi*x/p| := by
        exact mul_le_mul_of_nonneg_left Real.abs_sin_le_abs (abs_nonneg _)
      _ = |x| := by rw [← abs_mul, he]
      _ ≤ _ := hx
  apply Real.smoothTransition.one_of_one_le
  have hsq : (p/Real.pi * Real.sin (Real.pi*x/p))^2 ≤ (1/100 : ℝ)^2 := by
    simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) (by norm_num)).mpr hs
  nlinarith

theorem periodicBump_zero {p x : ℝ} (hp : 0 < p)
    (hx : 7/200 ≤ |x|) (hxp : |x| ≤ p/2) : periodicBump p x = 0 := by
  have harg : |Real.pi*x/p| ≤ Real.pi/2 := by
    rw [abs_div, abs_mul, abs_of_pos Real.pi_pos, abs_of_pos hp]
    apply (div_le_iff₀ hp).mpr
    nlinarith [Real.pi_pos]
  have hs := Real.mul_abs_le_abs_sin harg
  have he : (p/Real.pi) * (2/Real.pi * |Real.pi*x/p|) = 2/Real.pi * |x| := by
    rw [abs_div, abs_mul, abs_of_pos Real.pi_pos, abs_of_pos hp]
    field_simp
  have hl : 2/Real.pi * |x| ≤ |p/Real.pi * Real.sin (Real.pi*x/p)| := by
    rw [abs_mul, abs_of_pos (div_pos hp Real.pi_pos)]
    rw [← he]
    exact mul_le_mul_of_nonneg_left hs (by positivity)
  have hlow : 1/50 ≤ |p/Real.pi * Real.sin (Real.pi*x/p)| := by
    have hpi : Real.pi < 7/2 := lt_trans Real.pi_lt_d2 (by norm_num)
    have hc : (1/50 : ℝ) ≤ 2/Real.pi * |x| := by
      rw [div_mul_eq_mul_div]
      apply (le_div_iff₀ Real.pi_pos).mpr
      nlinarith
    exact hc.trans hl
  apply Real.smoothTransition.zero_of_nonpos
  have hsq := (sq_le_sq₀ (by norm_num : (0:ℝ) ≤ 1/50) (abs_nonneg _)).mpr hlow
  rw [sq_abs] at hsq
  nlinarith

def plateauDerivative (p c x : ℝ) : ℝ :=
  periodicBump p (x-c) - periodicBump p (x-(c+9/200))

theorem plateauDerivative_contDiff (p c : ℝ) : ContDiff ℝ ∞ (plateauDerivative p c) :=
  ((periodicBump_contDiff p).comp (contDiff_id.sub contDiff_const)).sub
    ((periodicBump_contDiff p).comp (contDiff_id.sub contDiff_const))

theorem plateauDerivative_periodic {p : ℝ} (hp : p ≠ 0) (c : ℝ) :
    Function.Periodic (plateauDerivative p c) p := by
  intro x
  have h₁ := periodicBump_periodic hp (x-c)
  have h₂ := periodicBump_periodic hp (x-(c+9/200))
  simpa only [plateauDerivative, add_sub_right_comm] using congrArg₂ (· - ·) h₁ h₂

theorem plateauDerivative_bounds (p c x : ℝ) : |plateauDerivative p c x| ≤ 1 := by
  obtain ⟨h₁,h₂⟩ := periodicBump_bounds p (x-c)
  obtain ⟨h₃,h₄⟩ := periodicBump_bounds p (x-(c+9/200))
  exact abs_le.mpr ⟨by dsimp [plateauDerivative]; linarith,
    by dsimp [plateauDerivative]; linarith⟩

theorem plateauDerivative_lipschitz (p c : ℝ) :
    ∃ K, LipschitzWith K (plateauDerivative p c) := by
  obtain ⟨K,hK⟩ := periodicBump_lipschitz p
  refine ⟨K+K, lipschitzWith_iff_norm_sub_le.mpr ?_⟩
  intro x y
  have h₁ := hK.norm_sub_le (x-c) (y-c)
  have h₂ := hK.norm_sub_le (x-(c+9/200)) (y-(c+9/200))
  simp only [sub_sub_sub_cancel_right, Real.norm_eq_abs] at h₁ h₂
  have he : plateauDerivative p c x - plateauDerivative p c y =
      (periodicBump p (x-c) - periodicBump p (y-c)) -
      (periodicBump p (x-(c+9/200)) - periodicBump p (y-(c+9/200))) := by
    dsimp [plateauDerivative]
    ring
  change |plateauDerivative p c x - plateauDerivative p c y| ≤ _
  rw [he]
  exact (abs_sub _ _).trans (by
    simpa only [NNReal.coe_add, Real.norm_eq_abs, add_mul] using add_le_add h₁ h₂)

theorem plateauDerivative_plus {p c x : ℝ} (hp : 11/100 ≤ p)
    (hx : |x-c| ≤ 1/100) : plateauDerivative p c x = 1 := by
  have hx' := abs_le.mp hx
  have hneg : x-(c+9/200) ≤ 0 := by linarith
  have hlow : 7/200 ≤ |x-(c+9/200)| := by rw [abs_of_nonpos hneg]; linarith
  have hupp : |x-(c+9/200)| ≤ p/2 := by rw [abs_of_nonpos hneg]; linarith
  rw [plateauDerivative, periodicBump_one (by linarith) hx,
    periodicBump_zero (by linarith) hlow hupp, sub_zero]

theorem plateauDerivative_minus {p c x : ℝ} (hp : 11/100 ≤ p)
    (hx : |x-(c+9/200)| ≤ 1/100) : plateauDerivative p c x = -1 := by
  have hx' := abs_le.mp hx
  have hpos : 0 ≤ x-c := by linarith
  have hlow : 7/200 ≤ |x-c| := by rw [abs_of_nonneg hpos]; linarith
  have hupp : |x-c| ≤ p/2 := by rw [abs_of_nonneg hpos]; linarith
  rw [plateauDerivative, periodicBump_zero (by linarith) hlow hupp,
    periodicBump_one (by linarith) hx, zero_sub]

theorem plateauDerivative_mean_zero {p : ℝ} (hp : p ≠ 0) (c : ℝ) :
    (∫ x in (0:ℝ)..p, plateauDerivative p c x) = 0 := by
  have hb := (periodicBump_contDiff p).continuous
  have hi (d : ℝ) : (∫ x in (0:ℝ)..p, periodicBump p (x-d)) =
      ∫ x in (0:ℝ)..p, periodicBump p x := by
    rw [intervalIntegral.integral_comp_sub_right]
    convert (periodicBump_periodic hp).intervalIntegral_add_eq (-d) 0 using 1 <;> congr 1 <;> ring
  change (∫ x in (0:ℝ)..p, periodicBump p (x-c) - periodicBump p (x-(c+9/200))) = 0
  have h₁ : Continuous (fun x : ℝ => periodicBump p (x-c)) := by fun_prop
  have h₂ : Continuous (fun x : ℝ => periodicBump p (x-(c+9/200))) := by fun_prop
  rw [intervalIntegral.integral_sub (h₁.intervalIntegrable _ _)
    (h₂.intervalIntegrable _ _), hi, hi, sub_self]

def plateauPotential (p c x : ℝ) : ℝ := ∫ t in (0:ℝ)..x, plateauDerivative p c t

theorem plateauPotential_hasDeriv (p c x : ℝ) :
    HasDerivAt (plateauPotential p c) (plateauDerivative p c x) x := by
  have hc := (plateauDerivative_contDiff p c).continuous
  exact intervalIntegral.integral_hasDerivAt_right (hc.intervalIntegrable _ _)
    hc.aestronglyMeasurable.stronglyMeasurableAtFilter hc.continuousAt

theorem plateauPotential_contDiff (p c : ℝ) : ContDiff ℝ ∞ (plateauPotential p c) := by
  apply contDiff_infty_iff_deriv.mpr
  refine ⟨fun x => (plateauPotential_hasDeriv p c x).differentiableAt, ?_⟩
  have he : deriv (plateauPotential p c) = plateauDerivative p c :=
    funext fun x => (plateauPotential_hasDeriv p c x).deriv
  rw [he]
  exact plateauDerivative_contDiff p c

theorem plateauPotential_periodic {p : ℝ} (hp : p ≠ 0) (c : ℝ) :
    Function.Periodic (plateauPotential p c) p := by
  intro x
  have hc := (plateauDerivative_contDiff p c).continuous
  have h := (plateauDerivative_periodic hp c).intervalIntegral_add_eq_add 0 x
    (fun a b => hc.intervalIntegrable a b)
  simpa only [zero_add, plateauDerivative_mean_zero hp c, add_zero, plateauPotential] using h

end EF21Muon.Divergence
