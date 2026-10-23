import EF21Muon.Distributed.Geometry

/-! Geometry for the article's arbitrary-pair generalized smoothness assumption
and constant-radius polar step. No constant global Lipschitz bound is assumed. -/

noncomputable section

namespace EF21Muon.Distributed

open LongTails.Algorithms.MatrixPolar
open LongTails.Proofs.MatrixSmooth
open EF21Muon.Stochastic
open scoped BigOperators

variable {p : ℕ} {m n : Fin p → ℕ}

/-- The coefficient is evaluated at the first argument, as in the article. -/
def GeneralizedSmooth (grad : Model m n → Model m n) (L₀ L₁ : Fin p → ℝ) : Prop :=
  ∀ x y i, nuclearNorm (grad x i - grad y i) ≤
    (L₀ i + L₁ i * nuclearNorm (grad x i)) * spectralNorm (x i - y i)

theorem generalized_gradient_continuous {grad : Model m n → Model m n}
    {L₀ L₁ : Fin p → ℝ} (hs : GeneralizedSmooth grad L₀ L₁)
    (h₀ : ∀ i, 0 ≤ L₀ i) (h₁ : ∀ i, 0 ≤ L₁ i) : Continuous grad := by
  apply continuous_pi
  intro i
  apply continuous_iff_continuousAt.mpr
  intro x
  let C := L₀ i + L₁ i*nuclearNorm (grad x i)
  have hC : 0 ≤ C := add_nonneg (h₀ i) (mul_nonneg (h₁ i) (nuclearNorm_nonneg _))
  apply continuousAt_of_locally_lipschitz (r := 1) (by norm_num) C
  intro y _
  simp only [dist_eq_norm]
  have h := hs x y i
  rw [← neg_sub (grad y i) (grad x i), nuclearNorm_neg,
    ← neg_sub (y i) (x i)] at h
  simp only [spectralNorm, map_neg, norm_neg] at h
  have hb : spectralNorm (y i-x i) ≤ ‖y-x‖ :=
    (spectralNorm_le_frobenius _).trans (norm_le_pi_norm (y-x) i)
  exact (frobenius_le_nuclear _).trans
    (h.trans (mul_le_mul_of_nonneg_left hb hC))

theorem generalized_smooth_upper {f : Model m n → ℝ}
    {grad : Model m n → Model m n} {L₀ L₁ : Fin p → ℝ}
    (hg : ∀ x, HasFDerivAt f (gradientMap (grad x)) x)
    (hs : GeneralizedSmooth grad L₀ L₁) (x y : Model m n) :
    f y ≤ f x + ∑ i, (inner ℝ (grad x i) (y i-x i) +
      (L₀ i + L₁ i * nuclearNorm (grad x i))/2 * spectralNorm (y i-x i)^2) := by
  apply smooth_upper_at hg x
  intro z i
  have h := hs x z i
  rw [← neg_sub (grad z i) (grad x i), nuclearNorm_neg,
    ← neg_sub (z i) (x i)] at h
  simpa only [spectralNorm, map_neg, norm_neg] using h

/-- The unit spectral-ball oracle, including the convention `polar 0 = 0`. -/
def radiusStep (η : Fin p → ℝ) (x G : Model m n) : Model m n :=
  fun i => x i - η i • polar (G i)

theorem radiusStep_distance (η : Fin p → ℝ) (hη : ∀ i, 0 ≤ η i)
    (x G : Model m n) (i : Fin p) :
    spectralNorm (radiusStep η x G i - x i) ≤ η i := by
  have he : radiusStep η x G i - x i = -(η i) • polar (G i) := by
    dsimp [radiusStep]
    module
  rw [he, spectralNorm_smul, abs_neg, abs_of_nonneg (hη i)]
  simpa using mul_le_mul_of_nonneg_left (polar_mem (G i)) (hη i)

theorem radius_direction_bound {a b : ℕ} (g G : Space a b) :
    nuclearNorm g - 2 * nuclearNorm (G-g) ≤ inner ℝ g (polar G) := by
  have hi := (le_abs_self (inner ℝ (G-g) (polar G))).trans
    (inner_le_nuclear_mul_spectral _ _)
  have hp := mul_le_mul_of_nonneg_left (polar_mem G) (nuclearNorm_nonneg (G-g))
  rw [inner_sub_left, polar_pairing_eq_nuclear] at hi
  have ht := nuclearNorm_add_le G (g-G)
  rw [add_sub_cancel, ← neg_sub G g, nuclearNorm_neg] at ht
  nlinarith only [hi, hp, ht]

/-- One step of the constant-radius method, with first-order tracking errors. -/
theorem generalized_radius_descent {f : Model m n → ℝ}
    {grad : Model m n → Model m n} {L₀ L₁ : Fin p → ℝ}
    (hg : ∀ x, HasFDerivAt f (gradientMap (grad x)) x)
    (hs : GeneralizedSmooth grad L₀ L₁)
    (h₀ : ∀ i, 0 ≤ L₀ i) (h₁ : ∀ i, 0 ≤ L₁ i)
    (η : Fin p → ℝ) (hη : ∀ i, 0 ≤ η i) (x G : Model m n) :
    f (radiusStep η x G) ≤ f x + ∑ i,
      (-η i * nuclearNorm (grad x i) + 2*η i*nuclearNorm (G i-grad x i) +
        η i^2/2 * (L₀ i + L₁ i * nuclearNorm (grad x i))) := by
  apply (generalized_smooth_upper hg hs x (radiusStep η x G)).trans
  apply add_le_add le_rfl
  apply Finset.sum_le_sum
  intro i _
  have hd := mul_le_mul_of_nonneg_left (radius_direction_bound (grad x i) (G i)) (hη i)
  have he : radiusStep η x G i - x i = -(η i) • polar (G i) := by
    dsimp [radiusStep]
    module
  rw [he, inner_smul_right]
  have hs2 : spectralNorm (radiusStep η x G i - x i)^2 ≤ η i^2 :=
    (sq_le_sq₀ (norm_nonneg _) (hη i)).mpr (radiusStep_distance η hη x G i)
  have hc : 0 ≤ (L₀ i + L₁ i*nuclearNorm (grad x i))/2 := by
    exact div_nonneg (add_nonneg (h₀ i) (mul_nonneg (h₁ i) (nuclearNorm_nonneg _)))
      (by norm_num)
  have hm := mul_le_mul_of_nonneg_left hs2 hc
  rw [he] at hm
  nlinarith only [hd, hm]

/-- Separating momentum tracking and compression tracking gives the two
first-moment errors used in the generalized stochastic analysis. -/
theorem generalized_tracking_descent {f : Model m n → ℝ}
    {grad : Model m n → Model m n} {L₀ L₁ ρ : Fin p → ℝ}
    (hg : ∀ x, HasFDerivAt f (gradientMap (grad x)) x)
    (hs : GeneralizedSmooth grad L₀ L₁)
    (h₀ : ∀ i, 0 ≤ L₀ i) (h₁ : ∀ i, 0 ≤ L₁ i) (hρ : ∀ i, 0 ≤ ρ i)
    (hdual : ∀ i (z : Space (m i) (n i)), nuclearNorm z ≤ ρ i*‖z‖)
    (η : Fin p → ℝ) (hη : ∀ i, 0 ≤ η i) (x M G : Model m n) :
    f (radiusStep η x G) ≤ f x + ∑ i,
      (-η i*nuclearNorm (grad x i) +
        2*η i*ρ i*(‖M i-grad x i‖ + ‖G i-M i‖) +
        η i^2/2*(L₀ i+L₁ i*nuclearNorm (grad x i))) := by
  apply (generalized_radius_descent hg hs h₀ h₁ η hη x G).trans
  apply add_le_add le_rfl
  apply Finset.sum_le_sum
  intro i _
  have he : nuclearNorm (G i-grad x i) ≤
      ρ i*(‖M i-grad x i‖ + ‖G i-M i‖) := by
    have hn := norm_add_le (M i-grad x i) (G i-M i)
    have hid : M i-grad x i + (G i-M i) = G i-grad x i := by module
    rw [hid] at hn
    exact (hdual i _).trans (mul_le_mul_of_nonneg_left hn (hρ i))
  have hm := mul_le_mul_of_nonneg_left he
    (show 0 ≤ 2*η i from mul_nonneg (by norm_num) (hη i))
  nlinarith only [hm]

/-- The lower bound on the objective controls a weighted sum of gradient
norms. This form has no division by an `L₁` coefficient and also covers `L₁=0`. -/
theorem generalized_gradient_self_bound {f : Model m n → ℝ}
    {grad : Model m n → Model m n} {L₀ L₁ : Fin p → ℝ} {fmin : ℝ}
    (hg : ∀ x, HasFDerivAt f (gradientMap (grad x)) x)
    (hs : GeneralizedSmooth grad L₀ L₁)
    (h₀ : ∀ i, 0 ≤ L₀ i) (h₁ : ∀ i, 0 ≤ L₁ i)
    (hlower : ∀ y, fmin ≤ f y)
    (η : Fin p → ℝ) (hη : ∀ i, 0 ≤ η i) (hsmall : ∀ i, η i*L₁ i ≤ 1)
    (x : Model m n) :
    (∑ i, η i*nuclearNorm (grad x i)) ≤
      2*(f x-fmin) + ∑ i, η i^2*L₀ i := by
  have hd := generalized_radius_descent hg hs h₀ h₁ η hη x (grad x)
  have hz (i : Fin p) : nuclearNorm (0 : Space (m i) (n i)) = 0 := by
    unfold nuclearNorm
    have he : (innerSL ℝ (0 : Space (m i) (n i))).comp
        (operatorEquiv (m i) (n i)).symm.toContinuousLinearMap = 0 := by
      ext z
      simp
    rw [he]
    exact @norm_zero ((EuclideanSpace ℝ (Fin (n i)) →L[ℝ]
      EuclideanSpace ℝ (Fin (m i))) →L[ℝ] ℝ) _
  simp only [sub_self, hz, mul_zero, add_zero] at hd
  have hterm (i : Fin p) :
      -η i*nuclearNorm (grad x i) + η i^2/2*(L₀ i+L₁ i*nuclearNorm (grad x i)) ≤
        -(η i*nuclearNorm (grad x i))/2 + η i^2*L₀ i/2 := by
    have hm := mul_le_mul_of_nonneg_left (hsmall i)
      (mul_nonneg (hη i) (nuclearNorm_nonneg (grad x i)))
    nlinarith only [hm]
  have hsum := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hterm i)
  have hl := hlower (radiusStep η x (grad x))
  have he : (∑ i, (-(η i*nuclearNorm (grad x i))/2 + η i^2*L₀ i/2)) =
      -(∑ i, η i*nuclearNorm (grad x i))/2 + (∑ i, η i^2*L₀ i)/2 := by
    simp only [Finset.sum_add_distrib, ← Finset.sum_div, Finset.sum_neg_distrib]
  rw [he] at hsum
  linarith only [hd, hsum, hl]

/-- A weighted gradient bound for any positive upper bound `M` on `wᵢ L₁ᵢ`.
In particular, the bound is usable when all generalized coefficients vanish. -/
theorem generalized_weighted_gradient_self_bound {f : Model m n → ℝ}
    {grad : Model m n → Model m n} {L₀ L₁ : Fin p → ℝ} {fmin M : ℝ}
    (hg : ∀ x, HasFDerivAt f (gradientMap (grad x)) x)
    (hs : GeneralizedSmooth grad L₀ L₁)
    (h₀ : ∀ i, 0 ≤ L₀ i) (h₁ : ∀ i, 0 ≤ L₁ i)
    (hlower : ∀ y, fmin ≤ f y)
    (w : Fin p → ℝ) (hw : ∀ i, 0 ≤ w i) (hM : 0 < M)
    (hbound : ∀ i, w i*L₁ i ≤ M) (x : Model m n) :
    (∑ i, w i*nuclearNorm (grad x i)) ≤
      2*M*(f x-fmin) + (∑ i, w i^2*L₀ i)/M := by
  have hsmall (i : Fin p) : (w i/M)*L₁ i ≤ 1 := by
    rw [div_mul_eq_mul_div]
    exact (div_le_one hM).mpr (hbound i)
  have hd := generalized_gradient_self_bound hg hs h₀ h₁ hlower
    (fun i => w i/M) (fun i => div_nonneg (hw i) hM.le) hsmall x
  have he₁ : (∑ i, (w i/M)*nuclearNorm (grad x i)) =
      (∑ i, w i*nuclearNorm (grad x i))/M := by
    simp only [div_mul_eq_mul_div, ← Finset.sum_div]
  have he₂ : (∑ i, (w i/M)^2*L₀ i) = (∑ i, w i^2*L₀ i)/M^2 := by
    simp only [div_pow, div_mul_eq_mul_div, ← Finset.sum_div]
  rw [he₁, he₂] at hd
  have hm := mul_le_mul_of_nonneg_right hd hM.le
  rw [div_mul_cancel₀ _ hM.ne'] at hm
  convert hm using 1
  field_simp

end EF21Muon.Distributed
