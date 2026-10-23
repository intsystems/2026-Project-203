import EF21Muon.Divergence.Waves
import EF21Muon.Divergence.Plateaus
import EF21Muon.Divergence.EF21Objective
import EF21Muon.Divergence.EF21Momentum

noncomputable section

namespace EF21Muon.Divergence.EF21SignMuon

open LongTails.Algorithms.MatrixPolar InnerProductSpace
open scoped ContDiff NNReal

def potential₁ : ℝ → ℝ := plateauPotential (21/20) (131/200)
def potential₂ : ℝ → ℝ := plateauPotential (7/20) (61/200)
def derivative₁ : ℝ → ℝ := plateauDerivative (21/20) (131/200)
def derivative₂ : ℝ → ℝ := plateauDerivative (7/20) (61/200)

theorem potential₁_hasDeriv (x : ℝ) : HasDerivAt potential₁ (derivative₁ x) x :=
  plateauPotential_hasDeriv _ _ x

theorem potential₂_hasDeriv (x : ℝ) : HasDerivAt potential₂ (derivative₂ x) x :=
  plateauPotential_hasDeriv _ _ x

theorem potential₁_periodic : Function.Periodic potential₁ (21/20) :=
  plateauPotential_periodic (by norm_num) _
theorem potential₂_periodic : Function.Periodic potential₂ (7/20) :=
  plateauPotential_periodic (by norm_num) _
theorem derivative₁_periodic : Function.Periodic derivative₁ (21/20) :=
  plateauDerivative_periodic (by norm_num) _
theorem derivative₂_periodic : Function.Periodic derivative₂ (7/20) :=
  plateauDerivative_periodic (by norm_num) _

theorem derivative₁_plateau_plus {x : ℝ} (hx : |x-131/200| ≤ 1/100) :
    derivative₁ x = 1 := plateauDerivative_plus (by norm_num) hx

theorem derivative₁_plateau_minus {x : ℝ} (hx : |x-7/10| ≤ 1/100) :
    derivative₁ x = -1 := by
  apply plateauDerivative_minus (by norm_num)
  convert hx using 1; norm_num

theorem derivative₂_plateau_plus {x : ℝ} (hx : |x-61/200| ≤ 1/100) :
    derivative₂ x = 1 := plateauDerivative_plus (by norm_num) hx

theorem derivative₂_plateau_minus {x : ℝ} (hx : |x| ≤ 1/100) :
    derivative₂ x = -1 := by
  have h := plateauDerivative_minus (p := (7/20:ℝ)) (c := (61/200:ℝ))
    (x := x+7/20) (by norm_num) (by convert hx using 1; congr 1; ring)
  exact (derivative₂_periodic x).symm.trans h

theorem derivative₁_plus : derivative₁ (131/200) = 1 :=
  derivative₁_plateau_plus (by norm_num)
theorem derivative₁_minus : derivative₁ (7/10) = -1 :=
  derivative₁_plateau_minus (by norm_num)
theorem derivative₂_plus : derivative₂ (61/200) = 1 :=
  derivative₂_plateau_plus (by norm_num)
theorem derivative₂_minus : derivative₂ 0 = -1 :=
  derivative₂_plateau_minus (by norm_num)

def coord (p : Fin 2 × Fin 2) : Space 2 2 →L[ℝ] ℝ := PiLp.proj 2 _ p

@[simp] theorem coord_apply (p : Fin 2 × Fin 2) (X : Space 2 2) : coord p X = X p := rfl

def baseField (A : ℝ) : Space 2 2 → ℝ := field A potential₁ potential₂

def baseGradient (A : ℝ) (X : Space 2 2) : Space 2 2 :=
  ofMatrix !![0, A*derivative₁ (X (0,1)); A*derivative₂ (X (1,0)), -7/12]

theorem baseField_contDiff (A : ℝ) : ContDiff ℝ ∞ (baseField A) := by
  have h₁ : ContDiff ℝ ∞ potential₁ := plateauPotential_contDiff _ _
  have h₂ : ContDiff ℝ ∞ potential₂ := plateauPotential_contDiff _ _
  exact ((contDiff_const.mul (coord (1,1)).contDiff).add
    (contDiff_const.mul (h₁.comp (coord (0,1)).contDiff))).add
    (contDiff_const.mul (h₂.comp (coord (1,0)).contDiff))

theorem baseField_hasGradient (A : ℝ) (X : Space 2 2) :
    HasGradientAt (baseField A) (baseGradient A X) X := by
  have h₀ := ((hasDerivAt_id (X (1,1))).const_mul (-(7/12 : ℝ))).comp_hasFDerivAt
    X (coord (1,1)).hasFDerivAt
  have h₁ := ((potential₁_hasDeriv (X (0,1))).const_mul A).comp_hasFDerivAt
    X (coord (0,1)).hasFDerivAt
  have h₂ := ((potential₂_hasDeriv (X (1,0))).const_mul A).comp_hasFDerivAt
    X (coord (1,0)).hasFDerivAt
  apply hasGradientAt_iff_hasFDerivAt.mpr
  convert (h₀.add h₁).add h₂ using 1
  all_goals
    ext Y
    norm_num [baseField, field, toDual_apply_apply, baseGradient, ofMatrix, matrixEquiv, PiLp.inner_apply,
      Fintype.sum_prod_type, Fin.sum_univ_succ, coord]
    <;> ring

theorem baseField_gradient (A : ℝ) (X : Space 2 2) :
    gradient (baseField A) X = baseGradient A X := (baseField_hasGradient A X).gradient

theorem baseField_lipschitz_gradient (A : ℝ) :
    ∃ K, LipschitzWith K (gradient (baseField A)) := by
  obtain ⟨K₁,hK₁⟩ := plateauDerivative_lipschitz (21/20) (131/200)
  obtain ⟨K₂,hK₂⟩ := plateauDerivative_lipschitz (7/20) (61/200)
  have hd₁ : LipschitzWith (‖A‖₊*K₁) (fun x => A*derivative₁ x) :=
    (lipschitzWith_smul A).comp hK₁
  have hd₂ : LipschitzWith (‖A‖₊*K₂) (fun x => A*derivative₂ x) :=
    (lipschitzWith_smul A).comp hK₂
  obtain ⟨L₀,hL₀⟩ := scalar_comp_lipschitz_gradient (coord (1,1))
    (fun x => (hasDerivAt_id x).const_mul (-(7/12 : ℝ))) (LipschitzWith.const _)
  obtain ⟨L₁,hL₁⟩ := scalar_comp_lipschitz_gradient (coord (0,1))
    (fun x => (potential₁_hasDeriv x).const_mul A) hd₁
  obtain ⟨L₂,hL₂⟩ := scalar_comp_lipschitz_gradient (coord (1,0))
    (fun x => (potential₂_hasDeriv x).const_mul A) hd₂
  have hf₀ : Differentiable ℝ (fun X : Space 2 2 => -(7/12 : ℝ)*X (1,1)) := by fun_prop
  have hf₁ : Differentiable ℝ (fun X : Space 2 2 => A*potential₁ (X (0,1))) := by
    have hp : ContDiff ℝ ∞ potential₁ := plateauPotential_contDiff _ _
    exact (contDiff_const.mul (hp.comp (coord (0,1)).contDiff)).differentiable (by simp)
  have hf₂ : Differentiable ℝ (fun X : Space 2 2 => A*potential₂ (X (1,0))) := by
    have hp : ContDiff ℝ ∞ potential₂ := plateauPotential_contDiff _ _
    exact (contDiff_const.mul (hp.comp (coord (1,0)).contDiff)).differentiable (by simp)
  refine ⟨L₀+L₁+L₂, ?_⟩
  have he : gradient (baseField A) = fun X =>
      gradient (fun X : Space 2 2 => -(7/12 : ℝ)*X (1,1)) X +
      gradient (fun X : Space 2 2 => A*potential₁ (X (0,1))) X +
      gradient (fun X : Space 2 2 => A*potential₂ (X (1,0))) X := by
    funext X
    exact (gradient_add (hf₀.add hf₁) hf₂ X).trans
      (congrArg (fun z => z + gradient (fun X : Space 2 2 => A*potential₂ (X (1,0))) X)
        (gradient_add hf₀ hf₁ X))
  rw [he]
  exact (hL₀.add hL₁).add hL₂

theorem baseGradient_decomposition (A : ℝ) (X : Space 2 2) : baseGradient A X =
    (A*derivative₁ (X (0,1))) • EuclideanSpace.single (0,1) (1:ℝ) +
    (A*derivative₂ (X (1,0))) • EuclideanSpace.single (1,0) (1:ℝ) +
    (-(7/12 : ℝ)) • EuclideanSpace.single (1,1) (1:ℝ) := by
  ext ⟨i,j⟩
  fin_cases i <;> fin_cases j <;> norm_num [baseGradient, ofMatrix, matrixEquiv, EuclideanSpace.single]

theorem baseField_bounded_gradient (A : ℝ) : ∃ G : ℝ≥0, ∀ X, ‖gradient (baseField A) X‖ ≤ G := by
  let B := |A| * (1 + 1) + 7/12
  refine ⟨⟨B, by dsimp [B]; positivity⟩, ?_⟩
  intro X
  rw [baseField_gradient, baseGradient_decomposition]
  have h₁ := plateauDerivative_bounds (21/20) (131/200) (X (0,1))
  have h₂ := plateauDerivative_bounds (7/20) (61/200) (X (1,0))
  have hn := (norm_add_le
    ((A*derivative₁ (X (0,1))) • EuclideanSpace.single (0,1) (1:ℝ) +
      (A*derivative₂ (X (1,0))) • EuclideanSpace.single (1,0) (1:ℝ) : Space 2 2)
    ((-(7/12 : ℝ)) • EuclideanSpace.single (1,1) (1:ℝ))).trans
    (add_le_add (norm_add_le _ _) le_rfl)
  simp only [norm_smul, PiLp.norm_single, Real.norm_eq_abs, abs_one,
    mul_one, abs_mul] at hn
  norm_num at hn
  change |derivative₁ (X (0,1))| ≤ _ at h₁
  change |derivative₂ (X (1,0))| ≤ _ at h₂
  change _ ≤ B
  dsimp [B]
  simp only [neg_smul]
  nlinarith [mul_le_mul_of_nonneg_left h₁ (abs_nonneg A),
    mul_le_mul_of_nonneg_left h₂ (abs_nonneg A)]

end EF21Muon.Divergence.EF21SignMuon
