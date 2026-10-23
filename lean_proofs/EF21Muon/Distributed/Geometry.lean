import EF21Muon.Distributed.Layer
import EF21Muon.Stochastic.Methods

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open LongTails.Proofs.MatrixSmooth
open scoped BigOperators

variable {p : ℕ} {m n : Fin p → ℕ}

abbrev Model (m n : Fin p → ℕ) := ∀ i, Space (m i) (n i)
abbrev States (N : ℕ) (m n : Fin p → ℕ) := ∀ i, LayerState N (m i) (n i)

def gradientMap (g : Model m n) : Model m n →L[ℝ] ℝ :=
  ∑ i, (innerSL ℝ (g i)).comp (ContinuousLinearMap.proj i)

theorem gradientMap_apply (g v : Model m n) : gradientMap g v = ∑ i, inner ℝ (g i) (v i) := by
  simp [gradientMap]

/-- A bound anchored at the starting point suffices for simultaneous descent.
This also accommodates coefficients that depend on the gradient at that point. -/
theorem smooth_upper_at {f : Model m n → ℝ} {grad : Model m n → Model m n} {L : Fin p → ℝ}
    (hg : ∀ x, HasFDerivAt f (gradientMap (grad x)) x)
    (x : Model m n)
    (hlip : ∀ z i, nuclearNorm (grad z i - grad x i) ≤ L i * spectralNorm (z i - x i))
    (y : Model m n) :
    f y ≤ f x + ∑ i, (inner ℝ (grad x i) (y i - x i) +
      L i / 2 * spectralNorm (y i - x i) ^ 2) := by
  let v := y - x
  let F := fun t : ℝ => f (x + t • v)
  let F' := fun t : ℝ => ∑ i, inner ℝ (grad (x + t • v) i) (v i)
  let a := ∑ i, inner ℝ (grad x i) (v i)
  let b := ∑ i, L i / 2 * spectralNorm (v i) ^ 2
  let B := fun t : ℝ => f x + t * a + t ^ 2 * b
  have hF (t : ℝ) : HasDerivAt F (F' t) t := by
    have h := (hg _).comp_hasDerivAt t (((hasDerivAt_id t).smul_const v).const_add x)
    simpa only [F, F', Function.comp_def, id_eq, one_smul, gradientMap_apply] using h
  have hB (t : ℝ) : HasDerivAt B (a + 2 * b * t) t := by
    convert (((hasDerivAt_id t).mul_const a).const_add (f x)).add
      (((hasDerivAt_id t).pow 2).mul_const b) using 1
    · funext u; dsimp [B]
    · dsimp; ring
  have hderiv : ∀ t ∈ Set.Ico (0 : ℝ) 1, F' t ≤ a + 2 * b * t := by
    intro t ht
    have hi (i : Fin p) : inner ℝ (grad (x + t • v) i) (v i) ≤
        inner ℝ (grad x i) (v i) + L i * spectralNorm (v i) ^ 2 * t := by
      have hp := (le_abs_self (inner ℝ (grad (x + t • v) i - grad x i) (v i))).trans
        (inner_le_nuclear_mul_spectral _ _)
      have hs := mul_le_mul_of_nonneg_right (hlip (x + t • v) i)
        (show 0 ≤ spectralNorm (v i) from norm_nonneg _)
      simp only [Pi.add_apply, Pi.smul_apply, add_sub_cancel_left,
        spectralNorm_smul, abs_of_nonneg ht.1] at hs
      rw [inner_sub_left] at hp
      nlinarith only [hp, hs]
    have h := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hi i)
    dsimp only [F', a, b]
    have hb : 2 * b = ∑ i, L i * spectralNorm (v i) ^ 2 := by
      simp only [b, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _; ring
    rw [hb]
    simpa only [Finset.sum_add_distrib, ← Finset.sum_mul] using h
  have h := image_le_of_deriv_right_le_deriv_boundary
    (fun t _ => (hF t).continuousAt.continuousWithinAt)
    (fun t _ => (hF t).hasDerivWithinAt)
    (show F 0 ≤ B 0 by simp [F, B])
    (fun t _ => (hB t).continuousAt.continuousWithinAt)
    (fun t _ => (hB t).hasDerivWithinAt) hderiv
    (show (1 : ℝ) ∈ Set.Icc 0 1 by constructor <;> norm_num)
  simpa [F, B, a, b, v, Finset.sum_add_distrib, add_assoc] using h

/-- The paper's arbitrary-pair layer smoothness implies the needed simultaneous upper bound. -/
theorem smooth_upper {f : Model m n → ℝ} {grad : Model m n → Model m n} {L : Fin p → ℝ}
    (hg : ∀ x, HasFDerivAt f (gradientMap (grad x)) x)
    (hlip : ∀ x y i, nuclearNorm (grad x i - grad y i) ≤ L i * spectralNorm (x i - y i))
    (x y : Model m n) :
    f y ≤ f x + ∑ i, (inner ℝ (grad x i) (y i - x i) +
      L i / 2 * spectralNorm (y i - x i) ^ 2) :=
  smooth_upper_at hg x (fun z i => hlip z x i) y

theorem gradient_continuous {grad : Model m n → Model m n} {L : Fin p → ℝ}
    (hL : ∀ i, 0 ≤ L i)
    (hlip : ∀ x y i, nuclearNorm (grad x i - grad y i) ≤ L i * spectralNorm (x i - y i)) :
    Continuous grad := by
  apply continuous_pi
  intro i
  apply (LipschitzWith.of_dist_le_mul (K := ⟨L i, hL i⟩) ?_).continuous
  intro x y
  simp only [dist_eq_norm]
  exact (frobenius_le_nuclear _).trans ((hlip x y i).trans
    (mul_le_mul_of_nonneg_left ((spectralNorm_le_frobenius _).trans
      (norm_le_pi_norm (x - y) i)) (hL i)))

theorem continuous_nuclearNorm {m n : ℕ} : Continuous (nuclearNorm : Space m n → ℝ) :=
  ((innerSL ℝ).continuous.clm_comp_const (operatorEquiv m n).symm.toContinuousLinearMap).norm

theorem measurable_signCompressor {m n : ℕ} : Measurable (signCompressor : Space m n → Space m n) := by
  apply (PiLp.continuous_toLp 2 (fun _ : Fin m × Fin n => ℝ)).measurable.comp
  apply Measurable.of_eval
  intro i
  have he (j : Fin m × Fin n) : Measurable (fun x : Space m n => x j) :=
    (PiLp.continuous_apply 2 (fun _ : Fin m × Fin n => ℝ) j).measurable
  have hb : Measurable (fun x : Space m n => Compression.signBit (x i)) := by
    exact Measurable.ite (measurableSet_le measurable_const (he i)) measurable_const measurable_const
  have habs (j : Fin m × Fin n) : Measurable (fun x : Space m n => |x j|) := by
    simpa only [Real.norm_eq_abs] using (he j).norm
  exact ((Finset.measurable_sum _ (fun j _ => habs j)).div_const _).mul hb

/-- Geometry of the polar step, before adding the smoothness remainder. -/
theorem direction_bound {N m n : ℕ} (c : Parameters) (hN : 0 < N)
    (hdual : ∀ z : Space m n, nuclearNorm z ≤ c.ρ * ‖z‖)
    (s : LayerState N m n) (g : Space m n) :
    inner ℝ g (nextX c s - s.x) ≤
      3 * c.ρ ^ 2 * c.γ * (globalError s g + compressionError s) -
        c.γ / 4 * nuclearNorm g ^ 2 - c.γ / 4 * nuclearNorm (average s.G) ^ 2 := by
  let G := average s.G
  let M := average s.M
  let a := nuclearNorm G
  let e := nuclearNorm (G - g)
  have hi := (le_abs_self (inner ℝ (G - g) (polar G))).trans (inner_le_nuclear_mul_spectral _ _)
  have hi' := mul_le_mul_of_nonneg_left (polar_mem G) (nuclearNorm_nonneg (G - g))
  rw [inner_sub_left, polar_pairing_eq_nuclear] at hi
  have hp : a - e ≤ inner ℝ g (polar G) := by dsimp [a, e]; linarith only [hi, hi']
  have ht := nuclearNorm_add_le G (g - G)
  rw [add_sub_cancel] at ht
  have hr : nuclearNorm (g - G) = e := by dsimp [e]; rw [← neg_sub G g, nuclearNorm_neg]
  rw [hr] at ht
  have hg2 := mul_self_le_mul_self (nuclearNorm_nonneg g) ht
  have hnorm := two_sq (G - M) (M - g)
  rw [sub_add_sub_cancel, norm_sub_rev G M] at hnorm
  have havg := average_sq_le hN (fun j => s.M j - s.G j)
  rw [average_sub] at havg
  have he2 := mul_self_le_mul_self (nuclearNorm_nonneg _) (hdual (G - g))
  have he3 := mul_le_mul_of_nonneg_left hnorm (sq_nonneg c.ρ)
  have he4 := mul_le_mul_of_nonneg_left havg (by positivity : 0 ≤ 2 * c.ρ ^ 2)
  have heb : e ^ 2 ≤ 2 * c.ρ ^ 2 * (globalError s g + compressionError s) := by
    dsimp [e, globalError, compressionError, average, M, G] at *
    nlinarith only [he2, he3, he4]
  have hγ := c.γ_pos.le
  have hγe := mul_le_mul_of_nonneg_left heb hγ
  have hγg := mul_le_mul_of_nonneg_left hg2 hγ
  have hpair := mul_le_mul_of_nonneg_left hp (mul_nonneg hγ (nuclearNorm_nonneg G))
  have hs := mul_nonneg hγ (sq_nonneg (a / 2 - e))
  have hs2 := mul_nonneg hγ (sq_nonneg (a - e))
  have hstep : nextX c s - s.x = -(c.γ * a) • polar G := by
    dsimp [nextX, serverPoint, LayerState.server, LayerState.x, LayerState.G, a, G]; module
  rw [hstep, inner_smul_right]
  change _ ≤ _ - c.γ / 4 * nuclearNorm g ^ 2 - c.γ / 4 * a ^ 2
  nlinarith only [hγe, hγg, hpair, hs, hs2]

end EF21Muon.Distributed
