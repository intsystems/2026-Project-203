import EF21Muon.Nesterov.AveragedConvergence
import EF21Muon.LongTails.Profiles

/-! The Nesterov profile with worker averaging recovered in its leading
variance term. All comparisons refer to the same problem, step scale and
trajectory score as the previously verified profile. -/

noncomputable section

namespace EF21Muon.LongTails

open MeasureTheory
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def averagedNesterovProfile (P : Problem N m n μ c) {g : ℝ} (hg : 0 < g) (x : Model m n) : Profile where
  a := (P.f x-P.fmin)/g+∑ i, 2*Nesterov.dWeight (c i)*‖P.grad x i‖^2
  h := ∑ i, (6*Nesterov.dWeight (c i)*‖P.grad x i‖^2+
    6*Nesterov.Averaged.rWeight (c i)*average (fun j => ‖P.localGrad j x i‖^2))
  s₁ := ∑ i, 2*Nesterov.dWeight (c i)/N*P.σ i^2
  s₂ := ∑ i, (Nesterov.dWeight (c i)/N+Nesterov.Averaged.rWeight (c i))*P.σ i^2
  s₃ := ∑ i, (6*Nesterov.dWeight (c i)/N+6*Nesterov.Averaged.rWeight (c i))*P.σ i^2
  a_nonneg := add_nonneg (div_nonneg (sub_nonneg.mpr (P.lower_bound x)) hg.le)
    (Finset.sum_nonneg (fun i _ => by unfold Nesterov.dWeight; positivity))
  h_nonneg := Finset.sum_nonneg (fun i _ => add_nonneg
    (by unfold Nesterov.dWeight; positivity)
    (mul_nonneg (mul_nonneg (by norm_num) (Nesterov.Averaged.weights_nonneg (c i)).2.2.2.1)
      (Distributed.average_nonneg (fun _ => sq_nonneg _))))
  s₁_nonneg := Finset.sum_nonneg (fun i _ => by unfold Nesterov.dWeight; positivity)
  s₂_nonneg := Finset.sum_nonneg (fun i _ => mul_nonneg
    (add_nonneg (by unfold Nesterov.dWeight; positivity)
      (Nesterov.Averaged.weights_nonneg (c i)).2.2.2.1) (sq_nonneg _))
  s₃_nonneg := Finset.sum_nonneg (fun i _ => mul_nonneg
    (add_nonneg (by unfold Nesterov.dWeight; positivity)
      (mul_nonneg (by norm_num) (Nesterov.Averaged.weights_nonneg (c i)).2.2.2.1)) (sq_nonneg _))

theorem averaged_nesterov_initial_profile (P : Problem N m n μ c) {g β : ℝ}
    (hg : 0 < g) (hβpos : 0 < β) (hβ : ∀ i, (c i).β = β)
    (hγ : ∀ i, (c i).γ = g*β) (x : Model m n) :
    Nesterov.Averaged.smoothPotential P (Nesterov.initial x) =
      g*((averagedNesterovProfile P hg x).a+(averagedNesterovProfile P hg x).h*β^2) := by
  have he : Nesterov.Averaged.smoothPotential P (Nesterov.initial x) = P.f x-P.fmin+
      ∑ i, (c i).γ*(Nesterov.Averaged.eWeight (c i)*‖P.grad x i‖^2+
        Nesterov.Averaged.lWeight (c i)*average (fun j => ‖P.localGrad j x i‖^2)) := by
    have hx : Nesterov.points (Nesterov.initial x : Nesterov.States N m n) = x := rfl
    simp only [Nesterov.Averaged.smoothPotential, hx]
    congr 1
    apply Finset.sum_congr rfl
    intro i _
    simp [Nesterov.initial, Nesterov.start, LayerState.start, Nesterov.Averaged.energy,
      globalError, localError, Nesterov.compression, Nesterov.increment,
      Nesterov.Averaged.meanIncrement, broadcastError, Nesterov.target,
      LayerState.M, LayerState.G, LayerState.x, LayerState.w, Probability.average, mul_assoc]
  rw [he]
  have hterm i : (c i).γ*(Nesterov.Averaged.eWeight (c i)*‖P.grad x i‖^2+
      Nesterov.Averaged.lWeight (c i)*average (fun j => ‖P.localGrad j x i‖^2)) =
      g*(2*Nesterov.dWeight (c i)*‖P.grad x i‖^2)+g*β^2*
        (6*Nesterov.dWeight (c i)*‖P.grad x i‖^2+
          6*Nesterov.Averaged.rWeight (c i)*average (fun j => ‖P.localGrad j x i‖^2)) := by
    rw [hγ i]
    unfold Nesterov.Averaged.eWeight Nesterov.Averaged.lWeight
    rw [hβ i]
    field_simp <;> ring
  simp_rw [hterm]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  dsimp [averagedNesterovProfile]
  field_simp <;> ring

theorem averaged_nesterov_noise_profile (P : Problem N m n μ c) {g β : ℝ}
    (hg : 0 < g) (hβpos : 0 < β) (hβ : ∀ i, (c i).β = β)
    (hγ : ∀ i, (c i).γ = g*β) (x : Model m n) :
    Nesterov.Averaged.smoothNoise P = g*β*((averagedNesterovProfile P hg x).s₁*β+
      (averagedNesterovProfile P hg x).s₂*β^2+(averagedNesterovProfile P hg x).s₃*β^3) := by
  dsimp [Nesterov.Averaged.smoothNoise, averagedNesterovProfile]
  simp only [Finset.mul_sum, Finset.sum_mul, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  rw [hγ i]
  unfold Nesterov.Averaged.noise Nesterov.Averaged.eWeight Nesterov.Averaged.lWeight
  rw [hβ i]
  field_simp <;> ring

theorem averaged_nesterov_polynomial_bound (P : Problem N m n μ c) (A : Scheme N m n c)
    {g β : ℝ} (hg : 0 < g) (hβpos : 0 < β) (hβ : ∀ i, (c i).β = β)
    (hγ : ∀ i, (c i).γ = g*β)
    (hsmall : ∀ i, g*(2*(c i).L+
      2*Real.sqrt (Nesterov.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) (T : ℕ) (hT : 0 < T) :
    Nesterov.smoothStationarity P A (g*β) (Nesterov.initial x) T ≤
      (averagedNesterovProfile P hg x).bound T β := by
  have hγ' i : (c i).γ = g*(c i).β := by rw [hβ i]; exact hγ i
  have hstep i := Nesterov.rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i)
    hg.le (hγ' i) (hsmall i)
  have h := Nesterov.Averaged.smooth_average P A hstep (mul_pos hg hβpos) (Nesterov.initial x) T hT
  rw [averaged_nesterov_initial_profile P hg hβpos hβ hγ x,
    averaged_nesterov_noise_profile P hg hβpos hβ hγ x] at h
  have he : 4*(g*((averagedNesterovProfile P hg x).a+(averagedNesterovProfile P hg x).h*β^2))/(g*β*T)+
      4*(g*β*((averagedNesterovProfile P hg x).s₁*β+(averagedNesterovProfile P hg x).s₂*β^2+
        (averagedNesterovProfile P hg x).s₃*β^3))/(g*β) =
      (averagedNesterovProfile P hg x).bound T β := by unfold Profile.bound; field_simp <;> ring
  exact h.trans_eq he

theorem averaged_nesterov_paper_polynomial_bound (P : Problem N m n μ c) (A : Scheme N m n c)
    (hCP : ∀ i, A.CP i 0 = 0)
    {g β : ℝ} (hg : 0 < g) (hβpos : 0 < β) (hβ : ∀ i, (c i).β = β)
    (hγ : ∀ i, (c i).γ = g*β)
    (hsmall : ∀ i, g*(2*(c i).L+
      2*Real.sqrt (Nesterov.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) (T : ℕ) (hT : 0 < T) :
    Nesterov.paperSmoothAverage P A (g*β) x T ≤
      (averagedNesterovProfile P hg x).doubleNoise.bound T β := by
  have hγ' i : (c i).γ = g*(c i).β := by rw [hβ i]; exact hγ i
  have hstep i := Nesterov.rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i)
    hg.le (hγ' i) (hsmall i)
  have h := (Nesterov.Averaged.paper_average P A hCP hstep (mul_pos hg hβpos) x T hT).2
  rw [averaged_nesterov_initial_profile P hg hβpos hβ hγ x,
    averaged_nesterov_noise_profile P hg hβpos hβ hγ x] at h
  have he : 4*(g*((averagedNesterovProfile P hg x).a+(averagedNesterovProfile P hg x).h*β^2))/(g*β*T)+
      8*(g*β*((averagedNesterovProfile P hg x).s₁*β+(averagedNesterovProfile P hg x).s₂*β^2+
        (averagedNesterovProfile P hg x).s₃*β^3))/(g*β) =
      (averagedNesterovProfile P hg x).doubleNoise.bound T β := by
    unfold Profile.bound Profile.doubleNoise
    field_simp <;> ring
  exact h.trans_eq he

/-- Every coefficient improves under the same assumptions and step scale. -/
theorem averaged_nesterov_coefficients_le (P : Problem N m n μ c) {g : ℝ}
    (hg : 0 < g) (x : Model m n) :
    let q := averagedNesterovProfile P hg x
    let old := nesterovProfile P hg x
    q.a ≤ old.a ∧ q.h ≤ old.h ∧ q.s₁ ≤ old.s₁ ∧ q.s₂ ≤ old.s₂ ∧ q.s₃ ≤ old.s₃ := by
  have hN : (0 : ℝ) < N := by exact_mod_cast P.workers_pos
  have hN1 : (1 : ℝ) ≤ N := by exact_mod_cast P.workers_pos
  have hd i : 0 ≤ Nesterov.dWeight (c i) := (Nesterov.Averaged.weights_nonneg (c i)).1
  have hdn i : Nesterov.dWeight (c i)/N ≤ Nesterov.dWeight (c i) := by
    apply (div_le_iff₀ hN).mpr
    nlinarith [mul_le_mul_of_nonneg_left hN1 (hd i)]
  have hG i : ‖P.grad x i‖^2 ≤ average (fun j => ‖P.localGrad j x i‖^2) := by
    rw [← P.mean_gradient x i]
    exact average_sq_le P.workers_pos _
  have hr i := (Nesterov.Averaged.weight_sums (c i)).2
  dsimp [averagedNesterovProfile, nesterovProfile]
  refine ⟨add_le_add (le_refl _) (Finset.sum_le_sum ?_), Finset.sum_le_sum ?_,
    Finset.sum_le_sum ?_, Finset.sum_le_sum ?_, Finset.sum_le_sum ?_⟩
  · intro i _
    exact mul_le_mul_of_nonneg_left (hG i) (mul_nonneg (by norm_num) (hd i))
  · intro i _
    have hh := mul_le_mul_of_nonneg_left (hG i)
      (show 0 ≤ 6*Nesterov.dWeight (c i) from mul_nonneg (by norm_num) (hd i))
    rw [← hr i]
    nlinarith only [hh]
  · intro i _
    have hh := mul_le_mul_of_nonneg_right (hdn i) (sq_nonneg (P.σ i))
    simp only [div_eq_mul_inv] at hh ⊢
    nlinarith only [hh]
  · intro i _
    have hh := mul_le_mul_of_nonneg_right (hdn i) (sq_nonneg (P.σ i))
    rw [← hr i]
    simp only [div_eq_mul_inv] at hh ⊢
    nlinarith only [hh]
  · intro i _
    have hh := mul_le_mul_of_nonneg_right (hdn i) (sq_nonneg (P.σ i))
    rw [← hr i]
    simp only [div_eq_mul_inv] at hh ⊢
    nlinarith only [hh]

/-- The leading variance coefficient is exactly the old coefficient divided by N. -/
theorem averaged_nesterov_leading_variance (P : Problem N m n μ c) {g : ℝ}
    (hg : 0 < g) (x : Model m n) :
    (averagedNesterovProfile P hg x).s₁ = (nesterovProfile P hg x).s₁/N := by
  dsimp [averagedNesterovProfile, nesterovProfile]
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem averaged_nesterov_bound_le (P : Problem N m n μ c) {g : ℝ}
    (hg : 0 < g) (x : Model m n) (T : ℕ) (β : ℝ) (hβ : 0 ≤ β) :
    (averagedNesterovProfile P hg x).bound T β ≤ (nesterovProfile P hg x).bound T β := by
  rcases averaged_nesterov_coefficients_le P hg x with ⟨ha, hh, h1, h2, h3⟩
  unfold Profile.bound
  gcongr

theorem averaged_nesterov_optimalBudget_le (P : Problem N m n μ c) {g ε : ℝ}
    (hg : 0 < g) (hε : 0 < ε) (x : Model m n)
    (ha : 0 < (averagedNesterovProfile P hg x).a) :
    (averagedNesterovProfile P hg x).optimalBudget ε ≤ (nesterovProfile P hg x).optimalBudget ε := by
  let old := nesterovProfile P hg x
  have hao : 0 < old.a := ha.trans_le (averaged_nesterov_coefficients_le P hg x).1
  have hs := old.optimalBudget_spec hao hε
  have hv := old.optimalBeta_valid hs.1 hao
  exact (averagedNesterovProfile P hg x).optimalBudget_minimal ha hε hs.1 hv.1 hv.2
    ((averaged_nesterov_bound_le P hg x _ _ hv.1.le).trans hs.2)

end EF21Muon.LongTails
