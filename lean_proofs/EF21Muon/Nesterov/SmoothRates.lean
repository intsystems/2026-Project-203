import EF21Muon.Nesterov.SmoothConvergence
import EF21Muon.Distributed.Rates

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

def rateEnvelope (c : Parameters) (ell2 τ : ℝ) : ℝ :=
  (4*dWeight c+15*rWeight c)*ell2+
    (6*dWeight c+21*rWeight c)*ell2*(1+2/c.αP)*2*τ/c.αP

def noiseScale (c : Parameters) : ℝ := 2*dWeight c+7*rWeight c

theorem beta_f_bound (c : Parameters) : c.β*fWeight c ≤ 2*dWeight c+6*rWeight c := by
  have hb := c.β_pos
  have hb2 : c.β^2 ≤ 1 := by nlinarith [c.β_le_one]
  have hr := (weights_nonneg c (le_refl (0:ℝ))).2.2.2.1
  have he : c.β*fWeight c = 2*dWeight c+6*c.β^2*rWeight c := by
    dsimp [fWeight]; field_simp <;> ring
  rw [he]
  nlinarith only [mul_le_mul_of_nonneg_right hb2 hr]

theorem injection_bound (c : Parameters) {ell2 : ℝ} (hell : 0 ≤ ell2) :
    injection c ell2 ≤ (6*dWeight c+21*rWeight c)*ell2 := by
  rcases weights_nonneg c hell with ⟨hd,_,_,hr,hf,_,_⟩
  have hb := c.β_pos
  have hb2 : c.β^2 ≤ 1 := by nlinarith [c.β_le_one]
  have hp : 0 ≤ c.β*fWeight c := mul_nonneg hb.le hf
  have hprod := mul_le_mul (beta_f_bound c) (show c.β+2 ≤ 3 by linarith [c.β_le_one])
    (by positivity : 0 ≤ c.β+2) (by positivity : 0 ≤ 2*dWeight c+6*rWeight c)
  have he : injection c ell2 = ((c.β*fWeight c)*(c.β+2)+3*c.β^2*rWeight c)*ell2 := by
    dsimp [injection]; field_simp <;> ring
  rw [he]
  exact mul_le_mul_of_nonneg_right
    (by nlinarith only [hprod,mul_le_mul_of_nonneg_right hb2 hr]) hell

theorem zeta_envelope (c : Parameters) {ell2 τ : ℝ} (hell : 0 ≤ ell2) (hτ : 0 ≤ τ) :
    c.β^2*zeta c ell2 τ ≤ rateEnvelope c ell2 τ := by
  rcases weights_nonneg c hell with ⟨hd,_,_,hr,_,hi,hp⟩
  have hb := c.β_pos
  have hP := c.αP_pos
  have hb2 : c.β^2 ≤ 1 := by nlinarith [c.β_le_one]
  have hb4 : c.β^4 ≤ 1 := by nlinarith [sq_nonneg (c.β^2-1)]
  have hfirst := mul_le_mul_of_nonneg_right (beta_f_bound c) (by positivity : 0 ≤ 2*ell2)
  have hsecond := mul_le_mul_of_nonneg_right hb4 (by positivity : 0 ≤ 3*rWeight c*ell2)
  have hbro0 : 0 ≤ (injection c ell2+pWeight c ell2)*2*τ/c.αP := by positivity
  have hbro1 := mul_le_mul_of_nonneg_right hb2 hbro0
  have hbro2 := mul_le_mul_of_nonneg_right (injection_bound c hell)
    (by positivity : 0 ≤ (1+2/c.αP)*2*τ/c.αP)
  have he : c.β^2*zeta c ell2 τ =
      2*(c.β*fWeight c)*ell2+3*c.β^4*rWeight c*ell2+
      c.β^2*((injection c ell2+pWeight c ell2)*2*τ/c.αP) := by
    dsimp [zeta]; field_simp <;> ring
  have heP : injection c ell2+pWeight c ell2 = injection c ell2*(1+2/c.αP) := by
    dsimp [pWeight]; ring
  rw [he,heP]
  rw [heP] at hbro1
  dsimp [rateEnvelope]
  simp only [div_eq_mul_inv] at hbro1 hbro2 ⊢
  nlinarith only [hfirst,hsecond,hbro1,hbro2]

theorem noise_bound (c : Parameters) : noise c ≤ c.β*noiseScale c := by
  rcases weights_nonneg c (le_refl (0:ℝ)) with ⟨hd,_,_,hr,_,_,_⟩
  have hb := c.β_pos
  have hb2 : c.β^2 ≤ 1 := by nlinarith [c.β_le_one]
  have he : noise c = c.β*(2*dWeight c+6*c.β^2*rWeight c+c.β*rWeight c) := by
    dsimp [noise,fWeight]; field_simp <;> ring
  rw [he]
  apply mul_le_mul_of_nonneg_left _ hb.le
  dsimp [noiseScale]
  nlinarith only [mul_le_mul_of_nonneg_right hb2 hr,
    mul_le_mul_of_nonneg_right c.β_le_one hr]

theorem rateEnvelope_nonneg (c : Parameters) {ell2 τ : ℝ}
    (hell : 0 ≤ ell2) (hτ : 0 ≤ τ) : 0 ≤ rateEnvelope c ell2 τ := by
  rcases weights_nonneg c hell with ⟨hd,_,_,hr,_,_,_⟩
  have hp := c.αP_pos
  dsimp [rateEnvelope]; positivity

theorem rate_step_condition (c : Parameters) {ell2 τ g : ℝ}
    (hell : 0 ≤ ell2) (hτ : 0 ≤ τ) (hg : 0 ≤ g) (hγ : c.γ = g*c.β)
    (hsmall : g*(2*c.L+2*Real.sqrt (rateEnvelope c ell2 τ)) ≤ 1) :
    2*c.L*c.γ+4*zeta c ell2 τ*c.γ^2 ≤ 1 := by
  have hz : 0 ≤ zeta c ell2 τ := by
    rcases weights_nonneg c hell with ⟨_,_,_,hr,hf,hi,hp⟩
    have hb := c.β_pos; have hP := c.αP_pos
    dsimp [zeta]; positivity
  have hs : c.β*Real.sqrt (zeta c ell2 τ) ≤ Real.sqrt (rateEnvelope c ell2 τ) := by
    have h := zeta_envelope c hell hτ
    have hsq := Real.sq_sqrt hz
    have hsqZ := Real.sq_sqrt (rateEnvelope_nonneg c hell hτ)
    nlinarith [Real.sqrt_nonneg (zeta c ell2 τ), Real.sqrt_nonneg (rateEnvelope c ell2 τ), c.β_pos]
  apply c.step_condition_for hz
  rw [hγ]
  have hL := mul_le_mul_of_nonneg_left c.β_le_one c.L_nonneg
  nlinarith [mul_le_mul_of_nonneg_left hs hg,mul_le_mul_of_nonneg_left hL hg]

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def rateInitial (P : Problem N m n μ c) (g : ℝ) (x : Model m n) : ℝ :=
  P.f x-P.fmin+g*∑ i, (2*dWeight (c i)+6*rWeight (c i))*
    average (fun j => ‖P.localGrad j x i‖^2)

def noiseTotalScale (P : Problem N m n μ c) : ℝ := ∑ i, noiseScale (c i)*P.σ i^2

theorem initial_potential_bound (P : Problem N m n μ c) {g : ℝ} (hg : 0 ≤ g)
    (hγ : ∀ i, (c i).γ = g*(c i).β) (x : Model m n) :
    smoothPotential P (initial x) ≤ rateInitial P g x := by
  have he : smoothPotential P (initial x) = P.f x-P.fmin+
      ∑ i, (c i).γ*fWeight (c i)*average (fun j => ‖P.localGrad j x i‖^2) := by
    have hx : points (initial x : States N m n) = x := rfl
    simp only [smoothPotential,hx]
    congr 1
    apply Finset.sum_congr rfl
    intro i _
    simp [initial, start, LayerState.start,
      energy, localError, compression, increment, broadcastError, target,
      LayerState.M, LayerState.G, LayerState.x, LayerState.w, average, mul_assoc]
  rw [he]
  dsimp [rateInitial]
  rw [Finset.mul_sum]
  apply add_le_add le_rfl
  apply Finset.sum_le_sum
  intro i _
  rw [hγ i]
  have h := mul_le_mul_of_nonneg_left (beta_f_bound (c i)) hg
  have hcoef : g*(c i).β*fWeight (c i) ≤ g*(2*dWeight (c i)+6*rWeight (c i)) := by
    nlinarith only [h]
  simpa only [mul_assoc] using mul_le_mul_of_nonneg_right hcoef
    (Distributed.average_nonneg (fun j => sq_nonneg ‖P.localGrad j x i‖))

/-- The O(T^(-1/2)) Nesterov rate with a horizon-independent explicit constant. -/
theorem smooth_rate (P : Problem N m n μ c) (A : Scheme N m n c)
    {g : ℝ} (hg : 0 < g) (T : ℕ) (hT : 0 < T)
    (hβ : ∀ i, (c i).β = 1/Real.sqrt T)
    (hγ : ∀ i, (c i).γ = g*(c i).β)
    (hsmall : ∀ i, g*(2*(c i).L+2*Real.sqrt (rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) :
    smoothStationarity P A (g/Real.sqrt T) (initial x) T ≤
      (4*rateInitial P g x/g+4*noiseTotalScale P)/Real.sqrt T := by
  have hTr : (0:ℝ) < T := by exact_mod_cast hT
  have hr := Real.sqrt_pos.mpr hTr
  have hrsq := Real.sq_sqrt hTr.le
  have hsteps i : (c i).γ = g/Real.sqrt T := by rw [hγ i,hβ i]; ring
  have hstep i := rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i) hg.le (hγ i) (hsmall i)
  have hbound := smooth_average P A hstep (div_pos hg hr) (initial x) T hT
  have hinit := initial_potential_bound P hg.le hγ x
  have hnoise : smoothNoise P ≤ (g/Real.sqrt T)*((1/Real.sqrt T)*noiseTotalScale P) := by
    unfold smoothNoise noiseTotalScale
    simp only [hsteps,mul_assoc,← Finset.mul_sum]
    apply mul_le_mul_of_nonneg_left _ (div_pos hg hr).le
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun i _ => by
      simpa only [hβ i,mul_assoc] using mul_le_mul_of_nonneg_right (noise_bound (c i)) (sq_nonneg (P.σ i)))
  have hi := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hinit (by norm_num : (0:ℝ) ≤ 4))
    (mul_pos (div_pos hg hr) hTr).le
  have hn := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hnoise (by norm_num : (0:ℝ) ≤ 4))
    (div_pos hg hr).le
  apply hbound.trans
  apply (add_le_add hi hn).trans
  apply le_of_eq
  field_simp
  rw [hrsq]
  ring

end EF21Muon.Nesterov
