import EF21Muon.Distributed.Geometry

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N m n : ℕ}

def energy (c : Parameters) (ell2 : ℝ) (s : LayerState N m n)
    (g : Space m n) (localG : Fin N → Space m n) : ℝ :=
  c.γ * (globalWeight c * globalError s g + localWeight c * localError s localG +
    c.uplinkWeight * compressionError s + broadcastWeight c ell2 * broadcastError s)

theorem energy_nonneg (c : Parameters) {ell2 : ℝ} (hell : 0 ≤ ell2)
    (s : LayerState N m n) (g : Space m n) (localG : Fin N → Space m n) :
    0 ≤ energy c ell2 s g localG := by
  rcases weights_nonneg c hell with ⟨hA, hB, hC, _, hD⟩
  have hE : 0 ≤ globalError s g := sq_nonneg _
  have hF : 0 ≤ localError s localG := average_nonneg (fun _ => sq_nonneg _)
  have hH : 0 ≤ compressionError s := average_nonneg (fun _ => sq_nonneg _)
  have hP : 0 ≤ broadcastError s := sq_nonneg _
  exact mul_nonneg c.γ_pos.le (add_nonneg (add_nonneg (add_nonneg
    (mul_nonneg hA hE) (mul_nonneg hB hF)) (mul_nonneg hC hH)) (mul_nonneg hD hP))

variable {Ω : Type*} [MeasurableSpace Ω] {p : Measure Ω} [IsProbabilityMeasure p]

theorem error_integrable (c : Parameters) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU) (CP : Space m n → Space m n)
    (s : LayerState N m n) (g : Space m n) (localG q : Fin N → Space m n)
    {σ : ℝ} (O : Noise N m n p q σ) :
    Integrable (fun ω => globalError (update c CU CP s (O.replies ω)) g) p ∧
    Integrable (fun ω => localError (update c CU CP s (O.replies ω)) localG) p ∧
    Integrable (fun ω => compressionError (update c CU CP s (O.replies ω))) p := by
  have hM j : MemLp (fun ω => (update c CU CP s (O.replies ω)).M j) 2 p :=
    newMomentum_memLp c.β (s.M j) (q j) (O.memLp j)
  have hE := square_integrable ((average_memLp hM).sub (memLp_const g))
  have hF j := square_integrable ((hM j).sub (memLp_const (localG j)))
  have hH j := square_integrable (residual_memLp hm c.αU_pos hc
    ((hM j).sub (memLp_const (s.G j))))
  simp only [Pi.sub_apply] at hE hF hH
  have he (ω : Ω) (j : Fin N) :
      ‖(update c CU CP s (O.replies ω)).M j - (update c CU CP s (O.replies ω)).G j‖ ^ 2 =
        ‖CU ((update c CU CP s (O.replies ω)).M j - s.G j) -
          ((update c CU CP s (O.replies ω)).M j - s.G j)‖ ^ 2 := by
    rw [norm_sub_rev]; congr 2
    dsimp [update, LayerState.G, LayerState.M]; module
  refine ⟨hE, ?_, ?_⟩
  · exact (integrable_finsetSum _ (fun j _ => hF j)).const_mul _
  · simp only [compressionError, he]
    exact (integrable_finsetSum _ (fun j _ => hH j)).const_mul _

theorem energy_integrable (c : Parameters) (ell2 : ℝ) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU) (CP : Space m n → Space m n)
    (s : LayerState N m n) (g : Space m n) (localG q : Fin N → Space m n)
    {σ : ℝ} (O : Noise N m n p q σ) :
    Integrable (fun ω => energy c ell2 (update c CU CP s (O.replies ω)) g localG) p := by
  rcases error_integrable c hm hc CP s g localG q O with ⟨hE, hF, hH⟩
  have hsum := ((((hE.const_mul (globalWeight c)).add (hF.const_mul (localWeight c))).add
    (hH.const_mul c.uplinkWeight)).add (integrable_const
      (broadcastWeight c ell2 * ‖nextW c CP s - nextX c s‖ ^ 2))).const_mul c.γ
  simpa only [Pi.add_apply, energy, broadcastError, LayerState.w, LayerState.x, update] using hsum

omit [IsProbabilityMeasure p] in
theorem integral_add4 {a b d e : Ω → ℝ} (ha : Integrable a p) (hb : Integrable b p)
    (hd : Integrable d p) (he : Integrable e p) :
    (∫ ω, a ω + b ω + d ω + e ω ∂p) =
      (∫ ω, a ω ∂p) + (∫ ω, b ω ∂p) + (∫ ω, d ω ∂p) + ∫ ω, e ω ∂p := by
  have hab : Integrable (fun ω => a ω + b ω) p := ha.add hb
  have habd : Integrable (fun ω => a ω + b ω + d ω) p := hab.add hd
  rw [integral_add habd he, integral_add hab hd, integral_add ha hb]

set_option maxHeartbeats 1200000 in
theorem layer_descent (c : Parameters) (hN : 0 < N) {ell2 τ σ : ℝ} (hell : 0 ≤ ell2)
    {CU CP : Space m n → Space m n} (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (hdown : BroadcastBound c CP τ) (hdual : ∀ z : Space m n, nuclearNorm z ≤ c.ρ * ‖z‖)
    (hsmall : 2 * c.L * c.γ + 4 * zeta c ell2 τ * c.γ ^ 2 ≤ 1)
    (s : LayerState N m n) (old new : Space m n) (oldJ newJ q : Fin N → Space m n)
    (O : Noise N m n p q σ)
    (ha : ‖old - new‖ ^ 2 ≤ c.L ^ 2 * c.γ ^ 2 * nuclearNorm (average s.G) ^ 2)
    (hb : ‖average q - new‖ ^ 2 ≤ c.L ^ 2 * ‖nextW c CP s - nextX c s‖ ^ 2)
    (haJ : average (fun j => ‖oldJ j - newJ j‖ ^ 2) ≤
      ell2 * c.γ ^ 2 * nuclearNorm (average s.G) ^ 2)
    (hbJ : average (fun j => ‖q j - newJ j‖ ^ 2) ≤
      ell2 * ‖nextW c CP s - nextX c s‖ ^ 2) :
    inner ℝ old (nextX c s - s.x) + c.L / 2 * spectralNorm (nextX c s - s.x) ^ 2 +
      (∫ ω, energy c ell2 (update c CU CP s (O.replies ω)) new newJ ∂p) +
        c.γ / 4 * nuclearNorm old ^ 2 ≤ energy c ell2 s old oldJ + c.γ * noise c N * σ ^ 2 := by
  rcases weights_nonneg c hell with ⟨hA, hB, hC, hI, hD⟩
  have hγ := c.γ_pos
  have hd := direction_bound c hN hdual s old
  have hs := mul_self_le_mul_self (show 0 ≤ spectralNorm (nextX c s - s.x) from norm_nonneg _)
    (movement_spectral c s.server)
  have hs' := mul_le_mul_of_nonneg_left hs (div_nonneg c.L_nonneg (by norm_num : (0 : ℝ) ≤ 2))
  have he := mul_le_mul_of_nonneg_left (global_recurrence c hN CU CP s old new q O ha hb)
    (mul_nonneg hγ.le hA)
  have hf := mul_le_mul_of_nonneg_left (local_recurrence c hN CU CP s oldJ newJ q O haJ hbJ)
    (mul_nonneg hγ.le hB)
  have hh := mul_le_mul_of_nonneg_left (compression_recurrence c hN hm hc CP s oldJ newJ q O haJ hbJ)
    (mul_nonneg hγ.le hC)
  have hp := mul_le_mul_of_nonneg_left (hdown s.server) (mul_nonneg hγ.le (add_nonneg hI hD))
  rcases coefficients c N ell2 with ⟨hE, hF, hH, hP, hNoise⟩
  have hE' := congrArg (fun r : ℝ => c.γ * r * globalError s old) hE
  have hF' := congrArg (fun r : ℝ => c.γ * r * localError s oldJ) hF
  have hH' := congrArg (fun r : ℝ => c.γ * r * compressionError s) hH
  have hP' := congrArg (fun r : ℝ => c.γ * r * broadcastError s) hP
  have hN' := congrArg (fun r : ℝ => c.γ * r * σ ^ 2) hNoise
  have hI' : c.γ * injection c ell2 * ‖nextW c CP s - nextX c s‖ ^ 2 =
      c.γ * (globalWeight c * c.β ^ 2 * (1 + 2 / c.β) * c.L ^ 2 +
        localWeight c * c.β ^ 2 * (1 + 2 / c.β) * ell2 +
          c.uplinkWeight * (6 * c.β ^ 2 * ell2 / c.αU)) * ‖nextW c CP s - nextX c s‖ ^ 2 := rfl
  have hZ : c.γ ^ 3 * zeta c ell2 τ * nuclearNorm (average s.G) ^ 2 =
      c.γ ^ 3 * (globalWeight c * (2 * c.L ^ 2 / c.β) + localWeight c * (2 * ell2 / c.β) +
        c.uplinkWeight * (6 * c.β ^ 2 * ell2 / c.αU) +
          (injection c ell2 + broadcastWeight c ell2) * (2 * τ / c.αP)) *
            nuclearNorm (average s.G) ^ 2 := rfl
  have hstep := mul_le_mul_of_nonneg_left hsmall
    (by positivity : 0 ≤ c.γ / 4 * nuclearNorm (average s.G) ^ 2)
  rcases error_integrable c hm hc CP s new newJ q O with ⟨hiE, hiF, hiH⟩
  have hbp (ω : Ω) : broadcastError (update c CU CP s (O.replies ω)) =
      ‖nextW c CP s - nextX c s‖ ^ 2 := rfl
  simp only [energy, hbp, integral_const_mul]
  rw [integral_add4 (hiE.const_mul _) (hiF.const_mul _) (hiH.const_mul _) (integrable_const _)]
  simp only [integral_const_mul, integral_const, probReal_univ, one_smul]
  change c.γ * (injection c ell2 + broadcastWeight c ell2) *
    ‖nextW c CP s - nextX c s‖ ^ 2 ≤ c.γ * (injection c ell2 + broadcastWeight c ell2) *
      ((1 - c.αP / 2) * broadcastError s +
        2 * τ / c.αP * c.γ ^ 2 * nuclearNorm (average s.G) ^ 2) at hp
  change c.L / 2 * (spectralNorm (nextX c s - s.x) * spectralNorm (nextX c s - s.x)) ≤
    c.L / 2 * (c.γ * nuclearNorm (average s.G) * (c.γ * nuclearNorm (average s.G))) at hs'
  simp only [div_eq_mul_inv] at hd hs' he hf hh hp hE' hF' hH' hP' hN' hI' hZ hstep ⊢
  nlinarith only [hd, hs', he, hf, hh, hp, hE', hF', hH', hP', hN', hI', hZ, hstep]

end EF21Muon.Distributed
