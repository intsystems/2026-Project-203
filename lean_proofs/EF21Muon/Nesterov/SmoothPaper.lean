import EF21Muon.Nesterov.Correspondence
import EF21Muon.Probability.PathStates

/-! Smooth convergence for the paper's gradient/uplink, then step/downlink
round order. The extra sample coordinate does not change the paper iterates;
it lets the index shift be proved on a single finite product probability space. -/

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} {c : Fin p → Parameters}

def sharpServer (A : Scheme N m n c) (s : States N m n) : States N m n :=
  fun i => ((nextX (c i) (s i).1, nextW (c i) (A.CP i) (s i).1,
    (s i).1.M, (s i).1.G), (s i).2)

def sharpWorker (P : Problem N m n μ c) (A : Scheme N m n c)
    (s : States N m n) (ω : Ω) : States N m n :=
  fun i =>
    let reply := fun j => P.reply (fun k => (s k).1.w) ω j i
    let M := fun j => (1-(c i).β) • (s i).1.M j+(c i).β • reply j
    let V := fun j => (1-(c i).β) • M j+(c i).β • reply j
    (((s i).1.x, (s i).1.w, M, fun j => (s i).1.G j+A.CU i (V j-(s i).1.G j)),(s i).1.M)

def sharpPaperNext (P : Problem N m n μ c) (A : Scheme N m n c)
    (s : States N m n) (ω : Ω) : States N m n := sharpServer A (sharpWorker P A s ω)

theorem sharpServer_start (A : Scheme N m n c) (hCP : ∀ i, A.CP i 0 = 0)
    (x : Model m n) : sharpServer A (initial x) = initial x := by
  funext i
  simp [sharpServer, initial, start, LayerState.start, nextX, nextW,
    LayerState.server, LayerState.x, LayerState.w, LayerState.M, LayerState.G,
    serverPoint, queryPoint, average, polar_zero, hCP]

theorem sharpPaperNext_project (P : Problem N m n μ c) (A : Scheme N m n c)
    (s : States N m n) (ω : Ω) :
    sharpPaperNext P A (sharpServer A s) ω = sharpServer A (next A P.reply s ω) := rfl

theorem sharpPaper_pathState (P : Problem N m n μ c) (A : Scheme N m n c)
    (hCP : ∀ i, A.CP i 0 = 0) (k : ℕ) (x : Model m n) (ξ : Samples Ω k) (t : Fin k) :
    pathState (sharpPaperNext P A) k (initial x) ξ t =
      sharpServer A (pathState (next A P.reply) k (initial x) ξ t) := by
  conv_lhs => rw [← sharpServer_start A hCP x]
  exact pathState_map _ _ _ (sharpPaperNext_project P A) _ _ _ _

def sharpPaperSum (P : Problem N m n μ c) (A : Scheme N m n c)
    (x : Model m n) (T : ℕ) (ξ : Samples Ω (T+1)) : ℝ :=
  ∑ t : Fin T, smoothCost P (pathState (sharpPaperNext P A) (T+1) (initial x) ξ t.castSucc)

theorem sharpPaperSum_eq (P : Problem N m n μ c) (A : Scheme N m n c)
    (hCP : ∀ i, A.CP i 0 = 0) (x : Model m n) (T : ℕ) (ξ : Samples Ω (T+1)) :
    sharpPaperSum P A x T ξ =
      ((run (next A P.reply) (T+1) (initial x) ξ).map (smoothCost P)).sum-
        smoothCost P (initial x) := by
  have hcost (s : States N m n) (ω : Ω) :
      smoothCost P (next A P.reply s ω) = smoothCost P (sharpServer A s) := rfl
  have he (t : Fin T) :
      smoothCost P (pathState (next A P.reply) (T+1) (initial x) ξ t.succ) =
        smoothCost P (pathState (sharpPaperNext P A) (T+1) (initial x) ξ t.castSucc) := by
    rw [sharpPaper_pathState P A hCP]
    exact pathState_adjacent_cost _ _ _ hcost T (initial x) ξ t
  rw [sum_run_eq_states, Fin.sum_univ_succ]
  rw [show pathState (next A P.reply) (T+1) (initial x) ξ 0 = initial x from rfl]
  simp_rw [he]
  change _ = smoothCost P (initial x) + _ - smoothCost P (initial x)
  unfold sharpPaperSum
  ring

def paperSmoothAverage (P : Problem N m n μ c) (A : Scheme N m n c)
    (γbar : ℝ) (x : Model m n) (T : ℕ) : ℝ :=
  4/(γbar*T)*(∫ ξ, sharpPaperSum P A x T ξ ∂pathMeasure μ (T+1))

theorem paperSmoothAverage_eq (P : Problem N m n μ c) (A : Scheme N m n c)
    (γbar : ℝ) (x : Model m n) (T : ℕ) :
    paperSmoothAverage P A γbar x T =
      (∫ ξ, ∑ t : Fin T, ∑ i, (c i).γ/γbar*nuclearNorm (P.grad
        (points (pathState (sharpPaperNext P A) (T+1) (initial x) ξ t.castSucc)) i)^2
        ∂pathMeasure μ (T+1))/T := by
  have he (s : States N m n) :
      (∑ i, (c i).γ/γbar*nuclearNorm (P.grad (points s) i)^2) = 4/γbar*smoothCost P s := by
    simp only [smoothCost, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _; ring
  simp only [paperSmoothAverage, sharpPaperSum, he, ← Finset.mul_sum, integral_const_mul]
  ring

variable [IsProbabilityMeasure μ]

theorem smooth_paper_average (P : Problem N m n μ c) (A : Scheme N m n c)
    (hCP : ∀ i, A.CP i 0 = 0)
    (hsmall : ∀ i, 2*(c i).L*(c i).γ+4*zeta (c i) (P.ell2 i) (A.τ i)*(c i).γ^2 ≤ 1)
    {γbar : ℝ} (hγ : 0 < γbar) (x : Model m n) (T : ℕ) (hT : 0 < T) :
    Integrable (sharpPaperSum P A x T) (pathMeasure μ (T+1)) ∧
      paperSmoothAverage P A γbar x T ≤
        4*smoothPotential P (initial x)/(γbar*T)+8*smoothNoise P/γbar := by
  have h := smooth_finite_time P A hsmall (T+1) (initial x)
  have hi := h.1.sub (integrable_const (smoothCost P (initial x)))
  change Integrable (fun ξ =>
    ((run (next A P.reply) (T+1) (initial x) ξ).map (smoothCost P)).sum-
      smoothCost P (initial x)) (pathMeasure μ (T+1)) at hi
  have hi' : Integrable (sharpPaperSum P A x T) (pathMeasure μ (T+1)) := by
    simpa only [← sharpPaperSum_eq P A hCP x T] using hi
  refine ⟨hi', ?_⟩
  have he : (∫ ξ, sharpPaperSum P A x T ξ ∂pathMeasure μ (T+1)) =
      (∫ ξ, ((run (next A P.reply) (T+1) (initial x) ξ).map (smoothCost P)).sum
        ∂pathMeasure μ (T+1))-smoothCost P (initial x) := by
    simp_rw [sharpPaperSum_eq P A hCP]
    rw [integral_sub h.1 (integrable_const _), integral_const, probReal_univ, one_smul]
  have hTr : (0:ℝ) < T := by exact_mod_cast hT
  have hT1 : (T:ℝ)+1 ≤ 2*T := by exact_mod_cast (show T+1 ≤ 2*T by omega)
  have hn := mul_le_mul_of_nonneg_right hT1 (smoothNoise_nonneg P)
  have hc := smoothCost_nonneg P (initial x)
  have hb : (∫ ξ, sharpPaperSum P A x T ξ ∂pathMeasure μ (T+1)) ≤
      smoothPotential P (initial x)+2*T*smoothNoise P := by
    rw [he]
    simp only [Nat.cast_add, Nat.cast_one] at h
    linarith only [h.2, hn, hc]
  unfold paperSmoothAverage
  have hb' := mul_le_mul_of_nonneg_left hb (by positivity : 0 ≤ 4/(γbar*T))
  convert hb' using 1 <;> field_simp <;> ring

theorem smooth_paper_rate (P : Problem N m n μ c) (A : Scheme N m n c)
    (hCP : ∀ i, A.CP i 0 = 0) {g : ℝ} (hg : 0 < g) (T : ℕ) (hT : 0 < T)
    (hβ : ∀ i, (c i).β = 1/Real.sqrt T) (hγ : ∀ i, (c i).γ = g*(c i).β)
    (hsmall : ∀ i, g*(2*(c i).L+2*Real.sqrt (rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) :
    paperSmoothAverage P A (g/Real.sqrt T) x T ≤
      (4*rateInitial P g x/g+8*noiseTotalScale P)/Real.sqrt T := by
  have hTr : (0:ℝ) < T := by exact_mod_cast hT
  have hr := Real.sqrt_pos.mpr hTr
  have hrsq := Real.sq_sqrt hTr.le
  have hsteps i : (c i).γ = g/Real.sqrt T := by rw [hγ i,hβ i]; ring
  have hstep i := rate_step_condition (c i) (P.ell2_nonneg i) (A.τ_nonneg i) hg.le (hγ i) (hsmall i)
  have hbound := (smooth_paper_average P A hCP hstep (div_pos hg hr) x T hT).2
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
  have hn := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hnoise (by norm_num : (0:ℝ) ≤ 8))
    (div_pos hg hr).le
  apply hbound.trans
  apply (add_le_add hi hn).trans
  apply le_of_eq
  field_simp
  rw [hrsq]
  ring

/-- A single fixed problem and oracle, with explicit horizon controls, directly
in the paper's round order. Both concrete downlink modes satisfy hCP. -/
theorem paper_horizon_rate_isBigO (P : Problem N m n μ c) (τ : Fin p → ℝ)
    (A : ∀ T, Scheme N m n (horizonControls c P.ell2 τ T))
    (hτ : ∀ T i, (A T).τ i = τ i) (hCP : ∀ T i, (A T).CP i 0 = 0) (x : Model m n) :
    (fun T => paperSmoothAverage (forHorizon P τ T) (A T)
      (rateGamma c P.ell2 τ / Real.sqrt T) x T)
      =O[Filter.atTop] (fun T : ℕ => 1 / Real.sqrt T) := by
  let g := rateGamma c P.ell2 τ
  have hg : 0 < g := rateGamma_pos c P.ell2 τ
  apply Asymptotics.IsBigO.of_bound (4*rateInitial P g x/g+8*noiseTotalScale P)
  filter_upwards [Filter.eventually_gt_atTop (0 : ℕ)] with T hT
  have hr := Real.sqrt_pos.mpr (show (0:ℝ) < T by exact_mod_cast hT)
  have hβ (i : Fin p) : (horizonControls c P.ell2 τ T i).β = 1/Real.sqrt T := by
    change horizonBeta T = _
    rw [horizonBeta, max_eq_right hT]
  have hsmall (i : Fin p) : g*(2*(horizonControls c P.ell2 τ T i).L+
      2*Real.sqrt (rateEnvelope (horizonControls c P.ell2 τ T i)
        ((forHorizon P τ T).ell2 i) ((A T).τ i))) ≤ 1 := by
    rw [hτ T i]
    exact (rate_common_step c P.ell2 τ).2 i
  have h := smooth_paper_rate (forHorizon P τ T) (A T) (hCP T) hg T hT hβ
    (fun _ => rfl) hsmall x
  change paperSmoothAverage (forHorizon P τ T) (A T) (g/Real.sqrt T) x T ≤
    (4*rateInitial P g x/g+8*noiseTotalScale P)/Real.sqrt T at h
  have hn : 0 ≤ paperSmoothAverage (forHorizon P τ T) (A T) (g/Real.sqrt T) x T := by
    unfold paperSmoothAverage sharpPaperSum
    apply mul_nonneg (by positivity)
    exact integral_nonneg fun _ => Finset.sum_nonneg fun _ _ => smoothCost_nonneg _ _
  rw [Real.norm_eq_abs, abs_of_nonneg hn, Real.norm_eq_abs,
    abs_of_nonneg (one_div_pos.mpr hr).le]
  simpa only [div_eq_mul_inv, mul_one, one_mul] using h

end EF21Muon.Nesterov
