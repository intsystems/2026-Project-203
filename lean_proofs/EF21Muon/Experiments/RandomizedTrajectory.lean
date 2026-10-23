import EF21Muon.Experiments.Trajectory

/-! Experimental fixed-momentum trajectories with measurable compression
operators depending on time, worker, and sample. This extends the deterministic
operator proof to fresh random ties without changing its tracking constants.
The noise, roundoff, and scalar recurrence lemmas are shared with Trajectory. -/

noncomputable section
namespace EF21Muon.Experiments.RandomizedTrajectory
set_option maxHeartbeats 800000
set_option backward.isDefEq.respectTransparency false
open scoped BigOperators
open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Distributed
open EF21Muon.Experiments.Trajectory (E Innovations Gradients Roundoff ema ema_memLp)
open LongTails.Algorithms.MatrixPolar

variable {N m n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Deterministic controls and uniform contracts for the implemented operators. -/
structure Controls (N m n : ℕ) (Ω : Type*) [MeasurableSpace Ω] where
  β : ℝ
  θ : ℝ
  qU : ℝ
  qP : ℝ
  B : ℝ
  peak : ℝ
  β_pos : 0 < β
  β_one : β ≤ 1
  θ_nonneg : 0 ≤ θ
  θ_one : θ ≤ 1
  qU_nonneg : 0 ≤ qU
  qU_one : qU < 1
  qP_nonneg : 0 ≤ qP
  qP_one : qP < 1
  B_nonneg : 0 ≤ B
  peak_nonneg : 0 ≤ peak
  η : ℕ → ℝ
  η_nonneg : ∀ t, 0 ≤ η t
  η_peak : ∀ t, η t ≤ peak
  CU : ℕ → Fin N → Ω → E m n → E m n
  CP : ℕ → Ω → E m n → E m n
  measurable_CU : ∀ t j, Measurable (fun z : Ω × E m n => CU t j z.1 z.2)
  measurable_CP : ∀ t, Measurable (fun z : Ω × E m n => CP t z.1 z.2)
  contracts_CU : ∀ t j ω z, ‖z-CU t j ω z‖ ≤ qU*‖z‖
  contracts_CP : ∀ t ω z, ‖z-CP t ω z‖ ≤ qP*‖z‖
  direction : ℕ → E m n → E m n
  measurable_direction : ∀ t, Measurable (direction t)
  direction_bound : ∀ t h, ‖direction t h‖ ≤ B

variable (g : Gradients N m n) (c : Controls N m n Ω)
  (O : Innovations N m n μ) (R : Roundoff N m n Ω)

omit [IsProbabilityMeasure μ] in
private theorem expectation_add (X Y : Ω → ℝ) (hX : Integrable X μ) (hY : Integrable Y μ) :
    (∫ ω, X ω+Y ω ∂μ) = (∫ ω, X ω ∂μ)+(∫ ω, Y ω ∂μ) := integral_add hX hY

private theorem expectation_affine (a b d : ℝ) (X Y : Ω → ℝ)
    (hX : Integrable X μ) (hY : Integrable Y μ) :
    (∫ ω, a+b*X ω+d*Y ω ∂μ) = a+b*(∫ ω, X ω ∂μ)+d*(∫ ω, Y ω ∂μ) := by
  rw [expectation_add (fun ω => a+b*X ω) (fun ω => d*Y ω)
    ((integrable_const a).add (hX.const_mul b)) (hY.const_mul d),
    expectation_add (fun _ => a) (fun ω => b*X ω) (integrable_const a) (hX.const_mul b),
    integral_const_mul, integral_const_mul, integral_const]
  simp

private theorem expectation_affine_three (a b d e : ℝ) (X Y Z : Ω → ℝ)
    (hX : Integrable X μ) (hY : Integrable Y μ) (hZ : Integrable Z μ) :
    (∫ ω, a+b*X ω+d*Y ω+e*Z ω ∂μ) =
      a+b*(∫ ω, X ω ∂μ)+d*(∫ ω, Y ω ∂μ)+e*(∫ ω, Z ω ∂μ) := by
  rw [expectation_add (fun ω => a+b*X ω+d*Y ω) (fun ω => e*Z ω)
    (((integrable_const a).add (hX.const_mul b)).add (hY.const_mul d)) (hZ.const_mul e),
    expectation_affine a b d X Y hX hY, integral_const_mul]

def reply (s : LayerState N m n) (t : ℕ) (ω : Ω) (j : Fin N) : E m n :=
  g.grad j s.w+O.value t j ω
def momentum (s : LayerState N m n) (t : ℕ) (ω : Ω) (j : Fin N) : E m n :=
  (1-c.β) • s.M j+c.β • reply g O s t ω j
def target (s : LayerState N m n) (t : ℕ) (ω : Ω) (j : Fin N) : E m n :=
  (1-c.θ) • momentum g c O s t ω j+c.θ • reply g O s t ω j
def upload (s : LayerState N m n) (t : ℕ) (ω : Ω) (j : Fin N) : E m n :=
  s.G j+c.CU t j ω (target g c O s t ω j-s.G j)+R.uplink t j ω
def estimate (s : LayerState N m n) (t : ℕ) (ω : Ω) : E m n :=
  average (upload g c O R s t ω)
def server (s : LayerState N m n) (t : ℕ) (ω : Ω) : E m n :=
  s.x-c.η t • c.direction t (estimate g c O R s t ω)
def advance (s : LayerState N m n) (t : ℕ) (ω : Ω) : LayerState N m n :=
  (server g c O R s t ω,
    s.w+c.CP t ω (server g c O R s t ω-s.w)+R.downlink t ω,
    momentum g c O s t ω, upload g c O R s t ω)

def run (x₀ : E m n) : ℕ → Ω → LayerState N m n
  | 0, _ => LayerState.start x₀
  | t+1, ω => advance g c O R (run x₀ t ω) t ω

def gapCap : ℝ := (c.qP*c.B*c.peak+R.epsP)/(1-c.qP)
def biasCap : ℝ := g.L*gapCap c R+g.L*c.B*c.peak/c.β
def incrementCap : ℝ :=
  g.L*c.B*c.peak+2*(c.β+(1-c.β)*c.θ)*g.L*gapCap c R+
    O.σ*((1-c.θ)*c.β*(1+Real.sqrt (c.β/(2-c.β)))+2*c.θ)
def noiseCap : ℝ := g.L*gapCap c R+(1-c.θ)*(1-c.β)*g.L*c.B*c.peak/c.β+
  O.σ*Real.sqrt (filterGain c.β c.θ/N)
def compressionCap : ℝ := c.qU/(1-c.qU)*incrementCap g c O R+R.epsU/(1-c.qU)

theorem gapCap_nonneg : 0 ≤ gapCap c R := by
  unfold gapCap
  exact div_nonneg (add_nonneg (mul_nonneg (mul_nonneg c.qP_nonneg c.B_nonneg)
    c.peak_nonneg) R.epsP_nonneg) (sub_pos.mpr c.qP_one).le

theorem biasCap_nonneg : 0 ≤ biasCap g c R := by
  unfold biasCap
  exact add_nonneg (mul_nonneg g.L_nonneg (gapCap_nonneg c R))
    (div_nonneg (mul_nonneg (mul_nonneg g.L_nonneg c.B_nonneg) c.peak_nonneg) c.β_pos.le)

omit [IsProbabilityMeasure μ] in
theorem server_movement (s : LayerState N m n) (t : ℕ) (ω : Ω) :
    ‖server g c O R s t ω-s.x‖ ≤ c.B*c.η t := by
  have he : server g c O R s t ω-s.x =
      -(c.η t • c.direction t (estimate g c O R s t ω)) := by unfold server; module
  rw [he, norm_neg, norm_smul, Real.norm_eq_abs, abs_of_nonneg (c.η_nonneg t)]
  nlinarith only [mul_le_mul_of_nonneg_left
    (c.direction_bound t (estimate g c O R s t ω)) (c.η_nonneg t)]

theorem run_gap (x₀ : E m n) (t : ℕ) (ω : Ω) :
    ‖(run g c O R x₀ t ω).x-(run g c O R x₀ t ω).w‖ ≤ gapCap c R := by
  have hp := gapCap_nonneg c R
  have hbal : (1-c.qP)*gapCap c R = c.qP*c.B*c.peak+R.epsP := by
    unfold gapCap
    exact mul_div_cancel₀ _ (ne_of_gt (sub_pos.mpr c.qP_one))
  induction t with
  | zero => simpa [run, LayerState.start, LayerState.x, LayerState.w] using hp
  | succ t ih =>
    let s := run g c O R x₀ t ω
    let x := server g c O R s t ω
    have hmove : ‖x-s.x‖ ≤ c.B*c.peak := (server_movement g c O R s t ω).trans
      (mul_le_mul_of_nonneg_left (c.η_peak t) c.B_nonneg)
    have hxw : ‖x-s.w‖ ≤ c.B*c.peak+gapCap c R := by
      have ht := norm_add_le (x-s.x) (s.x-s.w)
      rw [sub_add_sub_cancel] at ht
      linarith only [ht, hmove, ih]
    have he : x-(s.w+c.CP t ω (x-s.w)+R.downlink t ω) =
        (x-s.w-c.CP t ω (x-s.w))-R.downlink t ω := by module
    change ‖x-(s.w+c.CP t ω (x-s.w)+R.downlink t ω)‖ ≤ _
    rw [he]
    have h1 := norm_sub_le (x-s.w-c.CP t ω (x-s.w)) (R.downlink t ω)
    have h2 := c.contracts_CP t ω (x-s.w)
    have h3 := mul_le_mul_of_nonneg_left hxw c.qP_nonneg
    linarith only [h1, h2, h3, R.bound_downlink t ω, hbal]

theorem grad_memLp {W : Ω → E m n} (hW : MemLp W 2 μ) (j : Fin N) :
    MemLp (fun ω => g.grad j (W ω)) 2 μ := by
  have hm := ((g.measurable_local j).comp_aemeasurable
    hW.aestronglyMeasurable.aemeasurable).aestronglyMeasurable
  apply ((hW.norm.const_smul g.L).add (memLp_const ‖g.grad j 0‖)).of_le hm
  filter_upwards [] with ω
  change ‖g.grad j (W ω)‖ ≤ ‖g.L*‖W ω‖+‖g.grad j 0‖‖
  rw [Real.norm_of_nonneg (add_nonneg (mul_nonneg g.L_nonneg (norm_nonneg _)) (norm_nonneg _))]
  have ht := norm_sub_le (g.grad j (W ω)) (g.grad j 0)
  have h := g.lipschitz j (W ω) 0
  rw [sub_zero] at h
  have h' := norm_add_le (g.grad j (W ω)-g.grad j 0) (g.grad j 0)
  rw [sub_add_cancel] at h'
  linarith only [h, h']

omit [IsProbabilityMeasure μ] in
theorem compressor_memLp {C : Ω → E m n → E m n} (hm : Measurable (fun z : Ω × E m n => C z.1 z.2)) {q : ℝ}
    (hq : q ≤ 1) (hc : ∀ ω z, ‖z-C ω z‖ ≤ q*‖z‖)
    {X : Ω → E m n} (hX : MemLp X 2 μ) : MemLp (fun ω => C ω (X ω)) 2 μ := by
  have hres : MemLp (fun ω => X ω-C ω (X ω)) 2 μ := by
    apply hX.of_le (hX.aestronglyMeasurable.sub
      (hm.comp_aemeasurable (aemeasurable_id.prodMk hX.aestronglyMeasurable.aemeasurable)).aestronglyMeasurable)
    filter_upwards [] with ω
    exact (hc ω _).trans (by nlinarith only [mul_le_mul_of_nonneg_right hq (norm_nonneg (X ω))])
  have he : (fun ω => X ω-(X ω-C ω (X ω))) = (fun ω => C ω (X ω)) := by funext; module
  exact he ▸ hX.sub hres

/-- L2 integrability of every actual iterate and worker memory is derived from
the updates and primitive noise/operator assumptions. -/
theorem run_memLp (x₀ : E m n) (t : ℕ) :
    MemLp (fun ω => (run g c O R x₀ t ω).x) 2 μ ∧
    MemLp (fun ω => (run g c O R x₀ t ω).w) 2 μ ∧
    (∀ j, MemLp (fun ω => (run g c O R x₀ t ω).M j) 2 μ) ∧
    (∀ j, MemLp (fun ω => (run g c O R x₀ t ω).G j) 2 μ) := by
  induction t with
  | zero => exact ⟨memLp_const _, memLp_const _, fun _ => memLp_const _, fun _ => memLp_const _⟩
  | succ t ih =>
    let s := run g c O R x₀ t
    have ha j : MemLp (fun ω => reply g O (s ω) t ω j) 2 μ :=
      (grad_memLp g ih.2.1 j).add (O.memLp t j)
    have hM j : MemLp (fun ω => momentum g c O (s ω) t ω j) 2 μ := by
      change MemLp (fun ω => (1-c.β) • (s ω).M j+c.β • reply g O (s ω) t ω j) 2 μ
      exact ((ih.2.2.1 j).const_smul (1-c.β)).add ((ha j).const_smul c.β)
    have hV j : MemLp (fun ω => target g c O (s ω) t ω j) 2 μ := by
      change MemLp (fun ω => (1-c.θ) • momentum g c O (s ω) t ω j+
        c.θ • reply g O (s ω) t ω j) 2 μ
      exact ((hM j).const_smul (1-c.θ)).add ((ha j).const_smul c.θ)
    have hG j : MemLp (fun ω => upload g c O R (s ω) t ω j) 2 μ :=
      ((ih.2.2.2 j).add (compressor_memLp (c.measurable_CU t j) c.qU_one.le (c.contracts_CU t j)
        ((hV j).sub (ih.2.2.2 j)))).add
        (MemLp.of_bound (R.measurable_uplink t j).aestronglyMeasurable R.epsU
          (Filter.Eventually.of_forall (R.bound_uplink t j)))
    have hH : MemLp (fun ω => estimate g c O R (s ω) t ω) 2 μ := average_memLp hG
    have hD : MemLp (fun ω => c.direction t (estimate g c O R (s ω) t ω)) 2 μ :=
      MemLp.of_bound ((c.measurable_direction t).comp_aemeasurable
        hH.aestronglyMeasurable.aemeasurable).aestronglyMeasurable c.B
        (Filter.Eventually.of_forall (fun _ => c.direction_bound t _))
    have hX : MemLp (fun ω => server g c O R (s ω) t ω) 2 μ := ih.1.sub (hD.const_smul (c.η t))
    have hW := (ih.2.1.add (compressor_memLp (c.measurable_CP t) c.qP_one.le (c.contracts_CP t)
      (hX.sub ih.2.1))).add
      (MemLp.of_bound (R.measurable_downlink t).aestronglyMeasurable R.epsP
        (Filter.Eventually.of_forall (R.bound_downlink t)))
    exact ⟨hX, hW, hM, hG⟩

def noiseEMA (t : ℕ) (ω : Ω) (j : Fin N) : E m n :=
  ema c.β (fun s => O.value s j) t ω

omit [IsProbabilityMeasure μ] in
theorem average_noiseEMA (t : ℕ) (ω : Ω) :
    average (noiseEMA c O t ω) = ema c.β O.mean t ω := by
  induction t with
  | zero => simp [noiseEMA, ema, average]
  | succ t ih =>
    change average (fun j => (1-c.β) • noiseEMA c O t ω j+c.β • O.value t j ω) = _
    rw [average_add, average_smul, average_smul, ih]
    rfl

theorem run_bias (x₀ : E m n) (j : Fin N) (t : ℕ) (ω : Ω) :
    ‖(run g c O R x₀ t ω).M j-g.grad j (run g c O R x₀ t ω).x-noiseEMA c O t ω j‖ ≤
      (1-c.β)^t*‖g.grad j x₀‖+biasCap g c R := by
  apply momentum_bias_cap _ c.β_pos c.β_one (biasCap_nonneg g c R) (by
    simp [run, LayerState.start, LayerState.M, LayerState.x, noiseEMA, ema]) _ t
  intro k
  let s := run g c O R x₀ k ω
  have hb := radius_bias_bound c.β_pos.le c.β_one (s.M j) (noiseEMA c O k ω j)
    (g.grad j s.x) (g.grad j s.w) (g.grad j (server g c O R s k ω)) (O.value k j ω)
  have hgap := (g.lipschitz j s.w s.x).trans (mul_le_mul_of_nonneg_left
    (by simpa [norm_sub_rev] using run_gap g c O R x₀ k ω) g.L_nonneg)
  have hm : ‖s.x-server g c O R s k ω‖ ≤ c.B*c.peak := by
    rw [norm_sub_rev]
    exact (server_movement g c O R s k ω).trans
      (mul_le_mul_of_nonneg_left (c.η_peak k) c.B_nonneg)
  have hmove := (g.lipschitz j s.x (server g c O R s k ω)).trans
    (mul_le_mul_of_nonneg_left hm g.L_nonneg)
  have h1 := mul_le_mul_of_nonneg_left hgap c.β_pos.le
  change ‖((1-c.β) • s.M j+c.β • (g.grad j s.w+O.value k j ω))-
    g.grad j (server g c O R s k ω)-((1-c.β) • noiseEMA c O k ω j+c.β • O.value k j ω)‖ ≤ _
  have hbal : c.β*biasCap g c R = c.β*(g.L*gapCap c R)+g.L*c.B*c.peak := by
    unfold biasCap
    field_simp [ne_of_gt c.β_pos]
  linarith only [hb, h1, hmove, hbal]

theorem mean_gradient_lipschitz (x y : E m n) :
    ‖g.mean x-g.mean y‖ ≤ g.L*‖x-y‖ := by
  rw [Gradients.mean, Gradients.mean, ← average_sub]
  exact (norm_average_le _).trans ((average_mono (fun j => g.lipschitz j x y)).trans_eq
    (average_const g.workers_pos _))

omit [IsProbabilityMeasure μ] in
theorem average_momentum (s : LayerState N m n) (t : ℕ) (ω : Ω) :
    average (momentum g c O s t ω) =
      (1-c.β) • average s.M+c.β • (g.mean s.w+O.mean t ω) := by
  unfold momentum reply
  rw [average_add, average_smul, average_smul, average_add]
  rfl

theorem run_mean_bias (x₀ : E m n) (t : ℕ) (ω : Ω) :
    ‖average (run g c O R x₀ t ω).M-g.mean (run g c O R x₀ t ω).x-ema c.β O.mean t ω‖ ≤
      (1-c.β)^t*‖g.mean x₀‖+biasCap g c R := by
  apply momentum_bias_cap _ c.β_pos c.β_one (biasCap_nonneg g c R) (by
    simp [run, LayerState.start, LayerState.M, LayerState.x, average, ema]) _ t
  intro k
  let s := run g c O R x₀ k ω
  have hb := radius_bias_bound c.β_pos.le c.β_one (average s.M) (ema c.β O.mean k ω)
    (g.mean s.x) (g.mean s.w) (g.mean (server g c O R s k ω)) (O.mean k ω)
  have hgap := (mean_gradient_lipschitz g s.w s.x).trans (mul_le_mul_of_nonneg_left
    (by simpa [norm_sub_rev] using run_gap g c O R x₀ k ω) g.L_nonneg)
  have hm : ‖s.x-server g c O R s k ω‖ ≤ c.B*c.peak := by
    rw [norm_sub_rev]
    exact (server_movement g c O R s k ω).trans
      (mul_le_mul_of_nonneg_left (c.η_peak k) c.B_nonneg)
  have hmove := (mean_gradient_lipschitz g s.x (server g c O R s k ω)).trans
    (mul_le_mul_of_nonneg_left hm g.L_nonneg)
  have h1 := mul_le_mul_of_nonneg_left hgap c.β_pos.le
  change ‖average (momentum g c O s k ω)-g.mean (server g c O R s k ω)-
    ((1-c.β) • ema c.β O.mean k ω+c.β • O.mean k ω)‖ ≤ _
  rw [average_momentum]
  have hbal : c.β*biasCap g c R = c.β*(g.L*gapCap c R)+g.L*c.B*c.peak := by
    unfold biasCap
    field_simp [ne_of_gt c.β_pos]
  linarith only [hb, h1, hmove, hbal]

theorem reply_memLp (x₀ : E m n) (t : ℕ) (j : Fin N) :
    MemLp (fun ω => reply g O (run g c O R x₀ t ω) t ω j) 2 μ :=
  (grad_memLp g (run_memLp g c O R x₀ t).2.1 j).add (O.memLp t j)

theorem target_memLp (x₀ : E m n) (t : ℕ) (j : Fin N) :
    MemLp (fun ω => target g c O (run g c O R x₀ t ω) t ω j) 2 μ := by
  have hM : MemLp (fun ω => momentum g c O (run g c O R x₀ t ω) t ω j) 2 μ :=
    (((run_memLp g c O R x₀ t).2.2.1 j).const_smul (1-c.β)).add
      ((reply_memLp g c O R x₀ t j).const_smul c.β)
  exact (hM.const_smul (1-c.θ)).add ((reply_memLp g c O R x₀ t j).const_smul c.θ)

theorem estimate_memLp (x₀ : E m n) (t : ℕ) :
    MemLp (fun ω => estimate g c O R (run g c O R x₀ t ω) t ω) 2 μ :=
  average_memLp (run_memLp g c O R x₀ (t+1)).2.2.2

def previousTarget (x₀ : E m n) : ℕ → Ω → Fin N → E m n
  | 0, _, _ => 0
  | t+1, ω, j => target g c O (run g c O R x₀ t ω) t ω j

theorem previousTarget_memLp (x₀ : E m n) (t : ℕ) (j : Fin N) :
    MemLp (fun ω => previousTarget g c O R x₀ t ω j) 2 μ := by
  cases t with
  | zero => exact memLp_const _
  | succ t => exact target_memLp g c O R x₀ t j

def uplinkError (x₀ : E m n) (t : ℕ) : ℝ :=
  ∫ ω, average (fun j => ‖(run g c O R x₀ t ω).G j-previousTarget g c O R x₀ t ω j‖) ∂μ
def estimatorError (x₀ : E m n) (t : ℕ) : ℝ :=
  ∫ ω, ‖estimate g c O R (run g c O R x₀ t ω) t ω-g.mean (run g c O R x₀ t ω).x‖ ∂μ

theorem uplinkError_integrable (x₀ : E m n) (t : ℕ) :
    Integrable (fun ω => average (fun j =>
      ‖(run g c O R x₀ t ω).G j-previousTarget g c O R x₀ t ω j‖)) μ :=
  (average_memLp (fun j => ((run_memLp g c O R x₀ t).2.2.2 j).sub
    (previousTarget_memLp g c O R x₀ t j) |>.norm)).integrable (by norm_num)

theorem mean_gradient_memLp (x₀ : E m n) (t : ℕ) :
    MemLp (fun ω => g.mean (run g c O R x₀ t ω).x) 2 μ :=
  average_memLp (fun j => grad_memLp g (run_memLp g c O R x₀ t).1 j)

theorem estimatorError_integrable (x₀ : E m n) (t : ℕ) :
    Integrable (fun ω =>
      ‖estimate g c O R (run g c O R x₀ t ω) t ω-g.mean (run g c O R x₀ t ω).x‖) μ :=
  ((estimate_memLp g c O R x₀ t).sub (mean_gradient_memLp g c O R x₀ t)).integrable
    (by norm_num) |>.norm

/-- The old momentum is compared with the current reply after removing its
noise EMA. This preserves the square-root momentum gain in the noise term. -/
theorem reply_momentum_bound (x₀ : E m n) (t : ℕ) (j : Fin N) :
    (∫ ω, ‖reply g O (run g c O R x₀ t ω) t ω j-(run g c O R x₀ t ω).M j‖ ∂μ) ≤
      (1-c.β)^t*‖g.grad j x₀‖+biasCap g c R+g.L*gapCap c R+
        O.σ*Real.sqrt (c.β/(2-c.β))+O.σ := by
  let s := run g c O R x₀ t
  let Z := fun ω => noiseEMA c O t ω j
  have hZ := ema_memLp c.β (fun k => O.memLp k j) t
  have hl := ((reply_memLp g c O R x₀ t j).sub
    ((run_memLp g c O R x₀ t).2.2.1 j)).integrable (by norm_num) |>.norm
  have hr : Integrable (fun ω => (1-c.β)^t*‖g.grad j x₀‖+biasCap g c R+g.L*gapCap c R+
      ‖Z ω‖+‖O.value t j ω‖) μ :=
    ((integrable_const _).add (hZ.integrable (by norm_num)).norm).add
      ((O.memLp t j).integrable (by norm_num)).norm
  have hp ω : ‖reply g O (s ω) t ω j-(s ω).M j‖ ≤
      (1-c.β)^t*‖g.grad j x₀‖+biasCap g c R+g.L*gapCap c R+‖Z ω‖+‖O.value t j ω‖ := by
    have he : reply g O (s ω) t ω j-(s ω).M j =
        (g.grad j (s ω).w-g.grad j (s ω).x)-
          ((s ω).M j-g.grad j (s ω).x-Z ω)-Z ω+O.value t j ω := by unfold reply; module
    rw [he]
    have h1 := norm_add_le ((g.grad j (s ω).w-g.grad j (s ω).x)-
      ((s ω).M j-g.grad j (s ω).x-Z ω)-Z ω) (O.value t j ω)
    have h2 := norm_sub_le ((g.grad j (s ω).w-g.grad j (s ω).x)-
      ((s ω).M j-g.grad j (s ω).x-Z ω)) (Z ω)
    have h3 := norm_sub_le (g.grad j (s ω).w-g.grad j (s ω).x)
      ((s ω).M j-g.grad j (s ω).x-Z ω)
    have hgap := (g.lipschitz j (s ω).w (s ω).x).trans (mul_le_mul_of_nonneg_left
      (by simpa [norm_sub_rev] using run_gap g c O R x₀ t ω) g.L_nonneg)
    have hb := run_bias g c O R x₀ j t ω
    linarith only [h1, h2, h3, hgap, hb]
  have hi := integral_mono hl hr hp
  simp only [Pi.sub_apply] at hi
  have he := expectation_affine ((1-c.β)^t*‖g.grad j x₀‖+biasCap g c R+g.L*gapCap c R)
    1 1 (fun ω => ‖Z ω‖) (fun ω => ‖O.value t j ω‖)
    (hZ.integrable (by norm_num)).norm ((O.memLp t j).integrable (by norm_num)).norm
  simp only [one_mul] at he
  rw [he] at hi
  have hn := O.ema_norm_bound c.β_pos c.β_one t j
  change (∫ ω, ‖Z ω‖ ∂μ) ≤ _ at hn
  linarith only [hi, hn, O.norm_bound t j]

theorem replies_increment_bound (x₀ : E m n) (t : ℕ) (j : Fin N) :
    (∫ ω, ‖reply g O (run g c O R x₀ (t+1) ω) (t+1) ω j-
      reply g O (run g c O R x₀ t ω) t ω j‖ ∂μ) ≤
      g.L*(c.B*c.peak+2*gapCap c R)+2*O.σ := by
  let s := run g c O R x₀ t
  let s' := run g c O R x₀ (t+1)
  have hl := ((reply_memLp g c O R x₀ (t+1) j).sub
    (reply_memLp g c O R x₀ t j)).integrable (by norm_num) |>.norm
  have hr : Integrable (fun ω => g.L*(c.B*c.peak+2*gapCap c R)+
      ‖O.value (t+1) j ω‖+‖O.value t j ω‖) μ :=
    ((integrable_const _).add ((O.memLp (t+1) j).integrable (by norm_num)).norm).add
      ((O.memLp t j).integrable (by norm_num)).norm
  have hp ω : ‖reply g O (s' ω) (t+1) ω j-reply g O (s ω) t ω j‖ ≤
      g.L*(c.B*c.peak+2*gapCap c R)+‖O.value (t+1) j ω‖+‖O.value t j ω‖ := by
    have hquery : ‖(s' ω).w-(s ω).w‖ ≤ c.B*c.peak+2*gapCap c R := by
      have he : (s' ω).w-(s ω).w = ((s' ω).w-(s' ω).x)+
          ((s' ω).x-(s ω).x)+((s ω).x-(s ω).w) := by module
      rw [he]
      have h1 := norm_add_le ((s' ω).w-(s' ω).x+((s' ω).x-(s ω).x)) ((s ω).x-(s ω).w)
      have h2 := norm_add_le ((s' ω).w-(s' ω).x) ((s' ω).x-(s ω).x)
      have h3 := run_gap g c O R x₀ (t+1) ω
      rw [norm_sub_rev] at h3
      have h4 := run_gap g c O R x₀ t ω
      have h5 := (server_movement g c O R (s ω) t ω).trans
        (mul_le_mul_of_nonneg_left (c.η_peak t) c.B_nonneg)
      change ‖(s' ω).x-(s ω).x‖ ≤ c.B*c.peak at h5
      linarith only [h1, h2, h3, h4, h5]
    have hg := (g.lipschitz j (s' ω).w (s ω).w).trans
      (mul_le_mul_of_nonneg_left hquery g.L_nonneg)
    have he : reply g O (s' ω) (t+1) ω j-reply g O (s ω) t ω j =
        (g.grad j (s' ω).w-g.grad j (s ω).w)+O.value (t+1) j ω-O.value t j ω := by
      unfold reply; module
    rw [he]
    linarith only [norm_sub_le ((g.grad j (s' ω).w-g.grad j (s ω).w)+O.value (t+1) j ω)
      (O.value t j ω), norm_add_le (g.grad j (s' ω).w-g.grad j (s ω).w)
      (O.value (t+1) j ω), hg]
  have hi := integral_mono hl hr hp
  simp only [Pi.sub_apply] at hi
  have he := expectation_affine (g.L*(c.B*c.peak+2*gapCap c R)) 1 1
    (fun ω => ‖O.value (t+1) j ω‖) (fun ω => ‖O.value t j ω‖)
    ((O.memLp (t+1) j).integrable (by norm_num)).norm ((O.memLp t j).integrable (by norm_num)).norm
  simp only [one_mul] at he
  rw [he] at hi
  linarith only [hi, O.norm_bound (t+1) j, O.norm_bound t j]

omit [IsProbabilityMeasure μ] in
theorem incrementCap_nonneg : 0 ≤ incrementCap g c O R := by
  unfold incrementCap
  have := gapCap_nonneg c R
  have := g.L_nonneg
  have := c.B_nonneg
  have := c.peak_nonneg
  have := c.β_pos.le
  have := sub_nonneg.mpr c.β_one
  have := c.θ_nonneg
  have := sub_nonneg.mpr c.θ_one
  have := O.σ_nonneg
  positivity

theorem target_increment_expected (x₀ : E m n) (t : ℕ) (j : Fin N) :
    (∫ ω, ‖target g c O (run g c O R x₀ t ω) t ω j-
      previousTarget g c O R x₀ t ω j‖ ∂μ) ≤
      incrementCap g c O R+initialImpulse c.β c.θ ‖g.grad j x₀‖ t := by
  have hθ := sub_nonneg.mpr c.θ_one
  have hμ := sub_nonneg.mpr c.β_one
  have hL := g.L_nonneg
  have hB := c.B_nonneg
  have hp := c.peak_nonneg
  have hσ := O.σ_nonneg
  have hβ := c.β_pos.le
  have hk : 0 ≤ c.β+(1-c.β)*c.θ := add_nonneg c.β_pos.le (mul_nonneg hμ c.θ_nonneg)
  cases t with
  | zero =>
    have hl := ((target_memLp g c O R x₀ 0 j).sub
      (previousTarget_memLp g c O R x₀ 0 j)).integrable (by norm_num) |>.norm
    have hr : Integrable (fun ω => (c.β+(1-c.β)*c.θ)*
        (‖g.grad j x₀‖+‖O.value 0 j ω‖)) μ :=
      ((integrable_const _).add ((O.memLp 0 j).integrable (by norm_num)).norm).const_mul _
    have hp ω : ‖target g c O (run g c O R x₀ 0 ω) 0 ω j-
        previousTarget g c O R x₀ 0 ω j‖ ≤
        (c.β+(1-c.β)*c.θ)*(‖g.grad j x₀‖+‖O.value 0 j ω‖) := by
      have he : target g c O (run g c O R x₀ 0 ω) 0 ω j-
          previousTarget g c O R x₀ 0 ω j =
          (c.β+(1-c.β)*c.θ) • (g.grad j x₀+O.value 0 j ω) := by
        dsimp [target, momentum, reply, run, LayerState.start, LayerState.M,
          LayerState.w, previousTarget]
        module
      rw [he, norm_smul, Real.norm_eq_abs, abs_of_nonneg hk]
      exact mul_le_mul_of_nonneg_left (norm_add_le _ _) hk
    have hi := integral_mono hl hr hp
    simp only [Pi.sub_apply] at hi
    rw [integral_const_mul, integral_add (integrable_const _)
      ((O.memLp 0 j).integrable (by norm_num)).norm, integral_const] at hi
    simp only [probReal_univ, one_smul] at hi
    have hn := mul_le_mul_of_nonneg_left (O.norm_bound 0 j) hk
    have hJ : (c.β+(1-c.β)*c.θ)*O.σ ≤ incrementCap g c O R := by
      unfold incrementCap
      have h1 : 0 ≤ g.L*c.B*c.peak := by positivity
      have h2 : 0 ≤ 2*(c.β+(1-c.β)*c.θ)*g.L*gapCap c R :=
        mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) hk) g.L_nonneg) (gapCap_nonneg c R)
      have h3 : 0 ≤ O.σ*((1-c.θ)*c.β*Real.sqrt (c.β/(2-c.β))) := by positivity
      have h4 := mul_nonneg O.σ_nonneg c.θ_nonneg
      nlinarith only [h1, h2, h3, h4]
    simp only [initialImpulse, pow_zero, mul_one, ↓reduceIte]
    nlinarith only [hi, hn, hJ]
  | succ t =>
    let s := run g c O R x₀ (t+1)
    let old := run g c O R x₀ t
    have hleft := ((target_memLp g c O R x₀ (t+1) j).sub
      (previousTarget_memLp g c O R x₀ (t+1) j)).integrable (by norm_num) |>.norm
    have hfirst := ((reply_memLp g c O R x₀ (t+1) j).sub
      ((run_memLp g c O R x₀ (t+1)).2.2.1 j)).integrable (by norm_num) |>.norm
    have hsecond := ((reply_memLp g c O R x₀ (t+1) j).sub
      (reply_memLp g c O R x₀ t j)).integrable (by norm_num) |>.norm
    have hi := integral_mono hleft ((hfirst.const_mul ((1-c.θ)*c.β)).add
      (hsecond.const_mul c.θ)) (fun ω => by
        exact target_increment_bound c.β_pos.le c.θ_nonneg c.θ_one
          ((s ω).M j) (reply g O (s ω) (t+1) ω j) (reply g O (old ω) t ω j))
    simp only [Pi.add_apply, Pi.sub_apply] at hi
    have he := expectation_affine 0 ((1-c.θ)*c.β) c.θ
      (fun ω => ‖reply g O (s ω) (t+1) ω j-(s ω).M j‖)
      (fun ω => ‖reply g O (s ω) (t+1) ω j-reply g O (old ω) t ω j‖) hfirst hsecond
    simp only [zero_add] at he
    rw [he] at hi
    have h1 := mul_le_mul_of_nonneg_left (reply_momentum_bound g c O R x₀ (t+1) j)
      (mul_nonneg hθ c.β_pos.le)
    have h2 := mul_le_mul_of_nonneg_left (replies_increment_bound g c O R x₀ t j) c.θ_nonneg
    apply hi.trans
    convert add_le_add h1 h2 using 1
    simp only [initialImpulse, Nat.succ_ne_zero, ↓reduceIte, add_zero]
    unfold incrementCap biasCap
    field_simp [ne_of_gt c.β_pos]
    <;> ring

def initialLocal (x₀ : E m n) : ℝ := average (fun j => ‖g.grad j x₀‖)

theorem target_increment_average (x₀ : E m n) (t : ℕ) :
    (∫ ω, average (fun j => ‖target g c O (run g c O R x₀ t ω) t ω j-
      previousTarget g c O R x₀ t ω j‖) ∂μ) ≤
      incrementCap g c O R+initialImpulse c.β c.θ (initialLocal g x₀) t := by
  have hInt j : Integrable (fun ω => ‖target g c O (run g c O R x₀ t ω) t ω j-
      previousTarget g c O R x₀ t ω j‖) μ :=
    ((target_memLp g c O R x₀ t j).sub
      (previousTarget_memLp g c O R x₀ t j)).integrable (by norm_num) |>.norm
  rw [integral_average hInt]
  apply (average_mono (target_increment_expected g c O R x₀ t)).trans_eq
  rw [average_add, average_const g.workers_pos]
  congr 1
  unfold initialImpulse initialLocal
  rw [average_add, average_mul]
  split_ifs
  · rw [average_mul]
  · rw [average_const g.workers_pos]

/-- The compression recurrence is proved for the worker-resolved run. -/
theorem uplinkError_step (x₀ : E m n) (t : ℕ) :
    uplinkError g c O R x₀ (t+1) ≤ c.qU*(uplinkError g c O R x₀ t+
      incrementCap g c O R+initialImpulse c.β c.θ (initialLocal g x₀) t)+R.epsU := by
  let s := run g c O R x₀ t
  let V := fun ω j => target g c O (s ω) t ω j
  let prev := previousTarget g c O R x₀ t
  have hinc : Integrable (fun ω => average (fun j => ‖V ω j-prev ω j‖)) μ :=
    (average_memLp (fun j => ((target_memLp g c O R x₀ t j).sub
      (previousTarget_memLp g c O R x₀ t j)).norm)).integrable (by norm_num)
  have hr := (((uplinkError_integrable g c O R x₀ t).add hinc).const_mul c.qU).add
    (integrable_const R.epsU)
  have hp ω : average (fun j => ‖(run g c O R x₀ (t+1) ω).G j-
      previousTarget g c O R x₀ (t+1) ω j‖) ≤
      c.qU*(average (fun j => ‖(s ω).G j-prev ω j‖)+
        average (fun j => ‖V ω j-prev ω j‖))+R.epsU := by
    have hj j : ‖(run g c O R x₀ (t+1) ω).G j-
        previousTarget g c O R x₀ (t+1) ω j‖ ≤
        c.qU*(‖(s ω).G j-prev ω j‖+‖V ω j-prev ω j‖)+R.epsU := by
      have he : upload g c O R (s ω) t ω j-V ω j =
          -(V ω j-(s ω).G j-c.CU t j ω (V ω j-(s ω).G j))+R.uplink t j ω := by
        unfold upload; change (s ω).G j+c.CU t j ω (V ω j-(s ω).G j)+R.uplink t j ω-V ω j = _
        module
      change ‖upload g c O R (s ω) t ω j-V ω j‖ ≤ _
      rw [he]
      have h1 := norm_add_le (-(V ω j-(s ω).G j-c.CU t j ω (V ω j-(s ω).G j))) (R.uplink t j ω)
      rw [norm_neg] at h1
      have h2 := c.contracts_CU t j ω (V ω j-(s ω).G j)
      have h3 := norm_sub_le (V ω j-prev ω j) ((s ω).G j-prev ω j)
      rw [sub_sub_sub_cancel_right] at h3
      have h4 := mul_le_mul_of_nonneg_left h3 c.qU_nonneg
      linarith only [h1, h2, h4, R.bound_uplink t j ω]
    have h := average_mono hj
    rw [average_add, average_mul, average_add, average_const g.workers_pos] at h
    exact h
  have hi := integral_mono (uplinkError_integrable g c O R x₀ (t+1)) hr hp
  simp only [Pi.add_apply] at hi
  rw [expectation_add
    (fun ω => c.qU*(average (fun j => ‖(s ω).G j-prev ω j‖)+
      average (fun j => ‖V ω j-prev ω j‖))) (fun _ => R.epsU)
    (((uplinkError_integrable g c O R x₀ t).add hinc).const_mul c.qU) (integrable_const R.epsU),
    integral_const_mul,
    expectation_add (fun ω => average (fun j => ‖(s ω).G j-prev ω j‖))
      (fun ω => average (fun j => ‖V ω j-prev ω j‖))
      (uplinkError_integrable g c O R x₀ t) hinc, integral_const] at hi
  simp only [probReal_univ, one_smul] at hi
  have hm := mul_le_mul_of_nonneg_left (target_increment_average g c O R x₀ t) c.qU_nonneg
  exact hi.trans (by dsimp [uplinkError] at *; linarith only [hm])

theorem average_target (s : LayerState N m n) (t : ℕ) (ω : Ω) :
    average (target g c O s t ω) =
      (1-c.θ) • ((1-c.β) • average s.M+c.β • (g.mean s.w+O.mean t ω))+
        c.θ • (g.mean s.w+O.mean t ω) := by
  unfold target
  rw [average_add, average_smul, average_smul, average_momentum]
  congr 1
  unfold reply
  rw [average_add]
  rfl

theorem estimatorError_step (x₀ : E m n) (t : ℕ) :
    estimatorError g c O R x₀ t ≤ uplinkError g c O R x₀ (t+1)+
      (1-c.θ)*(1-c.β)^(t+1)*‖g.mean x₀‖+noiseCap g c O R := by
  let s := run g c O R x₀ t
  let a := (1-c.θ)*(1-c.β)
  let k := c.β+(1-c.β)*c.θ
  let Y := fun ω => a • ema c.β O.mean t ω+k • O.mean t ω
  have ha : 0 ≤ a := mul_nonneg (sub_nonneg.mpr c.θ_one) (sub_nonneg.mpr c.β_one)
  have hk : 0 ≤ k := add_nonneg c.β_pos.le (mul_nonneg (sub_nonneg.mpr c.β_one) c.θ_nonneg)
  have hY : MemLp Y 2 μ :=
    ((ema_memLp c.β O.mean_memLp t).const_smul a).add ((O.mean_memLp t).const_smul k)
  have hpt ω : ‖average (target g c O (s ω) t ω)-g.mean (s ω).x‖ ≤
      a*((1-c.β)^t*‖g.mean x₀‖+biasCap g c R)+k*(g.L*gapCap c R)+‖Y ω‖ := by
    rw [average_target, filtered_target_identity]
    have h1 := norm_add_le (a • (average (s ω).M-g.mean (s ω).x-ema c.β O.mean t ω)+
      k • (g.mean (s ω).w-g.mean (s ω).x)) (Y ω)
    have h2 := norm_add_le (a • (average (s ω).M-g.mean (s ω).x-ema c.β O.mean t ω))
      (k • (g.mean (s ω).w-g.mean (s ω).x))
    simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg ha, abs_of_nonneg hk] at h2
    have h3 := mul_le_mul_of_nonneg_left (run_mean_bias g c O R x₀ t ω) ha
    have h4 := mul_le_mul_of_nonneg_left ((mean_gradient_lipschitz g (s ω).w (s ω).x).trans
      (mul_le_mul_of_nonneg_left (by simpa [norm_sub_rev] using run_gap g c O R x₀ t ω)
        g.L_nonneg)) hk
    linarith only [h1, h2, h3, h4]
  have hp ω : ‖estimate g c O R (s ω) t ω-g.mean (s ω).x‖ ≤
      average (fun j => ‖(run g c O R x₀ (t+1) ω).G j-
        previousTarget g c O R x₀ (t+1) ω j‖)+
      (a*((1-c.β)^t*‖g.mean x₀‖+biasCap g c R)+k*(g.L*gapCap c R))+‖Y ω‖ := by
    have he : estimate g c O R (s ω) t ω-g.mean (s ω).x =
        (estimate g c O R (s ω) t ω-average (target g c O (s ω) t ω))+
          (average (target g c O (s ω) t ω)-g.mean (s ω).x) := by module
    have h1 := norm_add_le (estimate g c O R (s ω) t ω-average (target g c O (s ω) t ω))
      (average (target g c O (s ω) t ω)-g.mean (s ω).x)
    have h2 : ‖estimate g c O R (s ω) t ω-average (target g c O (s ω) t ω)‖ ≤
        average (fun j => ‖(run g c O R x₀ (t+1) ω).G j-
          previousTarget g c O R x₀ (t+1) ω j‖) := by
      change ‖average (upload g c O R (s ω) t ω)-average (target g c O (s ω) t ω)‖ ≤ _
      rw [← average_sub]
      exact norm_average_le _
    rw [he]
    linarith only [h1, h2, hpt ω]
  have hi := integral_mono (estimatorError_integrable g c O R x₀ t)
    (((uplinkError_integrable g c O R x₀ (t+1)).add (integrable_const _)).add
      (hY.integrable (by norm_num)).norm) hp
  change estimatorError g c O R x₀ t ≤ (∫ ω,
    average (fun j => ‖(run g c O R x₀ (t+1) ω).G j-
      previousTarget g c O R x₀ (t+1) ω j‖)+
    (a*((1-c.β)^t*‖g.mean x₀‖+biasCap g c R)+k*(g.L*gapCap c R))+‖Y ω‖ ∂μ) at hi
  have he := expectation_affine
    (a*((1-c.β)^t*‖g.mean x₀‖+biasCap g c R)+k*(g.L*gapCap c R)) 1 1
    (fun ω => average (fun j => ‖(run g c O R x₀ (t+1) ω).G j-
      previousTarget g c O R x₀ (t+1) ω j‖)) (fun ω => ‖Y ω‖)
    (uplinkError_integrable g c O R x₀ (t+1)) (hY.integrable (by norm_num)).norm
  simp only [one_mul] at he
  have hi' : estimatorError g c O R x₀ t ≤
      (∫ ω, (a*((1-c.β)^t*‖g.mean x₀‖+biasCap g c R)+k*(g.L*gapCap c R))+
        average (fun j => ‖(run g c O R x₀ (t+1) ω).G j-
          previousTarget g c O R x₀ (t+1) ω j‖)+‖Y ω‖ ∂μ) := by
    simpa only [add_comm, add_left_comm, add_assoc] using hi
  rw [he] at hi'
  change estimatorError g c O R x₀ t ≤
    a*((1-c.β)^t*‖g.mean x₀‖+biasCap g c R)+k*(g.L*gapCap c R)+
      uplinkError g c O R x₀ (t+1)+(∫ ω, ‖Y ω‖ ∂μ) at hi'
  have hn := O.filtered_mean_bound g.workers_pos c.β_pos c.β_one (θ := c.θ) t
  have heq : a*((1-c.β)^t*‖g.mean x₀‖+biasCap g c R)+k*(g.L*gapCap c R)+
      O.σ*Real.sqrt (filterGain c.β c.θ/N) =
      (1-c.θ)*(1-c.β)^(t+1)*‖g.mean x₀‖+noiseCap g c O R := by
    dsimp [a, k, biasCap, noiseCap]
    rw [pow_succ]
    field_simp [ne_of_gt c.β_pos]
    <;> ring
  change (∫ ω, ‖Y ω‖ ∂μ) ≤ _ at hn
  linarith only [hi', hn, heq]

def trackingBudget (x₀ : E m n) (T : ℕ) : ℝ :=
  c.peak*((1-c.θ)*(1-c.β)*‖g.mean x₀‖/c.β+c.qU/(1-c.qU)*initialLocal g x₀)+
    (∑ t ∈ Finset.range T, c.η t)*(noiseCap g c O R+compressionCap g c O R)

theorem compressionCap_nonneg : 0 ≤ compressionCap g c O R := by
  unfold compressionCap
  exact add_nonneg (mul_nonneg (div_nonneg c.qU_nonneg (sub_pos.mpr c.qU_one).le)
    (incrementCap_nonneg g c O R)) (div_nonneg R.epsU_nonneg (sub_pos.mpr c.qU_one).le)

omit [IsProbabilityMeasure μ] in
theorem compressionCap_balance :
    (1-c.qU)*compressionCap g c O R = c.qU*incrementCap g c O R+R.epsU := by
  unfold compressionCap
  field_simp [ne_of_gt (sub_pos.mpr c.qU_one)]

/-- Closed tracking for the full random N-worker trajectory. In particular,
neither a compression recurrence nor an estimator-error bound is a premise. -/
theorem tracking (x₀ : E m n) (T : ℕ) :
    (∑ t ∈ Finset.range T, c.η t*estimatorError g c O R x₀ t) ≤
      trackingBudget g c O R x₀ T := by
  apply fixed_momentum_error_sum c.η (uplinkError g c O R x₀) (estimatorError g c O R x₀)
    c.β_pos c.β_one c.θ_nonneg c.θ_one c.qU_nonneg c.qU_one (norm_nonneg _)
    (Distributed.average_nonneg (fun _ => norm_nonneg _)) c.peak_nonneg c.η_nonneg c.η_peak
    (compressionCap_nonneg g c O R) (compressionCap_balance g c O R)
    (by simp [uplinkError, run, previousTarget, LayerState.start, LayerState.G, average])
    (uplinkError_step g c O R x₀) (estimatorError_step g c O R x₀) T

/-- Primitive objective and numerical direction assumptions. The smoothness
inequality is the usual Frobenius descent inequality for every pair of points. -/
structure Objective (g : Gradients N m n) (c : Controls N m n Ω) where
  f : E m n → ℝ
  floor : ℝ
  Lf : ℝ
  ρ : ℝ
  r : ℝ
  gain : ℝ
  δ : ℕ → ℝ
  measurable_f : Measurable f
  derivative : ∀ x, HasFDerivAt f (innerSL ℝ (g.mean x)) x
  lower : ∀ x, floor ≤ f x
  Lf_nonneg : 0 ≤ Lf
  ρ_nonneg : 0 ≤ ρ
  r_nonneg : 0 ≤ r
  gain_pos : 0 < gain
  δ_nonneg : ∀ t, 0 ≤ δ t
  dual : ∀ z : E m n, nuclearNorm z ≤ ρ*‖z‖
  smooth : ∀ x y, f y ≤ f x+inner ℝ (g.mean x) (y-x)+Lf/2*‖y-x‖^2
  direction_radius : ∀ t h, spectralNorm (c.direction t h) ≤ r
  direction_pairing : ∀ t h, gain*nuclearNorm h-δ t ≤ inner ℝ h (c.direction t h)

variable (f : Objective g c)

theorem objective_integrable (x₀ : E m n) (t : ℕ) :
    Integrable (fun ω => f.f (run g c O R x₀ t ω).x) μ := by
  let X := fun ω => (run g c O R x₀ t ω).x
  have hX := (run_memLp g c O R x₀ t).1
  have hi : Integrable (fun ω => |f.f 0|+|f.floor|+
      ‖g.mean 0‖*‖X ω‖+f.Lf/2*‖X ω‖^2) μ :=
    ((integrable_const _).add ((hX.integrable (by norm_num)).norm.const_mul _)).add
      ((square_integrable hX).const_mul _)
  apply hi.mono' (f.measurable_f.comp_aemeasurable
    hX.aestronglyMeasurable.aemeasurable).aestronglyMeasurable
  filter_upwards [] with ω
  change |f.f (X ω)| ≤ _
  apply abs_le.mpr
  have hpos : 0 ≤ ‖g.mean 0‖*‖X ω‖+f.Lf/2*‖X ω‖^2 := by
    exact add_nonneg (mul_nonneg (norm_nonneg _) (norm_nonneg _))
      (mul_nonneg (div_nonneg f.Lf_nonneg (by norm_num)) (sq_nonneg _))
  have hs := f.smooth 0 (X ω)
  rw [sub_zero] at hs
  have hin := real_inner_le_norm (g.mean 0) (X ω)
  constructor
  · linarith only [f.lower (X ω), neg_abs_le f.floor, abs_nonneg (f.f 0), hpos]
  · linarith only [hs, hin, le_abs_self (f.f 0), abs_nonneg f.floor]

include f in
theorem stationarity_integrable (x₀ : E m n) (t : ℕ) :
    Integrable (fun ω => nuclearNorm (g.mean (run g c O R x₀ t ω).x)) μ := by
  have hX := mean_gradient_memLp g c O R x₀ t
  apply ((hX.integrable (by norm_num)).norm.const_mul f.ρ).mono'
    (continuous_nuclearNorm.measurable.comp_aemeasurable
      hX.aestronglyMeasurable.aemeasurable).aestronglyMeasurable
  filter_upwards [] with ω
  change ‖nuclearNorm (g.mean (run g c O R x₀ t ω).x)‖ ≤ _
  rw [Real.norm_of_nonneg (nuclearNorm_nonneg _)]
  exact f.dual _

omit [IsProbabilityMeasure μ] in
theorem objective_step (x₀ : E m n) (t : ℕ) (ω : Ω) :
    f.f (run g c O R x₀ (t+1) ω).x ≤ f.f (run g c O R x₀ t ω).x-
      c.η t*(f.gain*nuclearNorm (g.mean (run g c O R x₀ t ω).x))+
      c.η t*((f.gain+f.r)*f.ρ*
        ‖estimate g c O R (run g c O R x₀ t ω) t ω-g.mean (run g c O R x₀ t ω).x‖+
          f.δ t)+f.Lf*c.B^2/2*(c.η t)^2 := by
  let s := run g c O R x₀ t ω
  let H := estimate g c O R s t ω
  let D := c.direction t H
  have he : server g c O R s t ω-s.x = -(c.η t • D) := by unfold server; module
  have hs := f.smooth s.x (server g c O R s t ω)
  rw [he, inner_neg_right, inner_smul_right, norm_neg, norm_smul,
    Real.norm_eq_abs, abs_of_nonneg (c.η_nonneg t), mul_pow] at hs
  have hD := (sq_le_sq₀ (norm_nonneg D) c.B_nonneg).mpr (c.direction_bound t H)
  have hquad := mul_le_mul_of_nonneg_left hD
    (mul_nonneg (div_nonneg f.Lf_nonneg (by norm_num : (0:ℝ) ≤ 2)) (sq_nonneg (c.η t)))
  have horacle := radius_direction_gain (g.mean s.x) H D f.gain_pos.le
    (f.direction_radius t H) (f.direction_pairing t H)
  have hdual := mul_le_mul_of_nonneg_left (f.dual (g.mean s.x-H))
    (add_nonneg f.gain_pos.le f.r_nonneg)
  rw [norm_sub_rev] at hdual
  have hpair : f.gain*nuclearNorm (g.mean s.x)-(f.gain+f.r)*f.ρ*‖H-g.mean s.x‖-
      f.δ t ≤ inner ℝ (g.mean s.x) D := by nlinarith only [horacle, hdual]
  have hp := mul_le_mul_of_nonneg_left hpair (c.η_nonneg t)
  change f.f (server g c O R s t ω) ≤ _
  nlinarith only [hs, hquad, hp]

def score (x₀ : E m n) (t : ℕ) : ℝ :=
  ∫ ω, nuclearNorm (g.mean (run g c O R x₀ t ω).x) ∂μ
def objectiveValue (x₀ : E m n) (t : ℕ) : ℝ :=
  ∫ ω, f.f (run g c O R x₀ t ω).x ∂μ

theorem objective_drift (x₀ : E m n) (t : ℕ) :
    objectiveValue g c O R f x₀ (t+1) ≤ objectiveValue g c O R f x₀ t-
      c.η t*(f.gain*score g c O R x₀ t)+
      c.η t*((f.gain+f.r)*f.ρ*estimatorError g c O R x₀ t+f.δ t)+
        f.Lf*c.B^2/2*(c.η t)^2 := by
  have hF := objective_integrable g c O R f x₀ t
  have hS := stationarity_integrable g c O R f x₀ t
  have hE := estimatorError_integrable g c O R x₀ t
  have hi := integral_mono (objective_integrable g c O R f x₀ (t+1))
    (((hF.sub ((hS.const_mul f.gain).const_mul (c.η t))).add
      (((hE.const_mul ((f.gain+f.r)*f.ρ)).add (integrable_const (f.δ t))).const_mul (c.η t))).add
        (integrable_const (f.Lf*c.B^2/2*(c.η t)^2)))
    (objective_step g c O R f x₀ t)
  let X := fun ω => f.f (run g c O R x₀ t ω).x
  let Y := fun ω => nuclearNorm (g.mean (run g c O R x₀ t ω).x)
  let Z := fun ω => ‖estimate g c O R (run g c O R x₀ t ω) t ω-
    g.mean (run g c O R x₀ t ω).x‖
  have he : (∫ ω, X ω-c.η t*(f.gain*Y ω)+
      c.η t*((f.gain+f.r)*f.ρ*Z ω+f.δ t)+f.Lf*c.B^2/2*(c.η t)^2 ∂μ) =
      objectiveValue g c O R f x₀ t-c.η t*(f.gain*score g c O R x₀ t)+
        c.η t*((f.gain+f.r)*f.ρ*estimatorError g c O R x₀ t+f.δ t)+
          f.Lf*c.B^2/2*(c.η t)^2 := by
    calc
      _ = ∫ ω, (c.η t*f.δ t+f.Lf*c.B^2/2*(c.η t)^2)+
          1*X ω+(-c.η t*f.gain)*Y ω+(c.η t*((f.gain+f.r)*f.ρ))*Z ω ∂μ := by
        apply integral_congr_ae
        filter_upwards [] with ω
        ring
      _ = _ := by
        rw [expectation_affine_three _ _ _ _ X Y Z hF hS hE]
        dsimp [X, Y, Z, objectiveValue, score, estimatorError]
        ring
  change objectiveValue g c O R f x₀ (t+1) ≤ (∫ ω, X ω-c.η t*(f.gain*Y ω)+
      c.η t*((f.gain+f.r)*f.ρ*Z ω+f.δ t)+f.Lf*c.B^2/2*(c.η t)^2 ∂μ) at hi
  rw [he] at hi
  exact hi

/-- The complete closed stochastic guarantee, for either θ=0 (EMA) or θ=β
(implemented Nesterov), both downlinks, arbitrary nonnegative schedules,
bounded communication roundoff, and an approximate numerical matrix oracle. -/
theorem stationarity (x₀ : E m n) (T : ℕ)
    (hmass : 0 < ∑ t ∈ Finset.range T, c.η t) :
    (∑ t ∈ Finset.range T, c.η t*score g c O R x₀ t)/(∑ t ∈ Finset.range T, c.η t) ≤
      (f.f x₀-f.floor+(f.gain+f.r)*f.ρ*trackingBudget g c O R x₀ T+
        (∑ t ∈ Finset.range T, c.η t*f.δ t)+
        f.Lf*c.B^2/2*(∑ t ∈ Finset.range T, (c.η t)^2))/
      (f.gain*(∑ t ∈ Finset.range T, c.η t)) := by
  have hlower : f.floor ≤ objectiveValue g c O R f x₀ T := by
    have h := integral_mono (integrable_const f.floor)
      (objective_integrable g c O R f x₀ T) (fun ω => f.lower _)
    simpa [objectiveValue] using h
  have h := fixed_momentum_stationarity (objectiveValue g c O R f x₀) c.η
    (fun t => f.gain*score g c O R x₀ t) (estimatorError g c O R x₀)
    (uplinkError g c O R x₀) f.δ c.β_pos c.β_one c.θ_nonneg c.θ_one
    c.qU_nonneg c.qU_one (norm_nonneg _) (Distributed.average_nonneg (fun _ => norm_nonneg _))
    c.peak_nonneg c.η_nonneg c.η_peak (compressionCap_nonneg g c O R)
    (compressionCap_balance g c O R)
    (by simp [uplinkError, run, previousTarget, LayerState.start, LayerState.G, average])
    (uplinkError_step g c O R x₀) (estimatorError_step g c O R x₀)
    (mul_nonneg (add_nonneg f.gain_pos.le f.r_nonneg) f.ρ_nonneg) T hmass hlower
    (fun t _ => objective_drift g c O R f x₀ t)
  have he : (∑ t ∈ Finset.range T, c.η t*(f.gain*score g c O R x₀ t)) =
      f.gain*(∑ t ∈ Finset.range T, c.η t*score g c O R x₀ t) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro t _; ring
  rw [he] at h
  have hzero : objectiveValue g c O R f x₀ 0 = f.f x₀ := by
    simp [objectiveValue, run, LayerState.start, LayerState.x]
  rw [hzero] at h
  change f.gain*(∑ t ∈ Finset.range T, c.η t*score g c O R x₀ t)/_ ≤
    (f.f x₀-f.floor+(f.gain+f.r)*f.ρ*trackingBudget g c O R x₀ T+
      (∑ t ∈ Finset.range T, c.η t*f.δ t)+f.Lf*c.B^2/2*(∑ t ∈ Finset.range T, (c.η t)^2))/_ at h
  apply (le_div_iff₀ (mul_pos f.gain_pos hmass)).mpr
  have hh := (div_le_div_iff_of_pos_right hmass).mp h
  have hcancel : ((∑ t ∈ Finset.range T, c.η t*score g c O R x₀ t)/
      (∑ t ∈ Finset.range T, c.η t))*(f.gain*(∑ t ∈ Finset.range T, c.η t)) =
      f.gain*(∑ t ∈ Finset.range T, c.η t*score g c O R x₀ t) := by
    field_simp [ne_of_gt hmass]
  rw [hcancel]
  exact hh

include f in
theorem expected_score (x₀ : E m n) (T : ℕ) :
    (∫ ω, (∑ t ∈ Finset.range T,
      c.η t*nuclearNorm (g.mean (run g c O R x₀ t ω).x))/
        (∑ t ∈ Finset.range T, c.η t) ∂μ) =
      (∑ t ∈ Finset.range T, c.η t*score g c O R x₀ t)/
        (∑ t ∈ Finset.range T, c.η t) := by
  rw [integral_div, integral_finsetSum _ (fun t _ =>
    (stationarity_integrable g c O R f x₀ t).const_mul (c.η t))]
  simp only [integral_const_mul, score]

/-- Expectation of the score along one complete random trajectory, with all
worker memories generated by `run`, rather than supplied as hypotheses. -/
theorem expected_stationarity (x₀ : E m n) (T : ℕ)
    (hmass : 0 < ∑ t ∈ Finset.range T, c.η t) :
    (∫ ω, (∑ t ∈ Finset.range T,
      c.η t*nuclearNorm (g.mean (run g c O R x₀ t ω).x))/
        (∑ t ∈ Finset.range T, c.η t) ∂μ) ≤
      (f.f x₀-f.floor+(f.gain+f.r)*f.ρ*trackingBudget g c O R x₀ T+
        (∑ t ∈ Finset.range T, c.η t*f.δ t)+
        f.Lf*c.B^2/2*(∑ t ∈ Finset.range T, (c.η t)^2))/
      (f.gain*(∑ t ∈ Finset.range T, c.η t)) := by
  rw [expected_score g c O R f x₀ T]
  exact stationarity g c O R f x₀ T hmass

end EF21Muon.Experiments.RandomizedTrajectory
