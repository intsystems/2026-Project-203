import EF21Muon.RandomTies.Generalized
import EF21Muon.Distributed.Correspondence
import EF21Muon.Probability.PathStates

/-! Paper round order with random ties. The auxiliary rotated state retains
the downlink seed for its next server phase. Its projection therefore follows
the literal query/uplink/server/downlink order, on the same sample path.
The extra seed is ignored by the Lyapunov function and stationarity cost. -/

noncomputable section
namespace EF21Muon.RandomTies
set_option maxHeartbeats 800000
set_option backward.isDefEq.respectTransparency false
open MeasureTheory hiding average
open LongTails.Algorithms.MatrixPolar
open EF21Muon.Distributed EF21Muon.Stochastic EF21Muon.Probability
variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def emaServer (A : Scheme N m n c) (s : States N m n) : States N m n :=
  fun i => (nextX (c i) (s i),nextW (c i) (A.CP i) (s i),(s i).M,(s i).G)

def emaWorker (reply : Model m n → Ω → Fin N → Model m n)
    (s : States N m n) (z : Coins N m n × Ω) : States N m n :=
  fun i =>
    let a := fun j => reply (fun k => (s k).w) z.2 j i
    let M := fun j => (1-(c i).β) • (s i).M j+(c i).β • a j
    ((s i).x,(s i).w,M,fun j => (s i).G j+compressor (upBits z.1 i j) (M j-(s i).G j))

def emaPaperNext (A : Coins N m n → Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n) (s : States N m n)
    (z : Coins N m n × Ω) : States N m n := emaServer (A z.1) (emaWorker (c := c) reply s z)

def emaDelayedNext (A : Coins N m n → Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n)
    (s : States N m n × Coins N m n) (z : Coins N m n × Ω) : States N m n × Coins N m n :=
  (emaNext (fun _ => A s.2) reply s.1 z,z.1)

def emaProject (A : Coins N m n → Scheme N m n c) (s : States N m n × Coins N m n) :=
  emaServer (A s.2) s.1

theorem emaPaperNext_project (A : Coins N m n → Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n)
    (s : States N m n × Coins N m n) (z : Coins N m n × Ω) :
    emaPaperNext A reply (emaProject A s) z = emaProject A (emaDelayedNext A reply s z) := rfl

theorem emaProject_start (A : Coins N m n → Scheme N m n c)
    (hCP : ∀ b i, (A b).CP i 0 = 0) (x : Model m n) (b : Coins N m n) :
    emaProject A (start x,b) = start x := by
  funext i
  simp [emaProject, emaServer, start, LayerState.start, nextX, nextW,
    LayerState.server, LayerState.x, LayerState.w, LayerState.M, LayerState.G,
    serverPoint, queryPoint, average, polar_zero, hCP]

theorem ema_delayed_finite_time (P : Problem N m n μ c)
    (A : Coins N m n → Scheme N m n c) (hCU : ∀ b i, (A b).CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (hsmall : ∀ b i, 2*(c i).L*(c i).γ+
      4*zeta (c i) (P.ell2 i) ((A b).τ i)*(c i).γ^2 ≤ 1)
    (T : ℕ) (s : States N m n × Coins N m n) :
    Integrable (fun ξ => ((run (emaDelayedNext A P.reply) T s ξ).map
      (fun s => P.cost s.1)).sum) (pathMeasure ((coinLaw N m n).prod μ) T) ∧
    (∫ ξ, ((run (emaDelayedNext A P.reply) T s ξ).map (fun s => P.cost s.1)).sum
      ∂pathMeasure ((coinLaw N m n).prod μ) T) ≤ P.potential s.1+(T:ℝ)*P.noiseTotal := by
  have hnoise : 0 ≤ P.noiseTotal := by
    apply Finset.sum_nonneg
    intro i _
    have hγ := (c i).γ_pos
    have hβ := (c i).β_pos
    have hα := (c i).αU_pos
    have hα1 := sub_nonneg.mpr (c i).αU_le_one
    unfold noise
    positivity
  simpa only [one_pow, one_mul, Function.comp_def] using finite_choice_convergence (coinLaw N m n) μ
    (f := emaDelayedNext A P.reply)
    (base := fun b s ω => (Distributed.next (A s.2) P.reply s.1 ω,b))
    (a := 1) (by norm_num) hnoise
    (measurable_delayed_choice (fun old fresh =>
      ((measurable_emaNext (fun _ => A old) P.measurable_reply).comp
        (measurable_fst.prodMk (measurable_const.prodMk measurable_snd))).prodMk measurable_const))
    (P.cost_measurable.comp measurable_fst) (P.potential_measurable.comp measurable_fst)
    (fun s => P.cost_nonneg s.1) (fun s => P.potential_nonneg s.1)
    (fun s b ω => ema_potential_eq P (fun _ => A s.2) (fun _ => hCU s.2) hm hn s.1 b ω)
    (fun _ s => P.potential_next_integrable (A s.2) s.1)
    (fun _ s => by simpa only [one_mul, Function.comp_def] using P.potential_descent (A s.2) (hsmall s.2) s.1) T s

def emaPaperSum (P : Problem N m n μ c) (A : Coins N m n → Scheme N m n c)
    (x : Model m n) (T : ℕ) (ξ : Samples (Coins N m n × Ω) (T+1)) : ℝ :=
  ∑ t : Fin T, P.cost (pathState (emaPaperNext A P.reply) (T+1) (start x) ξ t.castSucc)

theorem emaPaperSum_eq (P : Problem N m n μ c) (A : Coins N m n → Scheme N m n c)
    (hCP : ∀ b i, (A b).CP i 0 = 0) (x : Model m n) (T : ℕ) (b : Coins N m n)
    (ξ : Samples (Coins N m n × Ω) (T+1)) :
    emaPaperSum P A x T ξ =
      ((run (emaDelayedNext A P.reply) (T+1) (start x,b) ξ).map
        (fun s => P.cost s.1)).sum-P.cost (start x) := by
  have hcost (s : States N m n × Coins N m n) (z : Coins N m n × Ω) :
      P.cost (emaDelayedNext A P.reply s z).1 = P.cost (emaProject A s) := rfl
  have hpath (t : Fin (T+1)) : pathState (emaPaperNext A P.reply) (T+1) (start x) ξ t =
      emaProject A (pathState (emaDelayedNext A P.reply) (T+1) (start x,b) ξ t) := by
    conv_lhs => rw [← emaProject_start A hCP x b]
    exact pathState_map _ _ _ (emaPaperNext_project A P.reply) _ _ _ _
  have he (t : Fin T) :
      P.cost (pathState (emaDelayedNext A P.reply) (T+1) (start x,b) ξ t.succ).1 =
        P.cost (pathState (emaPaperNext A P.reply) (T+1) (start x) ξ t.castSucc) := by
    rw [hpath]
    exact pathState_adjacent_cost (emaDelayedNext A P.reply) (fun s => P.cost s.1)
      (fun s => P.cost (emaProject A s)) hcost T (start x,b) ξ t
  rw [sum_run_eq_states, Fin.sum_univ_succ]
  rw [show pathState (emaDelayedNext A P.reply) (T+1) (start x,b) ξ 0 = (start x,b) from rfl]
  simp_rw [he]
  unfold emaPaperSum
  ring

theorem ema_paper_average (P : Problem N m n μ c)
    (A : Coins N m n → Scheme N m n c) (hCU : ∀ b i, (A b).CU i = signCompressor)
    (hCP : ∀ b i, (A b).CP i 0 = 0)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (hsmall : ∀ b i, 2*(c i).L*(c i).γ+
      4*zeta (c i) (P.ell2 i) ((A b).τ i)*(c i).γ^2 ≤ 1)
    {γbar : ℝ} (hγ : 0 < γbar) (x : Model m n) (T : ℕ) (hT : 0 < T) :
    Integrable (emaPaperSum P A x T) (pathMeasure ((coinLaw N m n).prod μ) (T+1)) ∧
      4/(γbar*T)*(∫ ξ, emaPaperSum P A x T ξ ∂pathMeasure ((coinLaw N m n).prod μ) (T+1)) ≤
        4*P.potential (start x)/(γbar*T)+8*P.noiseTotal/γbar := by
  let b : Coins N m n := fun _ => true
  have h := ema_delayed_finite_time P A hCU hm hn hsmall (T+1) (start x,b)
  have he := emaPaperSum_eq P A hCP x T b
  have hi : Integrable (emaPaperSum P A x T) (pathMeasure ((coinLaw N m n).prod μ) (T+1)) := by
    have hi0 := h.1.sub (integrable_const (P.cost (start x)))
    change Integrable (fun ξ => ((run (emaDelayedNext A P.reply) (T+1) (start x,b) ξ).map
      (fun s => P.cost s.1)).sum-P.cost (start x)) _ at hi0
    simpa only [← he] using hi0
  refine ⟨hi, ?_⟩
  have hint : (∫ ξ, emaPaperSum P A x T ξ ∂pathMeasure ((coinLaw N m n).prod μ) (T+1)) =
      (∫ ξ, ((run (emaDelayedNext A P.reply) (T+1) (start x,b) ξ).map
        (fun s => P.cost s.1)).sum ∂pathMeasure ((coinLaw N m n).prod μ) (T+1))-P.cost (start x) := by
    simp_rw [he]
    rw [integral_sub h.1 (integrable_const _), integral_const, probReal_univ, one_smul]
  have hTr : (0:ℝ) < T := by exact_mod_cast hT
  have hc := P.cost_nonneg (start x)
  have hn0 : 0 ≤ P.noiseTotal := by
    apply Finset.sum_nonneg
    intro i _
    have hγi := (c i).γ_pos
    have hβ := (c i).β_pos
    have hα := (c i).αU_pos
    have hα1 := sub_nonneg.mpr (c i).αU_le_one
    unfold noise
    positivity
  have hT1 : (T:ℝ)+1 ≤ 2*T := by exact_mod_cast (show T+1 ≤ 2*T by omega)
  have hn' := mul_le_mul_of_nonneg_right hT1 hn0
  have hb : (∫ ξ, emaPaperSum P A x T ξ ∂pathMeasure ((coinLaw N m n).prod μ) (T+1)) ≤
      P.potential (start x)+2*T*P.noiseTotal := by
    rw [hint]
    simp only [Nat.cast_add, Nat.cast_one] at h
    linarith only [h.2, hn', hc]
  have hf := mul_le_mul_of_nonneg_left hb (by positivity : 0 ≤ 4/(γbar*T))
  have heq : 4/(γbar*T)*(P.potential (start x)+2*T*P.noiseTotal) =
      4*P.potential (start x)/(γbar*T)+8*P.noiseTotal/γbar := by field_simp; ring
  exact hf.trans_eq heq

end EF21Muon.RandomTies
