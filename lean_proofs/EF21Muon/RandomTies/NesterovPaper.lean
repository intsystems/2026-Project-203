import EF21Muon.RandomTies.Paper
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

def nestServer (A : Scheme N m n c) (s : Nesterov.States N m n) : Nesterov.States N m n :=
  Nesterov.sharpServer A s

def nestWorker (reply : Model m n → Ω → Fin N → Model m n)
    (s : Nesterov.States N m n) (z : Coins N m n × Ω) : Nesterov.States N m n :=
  fun i =>
    let a := fun j => reply (fun k => (s k).1.w) z.2 j i
    let M := fun j => (1-(c i).β) • (s i).1.M j+(c i).β • a j
    let V := fun j => (2-(c i).β) • M j-(1-(c i).β) • (s i).1.M j
    (((s i).1.x,(s i).1.w,M,fun j => (s i).1.G j+
      compressor (upBits z.1 i j) (V j-(s i).1.G j)),(s i).1.M)

theorem nestWorker_blend (reply : Model m n → Ω → Fin N → Model m n)
    (s : Nesterov.States N m n) (z : Coins N m n × Ω) (i : Fin p) (j : Fin N) :
    Nesterov.target (c i) (nestWorker (c := c) reply s z i) j =
      (1-(c i).β) • (nestWorker (c := c) reply s z i).1.M j+
        (c i).β • reply (fun k => (s k).1.w) z.2 j i := by
  dsimp [nestWorker, Nesterov.target, LayerState.M]
  module

def nestPaperNext (A : Coins N m n → Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n) (s : Nesterov.States N m n)
    (z : Coins N m n × Ω) : Nesterov.States N m n := nestServer (A z.1) (nestWorker (c := c) reply s z)

def nestDelayedNext (A : Coins N m n → Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n)
    (s : Nesterov.States N m n × Coins N m n) (z : Coins N m n × Ω) : Nesterov.States N m n × Coins N m n :=
  (nestNext (fun _ => A s.2) reply s.1 z,z.1)

def nestProject (A : Coins N m n → Scheme N m n c) (s : Nesterov.States N m n × Coins N m n) :=
  nestServer (A s.2) s.1

theorem nestPaperNext_project (A : Coins N m n → Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n)
    (s : Nesterov.States N m n × Coins N m n) (z : Coins N m n × Ω) :
    nestPaperNext A reply (nestProject A s) z = nestProject A (nestDelayedNext A reply s z) := rfl

theorem nestProject_start (A : Coins N m n → Scheme N m n c)
    (hCP : ∀ b i, (A b).CP i 0 = 0) (x : Model m n) (b : Coins N m n) :
    nestProject A (Nesterov.initial x,b) = Nesterov.initial x :=
  Nesterov.sharpServer_start (A b) (hCP b) x

theorem nest_delayed_finite_time (P : Problem N m n μ c)
    (A : Coins N m n → Scheme N m n c) (hCU : ∀ b i, (A b).CU i = signCompressor)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (hsmall : ∀ b i, 2*(c i).L*(c i).γ+
      4*Nesterov.zeta (c i) (P.ell2 i) ((A b).τ i)*(c i).γ^2 ≤ 1)
    (T : ℕ) (s : Nesterov.States N m n × Coins N m n) :
    Integrable (fun ξ => ((run (nestDelayedNext A P.reply) T s ξ).map
      (fun s => Nesterov.smoothCost P s.1)).sum) (pathMeasure ((coinLaw N m n).prod μ) T) ∧
    (∫ ξ, ((run (nestDelayedNext A P.reply) T s ξ).map (fun s => Nesterov.smoothCost P s.1)).sum
      ∂pathMeasure ((coinLaw N m n).prod μ) T) ≤ Nesterov.Averaged.smoothPotential P s.1+(T:ℝ)*Nesterov.Averaged.smoothNoise P := by
  have hnoise := Nesterov.Averaged.smoothNoise_nonneg P
  simpa only [one_pow, one_mul, Function.comp_def] using finite_choice_convergence (coinLaw N m n) μ
    (f := nestDelayedNext A P.reply)
    (base := fun b s ω => (Nesterov.next (A s.2) P.reply s.1 ω,b))
    (a := 1) (by norm_num) hnoise
    (measurable_delayed_choice (fun old fresh =>
      ((measurable_nestNext (fun _ => A old) P.measurable_reply).comp
        (measurable_fst.prodMk (measurable_const.prodMk measurable_snd))).prodMk measurable_const))
    ((Nesterov.smoothCost_measurable P).comp measurable_fst) ((Nesterov.Averaged.smoothPotential_measurable P).comp measurable_fst)
    (fun s => Nesterov.smoothCost_nonneg P s.1) (fun s => Nesterov.Averaged.smoothPotential_nonneg P s.1)
    (fun s b ω => nest_potential_eq P (fun _ => A s.2) (fun _ => hCU s.2) hm hn s.1 b ω)
    (fun _ s => Nesterov.Averaged.smoothPotential_next_integrable P (A s.2) s.1)
    (fun _ s => by simpa only [one_mul, Function.comp_def] using Nesterov.Averaged.smoothPotential_descent P (A s.2) (hsmall s.2) s.1) T s

def nestPaperSum (P : Problem N m n μ c) (A : Coins N m n → Scheme N m n c)
    (x : Model m n) (T : ℕ) (ξ : Samples (Coins N m n × Ω) (T+1)) : ℝ :=
  ∑ t : Fin T, Nesterov.smoothCost P (pathState (nestPaperNext A P.reply) (T+1) (Nesterov.initial x) ξ t.castSucc)

theorem nestPaperSum_eq (P : Problem N m n μ c) (A : Coins N m n → Scheme N m n c)
    (hCP : ∀ b i, (A b).CP i 0 = 0) (x : Model m n) (T : ℕ) (b : Coins N m n)
    (ξ : Samples (Coins N m n × Ω) (T+1)) :
    nestPaperSum P A x T ξ =
      ((run (nestDelayedNext A P.reply) (T+1) (Nesterov.initial x,b) ξ).map
        (fun s => Nesterov.smoothCost P s.1)).sum-Nesterov.smoothCost P (Nesterov.initial x) := by
  have hcost (s : Nesterov.States N m n × Coins N m n) (z : Coins N m n × Ω) :
      Nesterov.smoothCost P (nestDelayedNext A P.reply s z).1 = Nesterov.smoothCost P (nestProject A s) := rfl
  have hpath (t : Fin (T+1)) : pathState (nestPaperNext A P.reply) (T+1) (Nesterov.initial x) ξ t =
      nestProject A (pathState (nestDelayedNext A P.reply) (T+1) (Nesterov.initial x,b) ξ t) := by
    conv_lhs => rw [← nestProject_start A hCP x b]
    exact pathState_map _ _ _ (nestPaperNext_project A P.reply) _ _ _ _
  have he (t : Fin T) :
      Nesterov.smoothCost P (pathState (nestDelayedNext A P.reply) (T+1) (Nesterov.initial x,b) ξ t.succ).1 =
        Nesterov.smoothCost P (pathState (nestPaperNext A P.reply) (T+1) (Nesterov.initial x) ξ t.castSucc) := by
    rw [hpath]
    exact pathState_adjacent_cost (nestDelayedNext A P.reply) (fun s => Nesterov.smoothCost P s.1)
      (fun s => Nesterov.smoothCost P (nestProject A s)) hcost T (Nesterov.initial x,b) ξ t
  rw [sum_run_eq_states, Fin.sum_univ_succ]
  rw [show pathState (nestDelayedNext A P.reply) (T+1) (Nesterov.initial x,b) ξ 0 = (Nesterov.initial x,b) from rfl]
  simp_rw [he]
  unfold nestPaperSum
  ring

theorem nest_paper_average (P : Problem N m n μ c)
    (A : Coins N m n → Scheme N m n c) (hCU : ∀ b i, (A b).CU i = signCompressor)
    (hCP : ∀ b i, (A b).CP i 0 = 0)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (hsmall : ∀ b i, 2*(c i).L*(c i).γ+
      4*Nesterov.zeta (c i) (P.ell2 i) ((A b).τ i)*(c i).γ^2 ≤ 1)
    {γbar : ℝ} (hγ : 0 < γbar) (x : Model m n) (T : ℕ) (hT : 0 < T) :
    Integrable (nestPaperSum P A x T) (pathMeasure ((coinLaw N m n).prod μ) (T+1)) ∧
      4/(γbar*T)*(∫ ξ, nestPaperSum P A x T ξ ∂pathMeasure ((coinLaw N m n).prod μ) (T+1)) ≤
        4*Nesterov.Averaged.smoothPotential P (Nesterov.initial x)/(γbar*T)+8*Nesterov.Averaged.smoothNoise P/γbar := by
  let b : Coins N m n := fun _ => true
  have h := nest_delayed_finite_time P A hCU hm hn hsmall (T+1) (Nesterov.initial x,b)
  have he := nestPaperSum_eq P A hCP x T b
  have hi : Integrable (nestPaperSum P A x T) (pathMeasure ((coinLaw N m n).prod μ) (T+1)) := by
    have hi0 := h.1.sub (integrable_const (Nesterov.smoothCost P (Nesterov.initial x)))
    change Integrable (fun ξ => ((run (nestDelayedNext A P.reply) (T+1) (Nesterov.initial x,b) ξ).map
      (fun s => Nesterov.smoothCost P s.1)).sum-Nesterov.smoothCost P (Nesterov.initial x)) _ at hi0
    simpa only [← he] using hi0
  refine ⟨hi, ?_⟩
  have hint : (∫ ξ, nestPaperSum P A x T ξ ∂pathMeasure ((coinLaw N m n).prod μ) (T+1)) =
      (∫ ξ, ((run (nestDelayedNext A P.reply) (T+1) (Nesterov.initial x,b) ξ).map
        (fun s => Nesterov.smoothCost P s.1)).sum ∂pathMeasure ((coinLaw N m n).prod μ) (T+1))-Nesterov.smoothCost P (Nesterov.initial x) := by
    simp_rw [he]
    rw [integral_sub h.1 (integrable_const _), integral_const, probReal_univ, one_smul]
  have hTr : (0:ℝ) < T := by exact_mod_cast hT
  have hc := Nesterov.smoothCost_nonneg P (Nesterov.initial x)
  have hn0 := Nesterov.Averaged.smoothNoise_nonneg P
  have hT1 : (T:ℝ)+1 ≤ 2*T := by exact_mod_cast (show T+1 ≤ 2*T by omega)
  have hn' := mul_le_mul_of_nonneg_right hT1 hn0
  have hb : (∫ ξ, nestPaperSum P A x T ξ ∂pathMeasure ((coinLaw N m n).prod μ) (T+1)) ≤
      Nesterov.Averaged.smoothPotential P (Nesterov.initial x)+2*T*Nesterov.Averaged.smoothNoise P := by
    rw [hint]
    simp only [Nat.cast_add, Nat.cast_one] at h
    linarith only [h.2, hn', hc]
  have hf := mul_le_mul_of_nonneg_left hb (by positivity : 0 ≤ 4/(γbar*T))
  have heq : 4/(γbar*T)*(Nesterov.Averaged.smoothPotential P (Nesterov.initial x)+2*T*Nesterov.Averaged.smoothNoise P) =
      4*Nesterov.Averaged.smoothPotential P (Nesterov.initial x)/(γbar*T)+8*Nesterov.Averaged.smoothNoise P/γbar := by field_simp; ring
  exact hf.trans_eq heq

end EF21Muon.RandomTies
