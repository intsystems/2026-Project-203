import EF21Muon.Experiments.Trajectory
import Mathlib.MeasureTheory.Function.ConditionalExpectation.PullOut
import Mathlib.Probability.Process.Filtration

/-! Conditional mean-zero innovations instantiate the full trajectory theorem.
Conditional worker cross moments may vanish without conditional independence;
only that weaker moment property is needed for the N-worker gain. -/

noncomputable section
namespace EF21Muon.Experiments.Trajectory
open scoped BigOperators
open MeasureTheory hiding average
open EF21Muon.Probability EF21Muon.Distributed

set_option maxHeartbeats 800000
set_option backward.isDefEq.respectTransparency false

variable {N m n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Noise revealed at the end of each round and centered conditionally on its
past. Conditional worker independence can be weakened to zero cross moments.
The second-moment bound is unconditional; a uniform conditional bound implies it. -/
structure MartingaleInnovations (N m n : ℕ) (μ : Measure Ω) where
  filtration : Filtration ℕ ‹MeasurableSpace Ω›
  σ : ℝ
  σ_nonneg : 0 ≤ σ
  value : ℕ → Fin N → Ω → E m n
  adapted : ∀ t j, StronglyMeasurable[filtration (t+1)] (value t j)
  memLp : ∀ t j, MemLp (value t j) 2 μ
  variance : ∀ t j, (∫ ω, ‖value t j ω‖^2 ∂μ) ≤ σ^2
  centered : ∀ t j, μ[value t j | filtration t] =ᵐ[μ] 0
  worker_cross : ∀ t i j, i ≠ j →
    μ[fun ω => inner ℝ (value t i ω) (value t j ω) | filtration t] =ᵐ[μ] 0

namespace MartingaleInnovations
variable (O : MartingaleInnovations N m n μ)

theorem integral_zero (t : ℕ) (j : Fin N) : (∫ ω, O.value t j ω ∂μ) = 0 := by
  rw [← integral_condExp (O.filtration.le t)]
  simpa only [Pi.zero_apply, MeasureTheory.integral_zero] using integral_congr_ae (O.centered t j)

theorem past_orthogonal {s t : ℕ} (hst : s < t) (i j : Fin N) :
    (∫ ω, inner ℝ (O.value s i ω) (O.value t j ω) ∂μ) = 0 := by
  have hp := (O.adapted s i).mono (O.filtration.mono hst)
  have hc := condExp_bilin_of_stronglyMeasurable_left (innerSL ℝ) hp
    (inner_integrable (O.memLp s i) (O.memLp t j)) ((O.memLp t j).integrable (by norm_num))
  change μ[fun ω => inner ℝ (O.value s i ω) (O.value t j ω) | O.filtration t] =ᵐ[μ]
    (fun ω => inner ℝ (O.value s i ω) (μ[O.value t j | O.filtration t] ω)) at hc
  have hz : μ[fun ω => inner ℝ (O.value s i ω) (O.value t j ω) | O.filtration t] =ᵐ[μ] 0 := by
    filter_upwards [hc, O.centered t j] with ω hcω hzω
    change _ = inner ℝ (O.value s i ω) (μ[O.value t j | O.filtration t] ω) at hcω
    simpa only [hzω, Pi.zero_apply, inner_zero_right] using hcω
  rw [← integral_condExp (O.filtration.le t)]
  simpa only [Pi.zero_apply, MeasureTheory.integral_zero] using integral_congr_ae hz

theorem orthogonal (s t : ℕ) (i j : Fin N) (hne : s ≠ t ∨ i ≠ j) :
    (∫ ω, inner ℝ (O.value s i ω) (O.value t j ω) ∂μ) = 0 := by
  rcases lt_trichotomy s t with hst | rfl | hts
  · exact O.past_orthogonal hst i j
  · have hij : i ≠ j := hne.resolve_left (not_not_intro rfl)
    rw [← integral_condExp (O.filtration.le s)]
    simpa only [Pi.zero_apply, MeasureTheory.integral_zero] using integral_congr_ae (O.worker_cross s i j hij)
  · have h := O.past_orthogonal hts j i
    simpa only [real_inner_comm] using h

/-- The cross-time orthogonality required by the filter proof is derived,
and is not an additional assumption on adaptive stochastic replies. -/
def innovations : Innovations N m n μ where
  σ := O.σ
  σ_nonneg := O.σ_nonneg
  value := O.value
  measurable_value := fun t j => ((O.adapted t j).mono (O.filtration.le (t+1))).measurable
  memLp := O.memLp
  centered := O.integral_zero
  variance := O.variance
  orthogonal := O.orthogonal
end MartingaleInnovations

variable (g : Gradients N m n) (c : Controls m n)
  (O : MartingaleInnovations N m n μ) (R : Roundoff N m n Ω)

/-- The actual stochastic reply is conditionally unbiased at the broadcast
query used by the optimizer, including for Nesterov's current-reply blend. -/
theorem replies_centered (x₀ : E m n) (t : ℕ) (j : Fin N) :
    μ[fun ω => reply g O.innovations (run g c O.innovations R x₀ t ω) t ω j-
      g.grad j (run g c O.innovations R x₀ t ω).w | O.filtration t] =ᵐ[μ] 0 := by
  simpa only [reply, MartingaleInnovations.innovations, add_sub_cancel_left] using O.centered t j

/-- With causal reconstruction errors, every actual model and worker memory
is measurable before the following round's noise is revealed. -/
theorem run_adapted
    (hU : ∀ t j, Measurable[O.filtration (t+1)] (R.uplink t j))
    (hP : ∀ t, Measurable[O.filtration (t+1)] (R.downlink t))
    (x₀ : E m n) (t : ℕ) :
    Measurable[O.filtration t] (run g c O.innovations R x₀ t) := by
  induction t with
  | zero => exact measurable_const
  | succ t ih =>
    let s := run g c O.innovations R x₀ t
    have hs : Measurable[O.filtration (t+1)] s := ih.mono
      (O.filtration.mono (Nat.le_succ t)) le_rfl
    have hx : Measurable[O.filtration (t+1)] (fun ω => (s ω).x) := hs.fst
    have hw : Measurable[O.filtration (t+1)] (fun ω => (s ω).w) := hs.snd.fst
    have hm j : Measurable[O.filtration (t+1)] (fun ω => (s ω).M j) :=
      (measurable_pi_apply j).comp hs.snd.snd.fst
    have hg j : Measurable[O.filtration (t+1)] (fun ω => (s ω).G j) :=
      (measurable_pi_apply j).comp hs.snd.snd.snd
    have ha j : Measurable[O.filtration (t+1)] (fun ω => reply g O.innovations (s ω) t ω j) :=
      ((g.measurable_local j).comp hw).add (O.adapted t j).measurable
    have hM j : Measurable[O.filtration (t+1)] (fun ω => momentum g c O.innovations (s ω) t ω j) :=
      ((hm j).const_smul (1-c.β)).add ((ha j).const_smul c.β)
    have hV j : Measurable[O.filtration (t+1)] (fun ω => target g c O.innovations (s ω) t ω j) :=
      ((hM j).const_smul (1-c.θ)).add ((ha j).const_smul c.θ)
    have hG j : Measurable[O.filtration (t+1)] (fun ω => upload g c O.innovations R (s ω) t ω j) :=
      ((hg j).add (c.measurable_CU.comp ((hV j).sub (hg j)))).add (hU t j)
    have hH : Measurable[O.filtration (t+1)] (fun ω => estimate g c O.innovations R (s ω) t ω) :=
      (Finset.measurable_sum Finset.univ (fun j _ => hG j)).const_smul (N : ℝ)⁻¹
    have hX : Measurable[O.filtration (t+1)] (fun ω => server g c O.innovations R (s ω) t ω) :=
      hx.sub (((c.measurable_direction t).comp hH).const_smul (c.η t))
    have hW := (hw.add (c.measurable_CP.comp (hX.sub hw))).add (hP t)
    have hMs : Measurable[O.filtration (t+1)]
        (fun ω => momentum g c O.innovations (s ω) t ω) := by
      let : MeasurableSpace Ω := O.filtration (t+1)
      exact Measurable.of_eval hM
    have hGs : Measurable[O.filtration (t+1)]
        (fun ω => upload g c O.innovations R (s ω) t ω) := by
      let : MeasurableSpace Ω := O.filtration (t+1)
      exact Measurable.of_eval hG
    exact hX.prodMk (hW.prodMk (hMs.prodMk hGs))

/-- One theorem for the entire adapted N-worker trajectory and its closed
stationarity bound. No tracking or objective-drift recurrence, trajectory
integrability, or stationarity bound is assumed. θ=0 and θ=β select EMA and
implemented Nesterov, respectively. -/
theorem fresh_stationarity (f : Objective g c)
    (hU : ∀ t j, Measurable[O.filtration (t+1)] (R.uplink t j))
    (hP : ∀ t, Measurable[O.filtration (t+1)] (R.downlink t))
    (x₀ : E m n) (T : ℕ) (hmass : 0 < ∑ t ∈ Finset.range T, c.η t) :
    (∀ t, Measurable[O.filtration t] (run g c O.innovations R x₀ t)) ∧
    (∫ ω, (∑ t ∈ Finset.range T,
      c.η t*LongTails.Algorithms.MatrixPolar.nuclearNorm
        (g.mean (run g c O.innovations R x₀ t ω).x))/
        (∑ t ∈ Finset.range T, c.η t) ∂μ) ≤
      (f.f x₀-f.floor+(f.gain+f.r)*f.ρ*trackingBudget g c O.innovations R x₀ T+
        (∑ t ∈ Finset.range T, c.η t*f.δ t)+
        f.Lf*c.B^2/2*(∑ t ∈ Finset.range T, (c.η t)^2))/
      (f.gain*(∑ t ∈ Finset.range T, c.η t)) := by
  exact ⟨run_adapted g c O R hU hP x₀,
    expected_stationarity g c O.innovations R f x₀ T hmass⟩

end EF21Muon.Experiments.Trajectory
