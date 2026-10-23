import EF21Muon.Experiments.RandomizedTrajectory
import EF21Muon.Experiments.FreshTrajectory

/-! Causality and conditional-noise convergence with randomized communication. -/
noncomputable section
namespace EF21Muon.Experiments.RandomizedTrajectory
open scoped BigOperators
open MeasureTheory hiding average
open EF21Muon.Probability EF21Muon.Distributed
open EF21Muon.Experiments.Trajectory (E Innovations Gradients Roundoff MartingaleInnovations)
set_option maxHeartbeats 800000
set_option backward.isDefEq.respectTransparency false
variable {N m n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

variable (g : Gradients N m n) (c : Controls N m n Ω)
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
    (hCU : ∀ t j, Measurable[(O.filtration (t+1)).prod
      (inferInstance : MeasurableSpace (E m n))]
      (fun z : Ω × E m n => c.CU t j z.1 z.2))
    (hCP : ∀ t, Measurable[(O.filtration (t+1)).prod
      (inferInstance : MeasurableSpace (E m n))]
      (fun z : Ω × E m n => c.CP t z.1 z.2))
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
      ((hg j).add ((hCU t j).comp (measurable_id.prodMk ((hV j).sub (hg j))))).add (hU t j)
    have hH : Measurable[O.filtration (t+1)] (fun ω => estimate g c O.innovations R (s ω) t ω) :=
      (Finset.measurable_sum Finset.univ (fun j _ => hG j)).const_smul (N : ℝ)⁻¹
    have hX : Measurable[O.filtration (t+1)] (fun ω => server g c O.innovations R (s ω) t ω) :=
      hx.sub (((c.measurable_direction t).comp hH).const_smul (c.η t))
    have hW := (hw.add ((hCP t).comp (measurable_id.prodMk (hX.sub hw)))).add (hP t)
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
    (hCU : ∀ t j, Measurable[(O.filtration (t+1)).prod
      (inferInstance : MeasurableSpace (E m n))]
      (fun z : Ω × E m n => c.CU t j z.1 z.2))
    (hCP : ∀ t, Measurable[(O.filtration (t+1)).prod
      (inferInstance : MeasurableSpace (E m n))]
      (fun z : Ω × E m n => c.CP t z.1 z.2))
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
  exact ⟨run_adapted g c O R hU hP hCU hCP x₀,
    expected_stationarity g c O.innovations R f x₀ T hmass⟩

end EF21Muon.Experiments.RandomizedTrajectory
