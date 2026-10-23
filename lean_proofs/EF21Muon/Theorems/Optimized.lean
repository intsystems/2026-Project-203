import EF21Muon.LongTails.Profiles
import EF21Muon.LongTails.Protocols

noncomputable section

namespace EF21Muon.LongTails

open MeasureTheory OracleTemplate
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

theorem ema_balanced_accuracy (P : Problem N m n μ c) (A : Scheme N m n c)
    {g ε : ℝ} (hg : 0 < g) (hε : 0 < ε) (x : Model m n)
    (hβ : ∀ i, (c i).β = (emaProfile P hg x).balancedBeta ε)
    (hγ : ∀ i, (c i).γ = g * (c i).β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Distributed.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1) :
    P.stationarity A (g * (emaProfile P hg x).balancedBeta ε) (Distributed.start x)
      ((emaProfile P hg x).balancedBudget ε) ≤ ε ^ 2 :=
  (ema_polynomial_bound P A hg ((emaProfile P hg x).balancedBeta_pos ε) hβ
    (fun i => by rw [hγ i, hβ i]) hsmall x _ ((emaProfile P hg x).balancedBudget_pos ε)).trans
    ((emaProfile P hg x).bound_at_balancedBudget hε)

theorem nesterov_balanced_accuracy (P : Problem N m n μ c) (A : Scheme N m n c)
    {g ε : ℝ} (hg : 0 < g) (hε : 0 < ε) (x : Model m n)
    (hβ : ∀ i, (c i).β = (nesterovProfile P hg x).balancedBeta ε)
    (hγ : ∀ i, (c i).γ = g * (c i).β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Nesterov.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1) :
    Nesterov.smoothStationarity P A (g * (nesterovProfile P hg x).balancedBeta ε)
      (Nesterov.initial x) ((nesterovProfile P hg x).balancedBudget ε) ≤ ε ^ 2 :=
  (nesterov_polynomial_bound P A hg ((nesterovProfile P hg x).balancedBeta_pos ε) hβ
    (fun i => by rw [hγ i, hβ i]) hsmall x _ ((nesterovProfile P hg x).balancedBudget_pos ε)).trans
    ((nesterovProfile P hg x).bound_at_balancedBudget hε)

theorem nesterov_paper_balanced_accuracy (P : Problem N m n μ c) (A : Scheme N m n c)
    (hCP : ∀ i, A.CP i 0 = 0)
    {g ε : ℝ} (hg : 0 < g) (hε : 0 < ε) (x : Model m n)
    (hβ : ∀ i, (c i).β = (nesterovProfile P hg x).doubleNoise.balancedBeta ε)
    (hγ : ∀ i, (c i).γ = g * (c i).β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Nesterov.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1) :
    Nesterov.paperSmoothAverage P A
      (g * (nesterovProfile P hg x).doubleNoise.balancedBeta ε) x
      ((nesterovProfile P hg x).doubleNoise.balancedBudget ε) ≤ ε ^ 2 :=
  (nesterov_paper_polynomial_bound P A hCP hg ((nesterovProfile P hg x).doubleNoise.balancedBeta_pos ε)
    hβ (fun i => by rw [hγ i, hβ i]) hsmall x _
    ((nesterovProfile P hg x).doubleNoise.balancedBudget_pos ε)).trans
    ((nesterovProfile P hg x).doubleNoise.bound_at_balancedBudget hε)

end EF21Muon.LongTails

namespace EF21Muon.Theorems.Optimized

open MeasureTheory OracleTemplate
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed EF21Muon.LongTails

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

theorem ema (P : Problem N m n μ c) (A : Scheme N m n c)
    {g ε : ℝ} (hg : 0 < g) (hε : 0 < ε) (x : Model m n)
    (ha : 0 < (emaProfile P hg x).a)
    (hβ : ∀ i, (c i).β = (emaProfile P hg x).optimalBeta ((emaProfile P hg x).optimalBudget ε))
    (hγ : ∀ i, (c i).γ = g * (c i).β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Distributed.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1) :
    let q := emaProfile P hg x
    let T := q.optimalBudget ε
    (∫ ξ, trajectoryScore P.grad
      (execute (sampledOracle P.reply) (emaProtocol μ A T x) (fun _ => T) ξ)
      ∂pathMeasure μ T) ≤ ε ^ 2 ∧
    (∀ ξ : Samples Ω T,
      (executeWithTrace (sampledOracle P.reply) (emaProtocol μ A T x)
        (fun _ => T) ξ).2.length = T ∧
      N * (executeWithTrace (sampledOracle P.reply) (emaProtocol μ A T x)
        (fun _ => T) ξ).2.length = N*T) ∧
    T ≤ q.collapsedBudget ε ∧ T ≤ q.balancedBudget ε ∧
    ∀ (t : ℕ), 0 < t → ∀ (β : ℝ), 0 < β → β ≤ 1 → q.bound t β ≤ ε^2 → T ≤ t := by
  dsimp only
  let q := emaProfile P hg x
  have hs := q.optimalBudget_spec ha hε
  have hv := q.optimalBeta_valid hs.1 ha
  have hsteps i : (c i).γ = g * q.optimalBeta (q.optimalBudget ε) := by rw [hγ i, hβ i]
  refine ⟨?_, ?_, q.optimalBudget_le_collapsed ha hε, q.optimalBudget_le_balanced ha hε, ?_⟩
  · exact (ema_score_eq P A (mul_pos hg hv.1) hsteps _ x).trans_le
      ((ema_polynomial_bound P A hg hv.1 hβ hsteps hsmall x _ hs.1).trans hs.2)
  · intro ξ
    exact ⟨execute_trace_length (sampledOracle P.reply) (emaProtocol μ A _ x) _ ξ (fun _ => ()),
      worker_query_count (sampledOracle P.reply) (emaProtocol μ A _ x) N _ ξ (fun _ => ())⟩
  · intro t ht β hb hb1 hbound
    exact q.optimalBudget_minimal ha hε ht hb hb1 hbound

theorem nesterov (P : Problem N m n μ c) (A : Scheme N m n c)
    {g ε : ℝ} (hg : 0 < g) (hε : 0 < ε) (x : Model m n)
    (ha : 0 < (nesterovProfile P hg x).a)
    (hβ : ∀ i, (c i).β = (nesterovProfile P hg x).optimalBeta ((nesterovProfile P hg x).optimalBudget ε))
    (hγ : ∀ i, (c i).γ = g * (c i).β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Nesterov.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1) :
    let q := nesterovProfile P hg x
    let T := q.optimalBudget ε
    (∫ ξ, trajectoryScore P.grad
      (execute (sampledOracle P.reply) (nesterovProtocol μ A T x) (fun _ => T) ξ)
      ∂pathMeasure μ T) ≤ ε ^ 2 ∧
    (∀ ξ : Samples Ω T,
      (executeWithTrace (sampledOracle P.reply) (nesterovProtocol μ A T x)
        (fun _ => T) ξ).2.length = T ∧
      N * (executeWithTrace (sampledOracle P.reply) (nesterovProtocol μ A T x)
        (fun _ => T) ξ).2.length = N*T) ∧
    T ≤ q.collapsedBudget ε ∧ T ≤ q.balancedBudget ε ∧
    ∀ (t : ℕ), 0 < t → ∀ (β : ℝ), 0 < β → β ≤ 1 → q.bound t β ≤ ε^2 → T ≤ t := by
  dsimp only
  let q := nesterovProfile P hg x
  have hs := q.optimalBudget_spec ha hε
  have hv := q.optimalBeta_valid hs.1 ha
  have hsteps i : (c i).γ = g * q.optimalBeta (q.optimalBudget ε) := by rw [hγ i, hβ i]
  refine ⟨?_, ?_, q.optimalBudget_le_collapsed ha hε, q.optimalBudget_le_balanced ha hε, ?_⟩
  · exact (nesterov_score_eq P A (mul_pos hg hv.1) hsteps _ x).trans_le
      ((nesterov_polynomial_bound P A hg hv.1 hβ hsteps hsmall x _ hs.1).trans hs.2)
  · intro ξ
    exact ⟨execute_trace_length (sampledOracle P.reply) (nesterovProtocol μ A _ x) _ ξ (fun _ => ()),
      worker_query_count (sampledOracle P.reply) (nesterovProtocol μ A _ x) N _ ξ (fun _ => ())⟩
  · intro t ht β hb hb1 hbound
    exact q.optimalBudget_minimal ha hε ht hb hb1 hbound

theorem nesterov_paper (P : Problem N m n μ c) (A : Scheme N m n c)
    (hCP : ∀ i, A.CP i 0 = 0)
    {g ε : ℝ} (hg : 0 < g) (hε : 0 < ε) (x : Model m n)
    (ha : 0 < (nesterovProfile P hg x).a)
    (hβ : ∀ i, (c i).β = (nesterovProfile P hg x).doubleNoise.optimalBeta
      ((nesterovProfile P hg x).doubleNoise.optimalBudget ε))
    (hγ : ∀ i, (c i).γ = g * (c i).β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Nesterov.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1) :
    let q := (nesterovProfile P hg x).doubleNoise
    let T := q.optimalBudget ε
    Nesterov.paperSmoothAverage P A (g * q.optimalBeta T) x T ≤ ε ^ 2 ∧
    T ≤ q.collapsedBudget ε ∧ T ≤ q.balancedBudget ε := by
  dsimp only
  let q := (nesterovProfile P hg x).doubleNoise
  have hs := q.optimalBudget_spec ha hε
  have hv := q.optimalBeta_valid hs.1 ha
  refine ⟨?_, q.optimalBudget_le_collapsed ha hε, q.optimalBudget_le_balanced ha hε⟩
  exact (nesterov_paper_polynomial_bound P A hCP hg hv.1 hβ
    (fun i => by rw [hγ i, hβ i]) hsmall x _ hs.1).trans hs.2

end EF21Muon.Theorems.Optimized
