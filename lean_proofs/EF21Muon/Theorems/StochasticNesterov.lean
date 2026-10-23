import EF21Muon.LongTails.Protocols

noncomputable section

namespace EF21Muon.Theorems.StochasticNesterov

open OracleTemplate MeasureTheory
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed EF21Muon.LongTails

theorem result {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}
    (P : Problem N m n μ c) (A : Scheme N m n c)
    {g ε : ℝ} (hg : 0 < g) (hε : 0 < ε)
    (hβ : ∀ i, (c i).β = accuracyBeta (Nesterov.noiseTotalScale P) ε)
    (hγ : ∀ i, (c i).γ = g * (c i).β)
    (hsmall : ∀ i, g * (2 * (c i).L +
      2 * Real.sqrt (Nesterov.rateEnvelope (c i) (P.ell2 i) (A.τ i))) ≤ 1)
    (x : Model m n) :
    let T := accuracyBudget (Nesterov.rateInitial P g x) g (Nesterov.noiseTotalScale P) ε
    (∫ ξ, trajectoryScore P.grad
      (execute (sampledOracle P.reply) (nesterovProtocol μ A T x) (fun _ => T) ξ)
      ∂pathMeasure μ T) ≤ ε ^ 2 ∧
    ∀ ξ : Samples Ω T,
      (executeWithTrace (sampledOracle P.reply) (nesterovProtocol μ A T x)
        (fun _ => T) ξ).2.length = T ∧
      N * (executeWithTrace (sampledOracle P.reply) (nesterovProtocol μ A T x)
        (fun _ => T) ξ).2.length = N * T := by
  dsimp only
  refine ⟨?_, fun ξ => ⟨execute_trace_length (sampledOracle P.reply)
    (nesterovProtocol μ A _ x) _ ξ (fun _ => ()),
    worker_query_count (sampledOracle P.reply) (nesterovProtocol μ A _ x) N _ ξ (fun _ => ())⟩⟩
  rw [nesterov_score_eq P A (mul_pos hg (accuracyBeta_pos (nesterovScale_nonneg P) hε))
    (fun i => by rw [hγ i, hβ i])]
  exact nesterov_accuracy P A hg hε hβ hγ hsmall x

end EF21Muon.Theorems.StochasticNesterov
