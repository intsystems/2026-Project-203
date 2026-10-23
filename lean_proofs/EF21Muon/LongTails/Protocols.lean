import EF21Muon.LongTails.Execution
import EF21Muon.LongTails.Tuning

noncomputable section

namespace EF21Muon.LongTails

open OracleTemplate MeasureTheory
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def emaUpdate (A : Scheme N m n c) (s : Distributed.States N m n)
    (reply : Fin N → Model m n) : Distributed.States N m n :=
  fun i => Distributed.update (c i) (A.CU i) (A.CP i) (s i) (fun j => reply j i)

def nesterovUpdate (A : Scheme N m n c) (s : Nesterov.States N m n)
    (reply : Fin N → Model m n) : Nesterov.States N m n :=
  fun i => Nesterov.update (c i) (A.CU i) (A.CP i) (s i) (fun j => reply j i)

def emaProtocol (μ : Measure Ω) [IsProbabilityMeasure μ] (A : Scheme N m n c)
    (T : ℕ) (x : Model m n) :=
  batchProtocol μ T (Distributed.queries A) (emaUpdate A) Distributed.points (Distributed.start x)

def nesterovProtocol (μ : Measure Ω) [IsProbabilityMeasure μ] (A : Scheme N m n c)
    (T : ℕ) (x : Model m n) :=
  batchProtocol μ T (Nesterov.queries A) (nesterovUpdate A) Nesterov.points (Nesterov.initial x)

def trajectoryScore (grad : Model m n → Model m n) (xs : List (Model m n)) : ℝ :=
  (xs.map (fun x => ∑ i, nuclearNorm (grad x i) ^ 2)).sum / xs.length

theorem execute_ema (P : Problem N m n μ c) (A : Scheme N m n c)
    (T : ℕ) (x : Model m n) (ξ : Samples Ω T) :
    execute (sampledOracle P.reply) (emaProtocol μ A T x) (fun _ => T) ξ =
      (run (Distributed.next A P.reply) T (Distributed.start x) ξ).map Distributed.points :=
  execute_batchProtocol μ _ _ _ _ _ _ _

theorem execute_nesterov (P : Problem N m n μ c) (A : Scheme N m n c)
    (T : ℕ) (x : Model m n) (ξ : Samples Ω T) :
    execute (sampledOracle P.reply) (nesterovProtocol μ A T x) (fun _ => T) ξ =
      (run (Nesterov.next A P.reply) T (Nesterov.initial x) ξ).map Nesterov.points :=
  execute_batchProtocol μ _ _ _ _ _ _ _

theorem ema_score_eq (P : Problem N m n μ c) (A : Scheme N m n c)
    {γ : ℝ} (hγ : 0 < γ) (hsteps : ∀ i, (c i).γ = γ) (T : ℕ) (x : Model m n) :
    (∫ ξ, trajectoryScore P.grad
      (execute (sampledOracle P.reply) (emaProtocol μ A T x) (fun _ => T) ξ)
      ∂pathMeasure μ T) = P.stationarity A γ (Distributed.start x) T := by
  simp_rw [execute_ema, trajectoryScore, List.map_map, List.length_map, run_length]
  rw [integral_div]
  unfold Problem.stationarity
  have hcost : P.weightedCost γ =
      fun s => ∑ i, nuclearNorm (P.grad (Distributed.points s) i) ^ 2 := by
    funext s
    exact P.uniform_cost hγ hsteps s
  rw [hcost]
  rfl

theorem nesterov_score_eq (P : Problem N m n μ c) (A : Scheme N m n c)
    {γ : ℝ} (hγ : 0 < γ) (hsteps : ∀ i, (c i).γ = γ) (T : ℕ) (x : Model m n) :
    (∫ ξ, trajectoryScore P.grad
      (execute (sampledOracle P.reply) (nesterovProtocol μ A T x) (fun _ => T) ξ)
      ∂pathMeasure μ T) =
      Nesterov.smoothStationarity P A γ (Nesterov.initial x) T := by
  simp_rw [execute_nesterov, trajectoryScore, List.map_map, List.length_map, run_length]
  rw [integral_div]
  unfold Nesterov.smoothStationarity
  simp only [hsteps, div_self hγ.ne', one_mul, Function.comp_def]

end EF21Muon.LongTails
