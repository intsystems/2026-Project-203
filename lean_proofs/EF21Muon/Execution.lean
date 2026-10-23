import EF21Muon.Protocol
import EF21Muon.Budget
import LongTails.Proofs.Execution

noncomputable section

namespace EF21Muon

open OracleTemplate LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {m n : ℕ}

theorem execute_protocol (β γ : ℝ) (grad CU CP : Space m n → Space m n)
    (x₀ : Space m n) (T : ℕ) :
    execute { answer := fun _ query _ => grad query.2 } (protocol β γ CU CP x₀) (fun _ => T) =
      (List.range T).map (fun t => (trajectory β γ grad CU CP x₀ t).x) := by
  refine (execute_iterate grad (protocol β γ CU CP x₀) T ()).trans ?_
  have hi : ∀ k, ((fun s : State m n × List (Space m n) =>
      (replyStep β γ CU CP s.1 (grad (queryPoint γ CP s.1)), s.2 ++ [s.1.x]))^[k]
        (initial x₀, [])) =
      (trajectory β γ grad CU CP x₀ k,
        (List.range k).map (fun t => (trajectory β γ grad CU CP x₀ t).x)) := by
    intro k
    induction k with
    | zero => rfl
    | succ k ih =>
      rw [Function.iterate_succ_apply', ih, List.range_succ, List.map_append]
      rfl
  change ((fun s : State m n × List (Space m n) =>
    (replyStep β γ CU CP s.1 (grad (queryPoint γ CP s.1)), s.2 ++ [s.1.x]))^[T]
      (initial x₀, [])).2 = _
  rw [hi]

theorem execute_protocol_length (β γ : ℝ) (grad CU CP : Space m n → Space m n)
    (x₀ : Space m n) (T : ℕ) :
    (execute { answer := fun _ query _ => grad query.2 } (protocol β γ CU CP x₀)
      (fun _ => T)).length = T := by
  rw [execute_protocol]
  simp

theorem list_range_sum (a : ℕ → ℝ) (T : ℕ) :
    ((List.range T).map a).sum = ∑ t ∈ Finset.range T, a t := by
  induction T with
  | zero => simp
  | succ T ih => simp [List.range_succ, Finset.sum_range_succ, ih]

theorem protocol_stationarity (c : Controls m n) {f : Space m n → ℝ}
    {grad CU CP : Space m n → Space m n} {fmin Δ G ε : ℝ}
    (hg : ∀ x, HasGradientAt f (grad x) x)
    (hlip : ∀ x y, ‖grad x - grad y‖ ≤ c.L * ‖x - y‖)
    (hmin : ∀ x, fmin ≤ f x) (hCU : Contracts CU c.qU) (hCP : Contracts CP c.qP)
    (x₀ : Space m n) (hΔ : f x₀ - fmin ≤ Δ) (hG : ‖grad x₀‖ ≤ G) (hε : 0 < ε) :
    let xs := execute { answer := fun _ query _ => grad query.2 }
      (protocol c.β c.γ CU CP x₀) (fun _ => iterationBudget c Δ G ε)
    (xs.map (fun x => nuclearNorm (grad x) ^ 2)).sum / xs.length ≤ ε ^ 2 := by
  dsimp only
  rw [execute_protocol, List.map_map, List.length_map, List.length_range, list_range_sum]
  exact stationarity_at_budget c hg hlip hmin hCU hCP x₀ hΔ hG hε

end EF21Muon
