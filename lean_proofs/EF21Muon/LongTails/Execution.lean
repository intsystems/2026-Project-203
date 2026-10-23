import LongTails.Proofs.Execution
import EF21Muon.Probability.Paths

noncomputable section

namespace EF21Muon.LongTails

open OracleTemplate MeasureTheory
open EF21Muon.Probability EF21Muon.Stochastic

variable {Ω S Q R X : Type} [MeasurableSpace Ω] [Zero R]

def sampleList : (T : ℕ) → Samples Ω T → List Ω
  | 0, _ => []
  | T + 1, ξ => ξ.1 :: sampleList T ξ.2

def sampledOracle (reply : Q → Ω → R) : Oracle (Q × Option Ω) R where
  answer _ q _ := q.2.2.elim 0 (reply q.2.1)

def batchProtocol (μ : Measure Ω) [IsProbabilityMeasure μ] (T : ℕ)
    (query : S → Q) (update : S → R → S) (point : S → X) (s₀ : S) :
    Algorithm (Q × Option Ω) R (List X) where
  Seed := Samples Ω T
  law := ⟨pathMeasure μ T, inferInstance⟩
  State := S × List Ω × List X
  initial ξ := (s₀, sampleList T ξ, [])
  query z := (0, (query z.1, z.2.1.head?))
  update z r := (update z.1 r, z.2.1.tail, z.2.2 ++ [point z.1])
  output z := z.2.2

def sampledStep (query : S → Q) (update : S → R → S)
    (point : S → X) (reply : Q → Ω → R) (z : S × List Ω × List X) :
    S × List Ω × List X :=
  (update z.1 (z.2.1.head?.elim 0 (reply (query z.1))), z.2.1.tail,
    z.2.2 ++ [point z.1])

omit [MeasurableSpace Ω] in
theorem sampledStep_iterate (query : S → Q) (update : S → R → S)
    (point : S → X) (reply : Q → Ω → R) (T : ℕ) (s : S)
    (ξ : Samples Ω T) (xs : List X) :
    ((sampledStep query update point reply)^[T] (s, sampleList T ξ, xs)).2.2 =
      xs ++ (run (fun s ω => update s (reply (query s) ω)) T s ξ).map point := by
  induction T generalizing s xs with
  | zero => simp [run]
  | succ T ih =>
    rw [Function.iterate_succ_apply]
    change ((sampledStep query update point reply)^[T]
      (update s (reply (query s) ξ.1), sampleList T ξ.2, xs ++ [point s])).2.2 = _
    rw [ih]
    simp [run, List.append_assoc]

theorem execute_batchProtocol (μ : Measure Ω) [IsProbabilityMeasure μ]
    (query : S → Q) (update : S → R → S) (point : S → X)
    (reply : Q → Ω → R) (T : ℕ) (s₀ : S) (ξ : Samples Ω T) :
    execute (sampledOracle reply) (batchProtocol μ T query update point s₀)
      (fun _ => T) ξ =
      (run (fun s ω => update s (reply (query s) ω)) T s₀ ξ).map point := by
  exact (execute_iterate (fun q : Q × Option Ω => q.2.elim 0 (reply q.1))
    (batchProtocol μ T query update point s₀) T ξ).trans
      ((sampledStep_iterate query update point reply T s₀ ξ []).trans (List.nil_append _))

theorem execute_trace_length {Query Reply Output : Type*}
    (O : Oracle Query Reply) (A : Algorithm Query Reply Output) (T : ℕ)
    (seed : A.Seed) (ξ : (k : ℕ) → O.Seed k) :
    (executeWithTrace O A (fun _ => T) seed ξ).2.length = T := by
  rw [executeWithTrace_indexed]
  let step := fun sc : A.State × List ((Fin 1 × Query) × Reply) =>
    let q := A.query sc.1
    let r := O.answer sc.2 q (ξ sc.2.length)
    (A.update sc.1 r, sc.2 ++ [(q, r)])
  have h : ∀ T (s : A.State) (history : List ((Fin 1 × Query) × Reply)),
      (step^[T] (s, history)).2.length = history.length + T := by
    intro T
    induction T with
    | zero => intro s history; simp
    | succ T ih =>
      intro s history
      rw [Function.iterate_succ_apply]
      change (step^[T] (_, history ++ [_])).2.length = _
      rw [ih]
      simp [List.length_append]
      omega
  simpa using h T (A.initial seed) []

theorem worker_query_count {Query Reply Output : Type*}
    (O : Oracle Query Reply) (A : Algorithm Query Reply Output) (N T : ℕ)
    (seed : A.Seed) (ξ : (k : ℕ) → O.Seed k) :
    N * (executeWithTrace O A (fun _ => T) seed ξ).2.length = N * T := by
  rw [execute_trace_length]

omit [MeasurableSpace Ω] in
theorem run_length (next : S → Ω → S) (T : ℕ) (s : S) (ξ : Samples Ω T) :
    (run next T s ξ).length = T := by
  induction T generalizing s with
  | zero => rfl
  | succ T ih => simp [run, ih]

end EF21Muon.LongTails
