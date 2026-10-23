import EF21Muon.Distributed.GeneralizedRates
import EF21Muon.Distributed.Correspondence

/-! Erasing the analysis-only noise EMA, and identifying the resulting update
with the paper's federated phases via the round rotation of Proposition 2. -/

noncomputable section

namespace EF21Muon.Distributed

open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω] {μ : MeasureTheory.Measure Ω}

def radiusActualNext (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters}
    (A : Scheme N m n c) (s : States N m n) (ω : Ω) : States N m n :=
  fun i => radiusUpdate (c i) (A.CU i) (s i) (fun j => P.reply (radiusPoints c s) ω j i)

theorem radiusNext_erases (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters}
    (A : Scheme N m n c) (z : RadiusStates N m n) (ω : Ω) :
    (radiusNext P A z ω).1 = radiusActualNext P A z.1 ω := rfl

theorem radius_run_erases (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters}
    (A : Scheme N m n c) (k : ℕ) (s : States N m n) (Z : Model m n) (ξ : Samples Ω k) :
    (run (radiusNext P A) k (s, Z) ξ).map Prod.fst = run (radiusActualNext P A) k s ξ := by
  induction k generalizing s Z with
  | zero => rfl
  | succ k ih =>
    change s :: (run (radiusNext P A) k (radiusNext P A (s,Z) ξ.1) ξ.2).map Prod.fst =
      s :: run (radiusActualNext P A) k (radiusActualNext P A s ξ.1) ξ.2
    have he : radiusNext P A (s,Z) ξ.1 =
        (radiusActualNext P A s ξ.1, (radiusNext P A (s,Z) ξ.1).2) := rfl
    rw [he, ih]

theorem radius_pathState_erases (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters}
    (A : Scheme N m n c) (k : ℕ) (s : States N m n) (Z : Model m n)
    (ξ : Samples Ω k) (t : Fin k) :
    (pathState (radiusNext P A) k (s,Z) ξ t).1 = pathState (radiusActualNext P A) k s ξ t := by
  induction k generalizing s Z with
  | zero => exact Fin.elim0 t
  | succ k ih =>
    refine Fin.cases rfl (fun l => ?_) t
    change (pathState (radiusNext P A) k (radiusNext P A (s,Z) ξ.1) ξ.2 l).1 = _
    have he : radiusNext P A (s,Z) ξ.1 =
        (radiusActualNext P A s ξ.1, (radiusNext P A (s,Z) ξ.1).2) := rfl
    rw [he, ih]
    rfl

namespace Correspondence

def ofRadiusScheme (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters}
    (A : Scheme N m n c) (ω : ℕ → Ω) : Operations N m n where
  μ i := 1-(c i).β
  uplink _ i _ := A.CU i
  downlink _ _ := id
  reply t x := P.reply x (ω t)
  step _ x g := radiusStep (fun i => (c i).γ) x g
  step_zero _ x := by funext i; simp [radiusStep, polar_zero]
  downlink_zero _ _ := rfl

theorem ofRadiusScheme_round (P : GeneralizedProblem N m n μ) {c : Fin p → Parameters}
    (A : Scheme N m n c) (ω : ℕ → Ω) (s : States N m n) (t k : ℕ) :
    workerPhase (ofRadiusScheme P A ω) t (serverPhase (ofRadiusScheme P A ω) k s) =
      radiusActualNext P A s (ω t) := by
  have hb (i : Fin p) : 1-(1-(c i).β) = (c i).β := by ring
  have hw (i : Fin p) : (s i).w+(radiusPoint (c i) (s i)-(s i).w) =
      radiusPoint (c i) (s i) := by module
  have hQ : (fun i => (s i).2.1 + ((s i).1 - (c i).γ • polar (average (s i).2.2.2) - (s i).2.1)) =
      radiusPoints c s := by
    funext i
    dsimp [radiusPoints, radiusPoint, LayerState.x, LayerState.G]
    module
  funext i
  simp only [workerPhase, serverPhase, ofRadiusScheme, hb, radiusStep, points,
    radiusActualNext, radiusUpdate, LayerState.x, LayerState.w, LayerState.M, LayerState.G,
    id_eq]
  change (radiusPoint (c i) (s i), (s i).w+(radiusPoint (c i) (s i)-(s i).w), _, _) = _
  simp only [hw, hQ]

end Correspondence

end EF21Muon.Distributed
