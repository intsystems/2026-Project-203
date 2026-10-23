import EF21Muon.Algorithm
import LongTails.Template

noncomputable section

namespace EF21Muon

open OracleTemplate LongTails.Algorithms.MatrixPolar

variable {m n : ℕ}

def queryPoint (γ : ℝ) (CP : Space m n → Space m n) (s : State m n) : Space m n :=
  s.w + CP (s.x - (γ * nuclearNorm s.estimator) • polar s.estimator - s.w)

def replyStep (β γ : ℝ) (CU CP : Space m n → Space m n)
    (s : State m n) (reply : Space m n) : State m n :=
  let x := s.x - (γ * nuclearNorm s.estimator) • polar s.estimator
  let w := queryPoint γ CP s
  let momentum := (1 - β) • s.momentum + β • reply
  let estimator := s.estimator + CU (momentum - s.estimator)
  ⟨x, w, momentum, estimator⟩

def protocol (β γ : ℝ) (CU CP : Space m n → Space m n) (x₀ : Space m n) :
    Algorithm (Space m n) (Space m n) (List (Space m n)) where
  State := EF21Muon.State m n × List (Space m n)
  initial _ := (initial x₀, [])
  query s := (0, queryPoint γ CP s.1)
  update s reply := (replyStep β γ CU CP s.1 reply, s.2 ++ [s.1.x])
  output s := s.2

end EF21Muon
