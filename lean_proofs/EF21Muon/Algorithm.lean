import LongTails.Algorithms.MatrixPolar
import EF21Muon.Compression.ScaledSign

noncomputable section

namespace EF21Muon

open LongTails.Algorithms.MatrixPolar

variable {m n : ℕ}

def geometryConstant (m n : ℕ) : ℝ :=
  1 + ‖(operatorEquiv m n).symm.toContinuousLinearMap‖

structure State (m n : ℕ) where
  x : Space m n
  w : Space m n
  momentum : Space m n
  estimator : Space m n

def initial (x : Space m n) : State m n := ⟨x, x, 0, 0⟩

def signCompressor (x : Space m n) : Space m n :=
  WithLp.toLp 2 (Compression.scaledSign (fun i => x i))

def step (β γ : ℝ) (grad CU CP : Space m n → Space m n) (s : State m n) : State m n :=
  let x := s.x - (γ * nuclearNorm s.estimator) • polar s.estimator
  let w := s.w + CP (x - s.w)
  let momentum := (1 - β) • s.momentum + β • grad w
  let estimator := s.estimator + CU (momentum - s.estimator)
  ⟨x, w, momentum, estimator⟩

def trajectory (β γ : ℝ) (grad CU CP : Space m n → Space m n)
    (x₀ : Space m n) : ℕ → State m n
  | 0 => initial x₀
  | t + 1 => step β γ grad CU CP (trajectory β γ grad CU CP x₀ t)

def muonUSign (β γ : ℝ) (grad : Space m n → Space m n) (x₀ : Space m n) :=
  trajectory β γ grad signCompressor id x₀

def muonSign (β γ : ℝ) (grad : Space m n → Space m n) (x₀ : Space m n) :=
  trajectory β γ grad signCompressor signCompressor x₀

def Contracts (C : Space m n → Space m n) (q : ℝ) : Prop :=
  ∀ z, ‖C z - z‖ ≤ q * ‖z‖

def lag (grad : Space m n → Space m n) (s : State m n) : ℝ :=
  ‖s.momentum - grad s.x‖

def uplinkError (s : State m n) : ℝ := ‖s.estimator - s.momentum‖

def downlinkError (s : State m n) : ℝ := ‖s.w - s.x‖

def trackingError (B : ℝ) (grad : Space m n → Space m n) (s : State m n) : ℝ :=
  2 * lag grad s + uplinkError s + B * downlinkError s

end EF21Muon
