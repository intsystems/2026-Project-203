import EF21Muon.Protocol
import EF21Muon.Controls
import Mathlib.Analysis.Calculus.Gradient.Basic

noncomputable section

namespace EF21Muon.Setups.Deterministic

open OracleTemplate LongTails.Algorithms.MatrixPolar

structure Public (m n : ℕ) where
  controls : Controls m n
  x₀ : Space m n
  Δ : ℝ
  G : ℝ
  ε : ℝ

def downlink (m n : ℕ) (compressed : Bool) : Space m n → Space m n :=
  if compressed then signCompressor else id

abbrev spec (m n : ℕ) (compressed : Bool) : Setup where
  Params := Public m n
  Public := Public m n
  Instance _ := Space m n → ℝ
  Query _ := Space m n
  Reply _ := Space m n
  Output _ := List (Space m n)
  publicInput := id
  valid p := 0 < m ∧ 0 < n ∧ 0 < p.ε ∧
    p.controls.qU = Real.sqrt (1 - 1 / (m * n : ℕ)) ∧
    p.controls.qP = if compressed then Real.sqrt (1 - 1 / (m * n : ℕ)) else 0
  admissible p f oracle :=
    (∀ x, HasGradientAt f (gradient f x) x) ∧
    (∀ x y, ‖gradient f x - gradient f y‖ ≤ p.controls.L * ‖x - y‖) ∧
    (∃ fmin, (∀ x, fmin ≤ f x) ∧ f p.x₀ - fmin ≤ p.Δ) ∧
    ‖gradient f p.x₀‖ ≤ p.G ∧
    ∀ history query seed, oracle.answer history (0, query) seed = gradient f query
  successful p f output _ := ∀ seed,
    (((output seed).map (fun x => nuclearNorm (gradient f x) ^ 2)).sum /
      (output seed).length) ≤ p.ε ^ 2
  allowedAlgorithm p A :=
    A = protocol p.controls.β p.controls.γ signCompressor (downlink m n compressed) p.x₀

end EF21Muon.Setups.Deterministic
