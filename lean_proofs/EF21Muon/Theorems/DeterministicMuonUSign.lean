import EF21Muon.Setups.Deterministic
import EF21Muon.Proofs.Deterministic

namespace EF21Muon.Theorems.DeterministicMuonUSign

open OracleTemplate

noncomputable def budget (m n : ℕ) (p : EF21Muon.Setups.Deterministic.Public m n) : ℕ := max 1 ⌈(8 * p.Δ / p.controls.γ + 64 * geometryConstant m n ^ 2 * p.G ^ 2 / p.controls.δ) / p.ε ^ 2⌉₊

theorem result (m n : ℕ) : Convergence (EF21Muon.Setups.Deterministic.spec m n false)
    (fun p _ => budget m n p) := EF21Muon.Proofs.Deterministic.convergence m n false

end EF21Muon.Theorems.DeterministicMuonUSign
