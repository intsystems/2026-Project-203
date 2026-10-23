import EF21Muon.Setups.Deterministic
import EF21Muon.Execution

noncomputable section

namespace EF21Muon.Proofs.Deterministic

open OracleTemplate LongTails.Algorithms.MatrixPolar
open EF21Muon.Setups.Deterministic

theorem convergence (m n : ℕ) (compressed : Bool) :
    Convergence (EF21Muon.Setups.Deterministic.spec m n compressed)
      (fun p _ => max 1 ⌈(8 * p.Δ / p.controls.γ +
        64 * geometryConstant m n ^ 2 * p.G ^ 2 / p.controls.δ) / p.ε ^ 2⌉₊) := by
  refine ⟨fun p => protocol p.controls.β p.controls.γ signCompressor
    (downlink m n compressed) p.x₀, ?_, ?_⟩
  · intro p hp f oracle ho
    rcases hp with ⟨hm, hn, hε, hqU, hqP⟩
    rcases ho with ⟨hg, hlip, ⟨fmin, hmin, hΔ⟩, hG, horacle⟩
    have hCU : Contracts (signCompressor : Space m n → Space m n) p.controls.qU := by
      rw [hqU]
      exact signCompressor_contracts hm hn
    have hCP : Contracts (downlink m n compressed) p.controls.qP := by
      cases compressed with
      | false => simpa only [downlink, Bool.false_eq_true, ↓reduceIte, hqP] using
          (identity_contracts (m := m) (n := n))
      | true => simpa only [downlink, ↓reduceIte, hqP, signContraction] using
          signCompressor_contracts hm hn
    have h := protocol_stationarity p.controls hg hlip hmin hCU hCP p.x₀ hΔ hG hε
    change ∀ seed, _
    intro seed
    have he := execute_eq_of_answer_eq oracle (gradient f) horacle
      (protocol p.controls.β p.controls.γ signCompressor (downlink m n compressed) p.x₀)
      (iterationBudget p.controls p.Δ p.G p.ε) seed.1 seed.2
    have hunit : seed.1 = () := @Subsingleton.elim Unit inferInstance _ _
    rw [hunit] at he
    change (((execute oracle (protocol p.controls.β p.controls.γ signCompressor
      (downlink m n compressed) p.x₀) (fun _ => iterationBudget p.controls p.Δ p.G p.ε)
      seed.1 seed.2).map (fun x => nuclearNorm (gradient f x) ^ 2)).sum /
      (execute oracle (protocol p.controls.β p.controls.γ signCompressor
        (downlink m n compressed) p.x₀) (fun _ => iterationBudget p.controls p.Δ p.G p.ε)
        seed.1 seed.2).length) ≤ p.ε ^ 2
    rw [hunit, he]
    exact h
  · intro p _
    rfl

end EF21Muon.Proofs.Deterministic
