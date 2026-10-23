import EF21Muon.Divergence.EF21Cycle

/-! The sign-after-polar EF21 algorithm of the negative theorem. This state
and update are separate from the convergent gradient-compression algorithms. -/

noncomputable section

namespace EF21Muon.Divergence.EF21SignMuon

open LongTails.Algorithms.MatrixPolar

structure State where
  x : Space 2 2
  momentum : Space 2 2
  estimator : Space 2 2

def effective (variant : MomentumVariant) (μ : ℝ) (old new : Space 2 2) : Space 2 2 :=
  match variant with
  | .standard => new
  | .nesterov => (1+μ) • new - μ • old

def step (variant : MomentumVariant) (μ η : ℝ) (grad : Space 2 2 → Space 2 2)
    (s : State) : State :=
  let M := μ • s.momentum + (1-μ) • grad s.x
  let d := estimateStep (polar (effective variant μ s.momentum M)) s.estimator
  ⟨s.x - η • d, M, d⟩

def run (variant : MomentumVariant) (μ η : ℝ) (grad : Space 2 2 → Space 2 2) : ℕ → State
  | 0 => ⟨0,0,0⟩
  | t+1 => step variant μ η grad (run variant μ η grad t)

theorem effective_smul (variant : MomentumVariant) (μ c : ℝ) (old new : Space 2 2) :
    effective variant μ (c • old) (c • new) = c • effective variant μ old new := by
  cases variant
  · rfl
  · simp only [effective, smul_sub, smul_smul]
    congr 1 <;> congr 1 <;> ring

theorem run_of_prescribed (variant : MomentumVariant) (μ : ℝ)
    (grad : Space 2 2 → Space 2 2) (M : ℕ → Space 2 2) (hM₀ : M 0 = 0)
    (hM : ∀ t, M (t+1) = μ • M t + (1-μ) • grad (position t))
    (hpolar : ∀ t, polar (effective variant μ (M t) (M (t+1))) = target (t+1)) :
    ∀ t, run variant μ 1 grad t = ⟨position t, M t, estimate t⟩ := by
  intro t
  induction t with
  | zero => simp [run, position, estimate, hM₀]
  | succ t ih =>
    rw [run, ih]
    simp only [step, ← hM t, hpolar t, estimate, position, one_smul]

def rescale (η c : ℝ) (s : State) : State := ⟨η • s.x, c • s.momentum, s.estimator⟩

theorem run_scale (variant : MomentumVariant) (μ : ℝ) {η c : ℝ}
    (hη : η ≠ 0) (hc : 0 < c) (grad : Space 2 2 → Space 2 2) (t : ℕ) :
    run variant μ η (fun x => c • grad (η⁻¹ • x)) t =
      rescale η c (run variant μ 1 grad t) := by
  induction t with
  | zero => simp [run, rescale]
  | succ t ih =>
    rw [run, ih, run]
    simp only [step, rescale, smul_smul, inv_mul_cancel₀ hη, one_smul]
    have hM (s : State) : (μ*c) • s.momentum + ((1-μ)*c) • grad s.x =
        c • (μ • s.momentum + (1-μ) • grad s.x) := by
      simp only [smul_add, smul_smul]
      congr 1 <;> congr 1 <;> ring
    rw [hM, effective_smul, polar_smul_pos hc]
    simp only [smul_sub]

end EF21Muon.Divergence.EF21SignMuon
