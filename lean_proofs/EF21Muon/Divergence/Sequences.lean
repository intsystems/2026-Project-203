import EF21Muon.Divergence.Linear

noncomputable section

namespace EF21Muon.Divergence

open Filter

theorem twoStep_formula {u : ℕ → ℝ} {c : ℝ} (start : ℕ)
    (h : ∀ t, start ≤ t → u (t+2) = u t + c) (k : ℕ) :
    u (2*k+start) = u start + (k : ℝ)*c := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [show 2*(k+1)+start = (2*k+start)+2 by omega, h _ (by omega), ih]
    push_cast
    ring

theorem tendsto_atTop_of_twoStep {u : ℕ → ℝ} {c : ℝ} (hc : 0 < c)
    (start : ℕ) (h : ∀ t, start ≤ t → u (t+2) = u t + c) :
    Tendsto u atTop atTop := by
  have hodd : ∀ t, start+1 ≤ t → u (t+2) = u t+c := fun t ht => h t (by omega)
  apply tendsto_atTop.mpr
  intro b
  obtain ⟨N,hN⟩ := exists_nat_gt ((b - min (u start) (u (start+1))) / c)
  have hNc : b - min (u start) (u (start+1)) < (N : ℝ)*c := (div_lt_iff₀ hc).mp hN
  filter_upwards [eventually_ge_atTop (2*N+start+1)] with t ht
  have hm := Nat.mod_add_div (t-start) 2
  have hml := Nat.mod_lt (t-start) (by omega : 0 < 2)
  by_cases he : (t-start)%2 = 0
  · let k := (t-start)/2
    have htk : t = 2*k+start := by dsimp [k]; omega
    have hk : N ≤ k := by dsimp [k]; omega
    rw [htk, twoStep_formula start h k]
    have hk' : (N : ℝ) ≤ k := by exact_mod_cast hk
    have hh := mul_le_mul_of_nonneg_right hk' hc.le
    linarith [min_le_left (u start) (u (start+1))]
  · let k := (t-start)/2
    have htk : t = 2*k+(start+1) := by dsimp [k]; omega
    have hk : N ≤ k := by dsimp [k]; omega
    rw [htk, twoStep_formula (start+1) hodd k]
    have hk' : (N : ℝ) ≤ k := by exact_mod_cast hk
    have hh := mul_le_mul_of_nonneg_right hk' hc.le
    linarith [min_le_right (u start) (u (start+1))]

end EF21Muon.Divergence
