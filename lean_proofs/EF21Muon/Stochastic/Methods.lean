import EF21Muon.Stochastic.Convergence
import EF21Muon.Stochastic.MatrixNorms

noncomputable section

namespace EF21Muon.Stochastic

open LongTails.Algorithms.MatrixPolar

variable {m n : ℕ} {Ω : Type*} [Fintype Ω]

def signAlpha (m n : ℕ) : ℝ := 1 / ((m * n : ℕ) : ℝ)

theorem signAlpha_pos (hm : 0 < m) (hn : 0 < n) : 0 < signAlpha m n := by
  unfold signAlpha
  exact one_div_pos.mpr (by exact_mod_cast Nat.mul_pos hm hn)

theorem signAlpha_le_one (hm : 0 < m) (hn : 0 < n) : signAlpha m n ≤ 1 := by
  have hd : (1 : ℝ) ≤ ((m * n : ℕ) : ℝ) := by exact_mod_cast Nat.mul_pos hm hn
  exact (div_le_one (by exact_mod_cast Nat.mul_pos hm hn)).mpr hd

theorem sign_squared_contracts (hm : 0 < m) (hn : 0 < n) :
    SquaredContracts (signCompressor : Space m n → Space m n) (signAlpha m n) := by
  have : NeZero m := ⟨Nat.ne_of_gt hm⟩
  have : NeZero n := ⟨Nat.ne_of_gt hn⟩
  intro z
  simpa only [signCompressor, signAlpha, Compression.scaledSign,
    Fintype.card_prod, Fintype.card_fin] using
    Compression.scaledSign_euclidean_contracts z _ (Compression.compatible_signBit (fun i => z i))

theorem identity_squared_contracts : SquaredContracts (id : Space m n → Space m n) 1 := by
  intro z
  simp

def matrixParameters (m n : ℕ) (compressed : Bool) (β γ L : ℝ)
    (hm : 0 < m) (hn : 0 < n) (hβ : 0 < β) (hβ1 : β ≤ 1) (hγ : 0 < γ) (hL : 0 ≤ L) :
    Parameters where
  β := β
  γ := γ
  L := L
  αU := signAlpha m n
  αP := if compressed then signAlpha m n else 1
  ρ := Real.sqrt ((min m n : ℕ) : ℝ)
  β_pos := hβ
  β_le_one := hβ1
  γ_pos := hγ
  L_nonneg := hL
  αU_pos := signAlpha_pos hm hn
  αU_le_one := signAlpha_le_one hm hn
  αP_pos := by
    cases compressed with
    | false => norm_num
    | true => exact signAlpha_pos hm hn
  αP_le_one := by
    cases compressed with
    | false => exact le_rfl
    | true => exact signAlpha_le_one hm hn
  ρ_nonneg := Real.sqrt_nonneg _

/-- The paper's nuclear-variance assumption implies the Frobenius oracle model. -/
def Oracle.ofNuclearVariance {p : Sampling Ω} {grad : Space m n → Space m n} {σ : ℝ}
    (reply : Space m n → Ω → Space m n)
    (hunbiased : ∀ x, p.expect (reply x) = grad x)
    (hvariance : ∀ x, p.expect (fun ω => nuclearNorm (reply x ω - grad x) ^ 2) ≤ σ ^ 2) :
    Oracle p grad σ where
  reply := reply
  unbiased := hunbiased
  variance x := (p.expect_mono (fun ω =>
    pow_le_pow_left₀ (norm_nonneg _) (frobenius_le_nuclear (reply x ω - grad x)) 2)).trans
      (hvariance x)

def pathStationarity (c : Parameters) (p : Sampling Ω)
    (oracle : Space m n → Ω → Space m n) (CU CP : Space m n → Space m n)
    (f : Space m n → ℝ) (x₀ : Space m n) (T : ℕ) : ℝ :=
  (p.paths T).expect (fun ξ => ((run (advance c oracle CU CP) T (initial x₀) ξ).map
    (fun s => nuclearNorm (gradient f s.x) ^ 2)).sum) / T

theorem matrix_stationarity (compressed : Bool) {p : Sampling Ω}
    {f : Space m n → ℝ} {σ fmin β γ L : ℝ}
    (hm : 0 < m) (hn : 0 < n) (hβ : 0 < β) (hβ1 : β ≤ 1) (hγ : 0 < γ) (hL : 0 ≤ L)
    (O : Oracle p (gradient f) σ)
    (hg : ∀ x, HasGradientAt f (gradient f x) x)
    (hlip : ∀ x y, nuclearNorm (gradient f x - gradient f y) ≤ L * spectralNorm (x - y))
    (hmin : ∀ x, fmin ≤ f x)
    (hstep : let c := matrixParameters m n compressed β γ L hm hn hβ hβ1 hγ hL
      γ * (2 * L + 2 * Real.sqrt c.zeta) ≤ 1)
    (x₀ : Space m n) (T : ℕ) (hT : 0 < T) :
    let c := matrixParameters m n compressed β γ L hm hn hβ hβ1 hγ hL
    pathStationarity c p O.reply signCompressor (if compressed then signCompressor else id) f x₀ T ≤
      4 * (f x₀ - fmin) / (γ * T) +
        4 * c.momentumWeight * ‖gradient f x₀‖ ^ 2 / T + 4 * c.noise * σ ^ 2 := by
  let c := matrixParameters m n compressed β γ L hm hn hβ hβ1 hγ hL
  have hCP : SquaredContracts (m := m) (n := n) (if compressed then signCompressor else id) c.αP := by
    cases compressed
    · exact identity_squared_contracts
    · exact sign_squared_contracts hm hn
  have hdown : BroadcastBound (m := m) (n := n) c
      (if compressed then signCompressor else id) (c.ρ ^ 2) :=
    downlink_rec c hCP polar_frobenius_le_sqrt_rank
  have h := average_stationarity c (c.ρ ^ 2) O hg hlip hmin
    nuclearNorm_le_sqrt_rank (sign_squared_contracts hm hn) hdown
    (c.step_condition hstep) x₀ T hT
  rw [expectedTotal_eq_paths] at h
  exact h

theorem muonUSign_stationarity {p : Sampling Ω} {f : Space m n → ℝ} {σ fmin β γ L : ℝ}
    (hm : 0 < m) (hn : 0 < n) (hβ : 0 < β) (hβ1 : β ≤ 1) (hγ : 0 < γ) (hL : 0 ≤ L)
    (O : Oracle p (gradient f) σ)
    (hg : ∀ x, HasGradientAt f (gradient f x) x)
    (hlip : ∀ x y, nuclearNorm (gradient f x - gradient f y) ≤ L * spectralNorm (x - y))
    (hmin : ∀ x, fmin ≤ f x)
    (hstep : let c := matrixParameters m n false β γ L hm hn hβ hβ1 hγ hL
      γ * (2 * L + 2 * Real.sqrt c.zeta) ≤ 1)
    (x₀ : Space m n) (T : ℕ) (hT : 0 < T) :
    let c := matrixParameters m n false β γ L hm hn hβ hβ1 hγ hL
    pathStationarity c p O.reply signCompressor id f x₀ T ≤
      4 * (f x₀ - fmin) / (γ * T) +
        4 * c.momentumWeight * ‖gradient f x₀‖ ^ 2 / T + 4 * c.noise * σ ^ 2 :=
  matrix_stationarity false hm hn hβ hβ1 hγ hL O hg hlip hmin hstep x₀ T hT

theorem muonSign_stationarity {p : Sampling Ω} {f : Space m n → ℝ} {σ fmin β γ L : ℝ}
    (hm : 0 < m) (hn : 0 < n) (hβ : 0 < β) (hβ1 : β ≤ 1) (hγ : 0 < γ) (hL : 0 ≤ L)
    (O : Oracle p (gradient f) σ)
    (hg : ∀ x, HasGradientAt f (gradient f x) x)
    (hlip : ∀ x y, nuclearNorm (gradient f x - gradient f y) ≤ L * spectralNorm (x - y))
    (hmin : ∀ x, fmin ≤ f x)
    (hstep : let c := matrixParameters m n true β γ L hm hn hβ hβ1 hγ hL
      γ * (2 * L + 2 * Real.sqrt c.zeta) ≤ 1)
    (x₀ : Space m n) (T : ℕ) (hT : 0 < T) :
    let c := matrixParameters m n true β γ L hm hn hβ hβ1 hγ hL
    pathStationarity c p O.reply signCompressor signCompressor f x₀ T ≤
      4 * (f x₀ - fmin) / (γ * T) +
        4 * c.momentumWeight * ‖gradient f x₀‖ ^ 2 / T + 4 * c.noise * σ ^ 2 :=
  matrix_stationarity true hm hn hβ hβ1 hγ hL O hg hlip hmin hstep x₀ T hT

/-- Exact downlink: the same bound holds with every broadcast term removed from zeta. -/
theorem muonUSign_tighter_stationarity {p : Sampling Ω}
    {f : Space m n → ℝ} {σ fmin β γ L : ℝ}
    (hm : 0 < m) (hn : 0 < n) (hβ : 0 < β) (hβ1 : β ≤ 1) (hγ : 0 < γ) (hL : 0 ≤ L)
    (O : Oracle p (gradient f) σ)
    (hg : ∀ x, HasGradientAt f (gradient f x) x)
    (hlip : ∀ x y, nuclearNorm (gradient f x - gradient f y) ≤ L * spectralNorm (x - y))
    (hmin : ∀ x, fmin ≤ f x)
    (hstep : let c := matrixParameters m n false β γ L hm hn hβ hβ1 hγ hL
      γ * (2 * L + 2 * Real.sqrt (c.zetaWith 0)) ≤ 1)
    (x₀ : Space m n) (T : ℕ) (hT : 0 < T) :
    let c := matrixParameters m n false β γ L hm hn hβ hβ1 hγ hL
    pathStationarity c p O.reply signCompressor id f x₀ T ≤
      4 * (f x₀ - fmin) / (γ * T) +
        4 * c.momentumWeight * ‖gradient f x₀‖ ^ 2 / T + 4 * c.noise * σ ^ 2 := by
  let c := matrixParameters m n false β γ L hm hn hβ hβ1 hγ hL
  have h := average_stationarity c 0 O hg hlip hmin nuclearNorm_le_sqrt_rank
    (sign_squared_contracts hm hn) (identity_broadcast_bound c)
    (c.step_condition_for c.zetaWith_zero_nonneg hstep) x₀ T hT
  rw [expectedTotal_eq_paths] at h
  exact h

end EF21Muon.Stochastic
