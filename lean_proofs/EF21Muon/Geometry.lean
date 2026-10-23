import EF21Muon.Algorithm
import LongTails.Proofs.MatrixPolarGeometry

noncomputable section

namespace EF21Muon

open LongTails.Algorithms.MatrixPolar

variable {m n : ℕ}

theorem geometryConstant_pos : 0 < geometryConstant m n := by
  unfold geometryConstant
  positivity

theorem nuclearNorm_le_geometry (z : Space m n) :
    nuclearNorm z ≤ geometryConstant m n * ‖z‖ := by
  have h := ContinuousLinearMap.opNorm_comp_le (innerSL ℝ z)
    (operatorEquiv m n).symm.toContinuousLinearMap
  have hi : ‖innerSL ℝ z‖ = ‖z‖ := by simp
  rw [hi] at h
  unfold nuclearNorm
  apply h.trans
  unfold geometryConstant
  nlinarith [norm_nonneg z]

theorem norm_polar_le_geometry (z : Space m n) :
    ‖polar z‖ ≤ geometryConstant m n := by
  have h := (operatorEquiv m n).symm.toContinuousLinearMap.le_opNorm
    (operatorEquiv m n (polar z))
  simp only [ContinuousLinearEquiv.coe_coe, ContinuousLinearEquiv.symm_apply_apply] at h
  have hp : spectralNorm (polar z) ≤ 1 := polar_mem z
  change ‖polar z‖ ≤ ‖(operatorEquiv m n).symm.toContinuousLinearMap‖ *
    spectralNorm (polar z) at h
  have hn := norm_nonneg (operatorEquiv m n).symm.toContinuousLinearMap
  unfold geometryConstant
  nlinarith [mul_le_mul_of_nonneg_left hp hn]

theorem identity_contracts : Contracts (id : Space m n → Space m n) 0 := by
  intro z
  simp

def signContraction (m n : ℕ) : ℝ := Real.sqrt (1 - 1 / (m * n : ℕ))

theorem signContraction_nonneg : 0 ≤ signContraction m n := Real.sqrt_nonneg _

theorem signContraction_lt_one (hm : 0 < m) (hn : 0 < n) :
    signContraction m n < 1 := by
  have hd : 0 < ((m * n : ℕ) : ℝ) := by exact_mod_cast Nat.mul_pos hm hn
  unfold signContraction
  have hq : 0 ≤ 1 - 1 / ((m * n : ℕ) : ℝ) := by
    have hd1 : (1 : ℝ) ≤ ((m * n : ℕ) : ℝ) := by
      exact_mod_cast Nat.one_le_iff_ne_zero.mpr (Nat.ne_of_gt (Nat.mul_pos hm hn))
    have := (div_le_one hd).mpr hd1
    linarith
  apply (Real.sqrt_lt hq (by norm_num : (0 : ℝ) ≤ 1)).mpr
  have := one_div_pos.mpr hd
  simpa only [one_pow] using (sub_lt_self (1 : ℝ) (one_div_pos.mpr hd))

theorem signCompressor_contracts (hm : 0 < m) (hn : 0 < n) :
    Contracts (signCompressor : Space m n → Space m n) (signContraction m n) := by
  have : NeZero m := ⟨Nat.ne_of_gt hm⟩
  have : NeZero n := ⟨Nat.ne_of_gt hn⟩
  intro z
  have h := Compression.scaledSign_euclidean_contracts z _
    (Compression.compatible_signBit (fun i => z i))
  have hd : (1 : ℝ) ≤ ((m * n : ℕ) : ℝ) := by
    exact_mod_cast Nat.one_le_iff_ne_zero.mpr (Nat.ne_of_gt (Nat.mul_pos hm hn))
  have hq : 0 ≤ 1 - 1 / ((m * n : ℕ) : ℝ) := by
    have := (div_le_one (by linarith : (0 : ℝ) < ((m * n : ℕ) : ℝ))).mpr hd
    linarith
  have hs := Real.sq_sqrt hq
  apply (sq_le_sq₀ (norm_nonneg _) (mul_nonneg signContraction_nonneg (norm_nonneg z))).mp
  simpa only [signCompressor, Compression.scaledSign, Fintype.card_prod,
    Fintype.card_fin, signContraction, mul_pow, hs] using h

theorem trackingError_nonneg {B : ℝ} (hB : 0 ≤ B)
    (grad : Space m n → Space m n) (s : State m n) : 0 ≤ trackingError B grad s := by
  unfold trackingError lag uplinkError downlinkError
  positivity

theorem estimator_error_le_tracking {B : ℝ} (hB : 0 ≤ B)
    (grad : Space m n → Space m n) (s : State m n) :
    ‖s.estimator - grad s.x‖ ≤ trackingError B grad s := by
  have h := norm_sub_le_norm_sub_add_norm_sub s.estimator s.momentum (grad s.x)
  unfold trackingError lag uplinkError downlinkError
  nlinarith [norm_nonneg (s.momentum - grad s.x),
    mul_nonneg hB (norm_nonneg (s.w - s.x))]

theorem gradient_sq_le_tracking {B : ℝ} (hB : 0 ≤ B)
    (grad : Space m n → Space m n) (s : State m n) :
    nuclearNorm (grad s.x) ^ 2 ≤ 2 * nuclearNorm s.estimator ^ 2 +
      2 * geometryConstant m n ^ 2 * trackingError B grad s ^ 2 := by
  have ht := nuclearNorm_add_le s.estimator (grad s.x - s.estimator)
  rw [add_sub_cancel] at ht
  have he := nuclearNorm_le_geometry (grad s.x - s.estimator)
  rw [norm_sub_rev] at he
  have he' := mul_le_mul_of_nonneg_left (estimator_error_le_tracking hB grad s)
    (geometryConstant_pos (m := m) (n := n)).le
  have h : nuclearNorm (grad s.x) ≤ nuclearNorm s.estimator +
      geometryConstant m n * trackingError B grad s := by linarith
  have hs := mul_self_le_mul_self (nuclearNorm_nonneg _) h
  nlinarith [sq_nonneg (nuclearNorm s.estimator -
    geometryConstant m n * trackingError B grad s)]

end EF21Muon
