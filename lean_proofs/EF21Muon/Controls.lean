import EF21Muon.Algorithm

namespace EF21Muon

structure Controls (m n : ℕ) where
  β : ℝ
  γ : ℝ
  L : ℝ
  qU : ℝ
  qP : ℝ
  B : ℝ
  δ : ℝ
  C : ℝ
  β_pos : 0 < β
  β_le_one : β ≤ 1
  γ_pos : 0 < γ
  L_nonneg : 0 ≤ L
  qU_nonneg : 0 ≤ qU
  qU_le_one : qU ≤ 1
  qP_nonneg : 0 ≤ qP
  B_nonneg : 0 ≤ B
  δ_pos : 0 < δ
  δ_le_one : δ ≤ 1
  C_nonneg : 0 ≤ C
  lag_decay : 2 * δ ≤ β
  uplink_decay : qU ≤ 1 - δ
  downlink_decay : qP * (B + 3 * β * L) ≤ (1 - δ) * B
  movement : 2 * L + qP * (B + 3 * β * L) ≤ C
  smooth_step : 2 * L * geometryConstant m n ^ 2 * γ ≤ 1
  tracking_step : 4 * geometryConstant m n ^ 2 * C * γ ≤ δ

end EF21Muon
