import EF21Muon.Divergence.Scaling

noncomputable section

namespace EF21Muon.Divergence

open LongTails.Algorithms.MatrixPolar InnerProductSpace
open scoped ContDiff

def linearObjective {m n : ℕ} (G X : Space m n) : ℝ := inner ℝ G X

theorem linearObjective_contDiff {m n : ℕ} (G : Space m n) :
    ContDiff ℝ ∞ (linearObjective G) := (innerSL ℝ G).contDiff

theorem linearObjective_hasGradient {m n : ℕ} (G X : Space m n) :
    HasGradientAt (linearObjective G) G X :=
  hasGradientAt_iff_hasFDerivAt.mpr (innerSL ℝ G).hasFDerivAt

theorem linearObjective_gradient {m n : ℕ} (G X : Space m n) :
    gradient (linearObjective G) X = G := (linearObjective_hasGradient G X).gradient

theorem linearObjective_spectralSmooth {m n : ℕ} (G : Space m n) {L : ℝ} (hL : 0 ≤ L) :
    SpectralSmooth (linearObjective G) L := by
  intro X Y
  rw [linearObjective_gradient, linearObjective_gradient, sub_self]
  have hs : 0 ≤ spectralNorm (X-Y) := norm_nonneg (operatorEquiv m n (X-Y))
  simpa [nuclearNorm, ContinuousLinearMap.opNorm_zero] using mul_nonneg hL hs

end EF21Muon.Divergence
