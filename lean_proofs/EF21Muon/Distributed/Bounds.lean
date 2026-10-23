import EF21Muon.Distributed.Convergence

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def Problem.weightedCost (P : Problem N m n μ c) (γbar : ℝ) (s : States N m n) : ℝ :=
  4 / γbar * P.cost s

omit [IsProbabilityMeasure μ] in
theorem Problem.weightedCost_eq (P : Problem N m n μ c) (γbar : ℝ) (s : States N m n) :
    P.weightedCost γbar s = ∑ i, (c i).γ / γbar * nuclearNorm (P.grad (points s) i) ^ 2 := by
  unfold Problem.weightedCost Problem.cost
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _; ring

def Problem.stationarity (P : Problem N m n μ c) (A : Scheme N m n c)
    (γbar : ℝ) (s : States N m n) (T : ℕ) : ℝ :=
  (∫ ξ, ((run (next A P.reply) T s ξ).map (P.weightedCost γbar)).sum ∂pathMeasure μ T) / T

theorem sum_map_scale {S : Type*} (a : ℝ) (f : S → ℝ) (xs : List S) :
    (xs.map (fun s => a * f s)).sum = a * (xs.map f).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp only [List.map_cons, List.sum_cons, ih, mul_add]

/-- The paper's weighted average, with gammaBar arbitrary and positive.
Set gammaBar to the arithmetic mean of the layer steps to obtain its normalization. -/
theorem Problem.average_stationarity (P : Problem N m n μ c) (A : Scheme N m n c)
    (hsmall : ∀ i, 2 * (c i).L * (c i).γ + 4 * zeta (c i) (P.ell2 i) (A.τ i) * (c i).γ ^ 2 ≤ 1)
    {γbar : ℝ} (hγ : 0 < γbar) (s : States N m n) (T : ℕ) (hT : 0 < T) :
    P.stationarity A γbar s T ≤ 4 * P.potential s / (γbar * T) + 4 * P.noiseTotal / γbar := by
  have h := (P.finite_time A hsmall T s).2
  have hTr : (0 : ℝ) < T := by exact_mod_cast hT
  unfold Problem.stationarity Problem.weightedCost
  simp_rw [sum_map_scale, integral_const_mul]
  apply (div_le_iff₀ hTr).mpr
  have hh := mul_le_mul_of_nonneg_left h (by positivity : 0 ≤ 4 / γbar)
  have he : (4 * P.potential s / (γbar * T) + 4 * P.noiseTotal / γbar) * T =
      4 / γbar * (P.potential s + T * P.noiseTotal) := by field_simp
  rw [he]
  exact hh

def start (x : Model m n) : States N m n := fun i => LayerState.start (x i)

omit [IsProbabilityMeasure μ] in
theorem Problem.initial_potential (P : Problem N m n μ c) (x : Model m n) :
    P.potential (start x) = P.f x - P.fmin + ∑ i, (c i).γ *
      (globalWeight (c i) * ‖P.grad x i‖ ^ 2 +
        localWeight (c i) * average (fun j => ‖P.localGrad j x i‖ ^ 2)) := by
  have hx : points (start x : States N m n) = x := rfl
  unfold Problem.potential
  rw [hx]
  simp [start, LayerState.start, energy, globalError,
    localError, compressionError, broadcastError, LayerState.M, LayerState.G,
    LayerState.x, LayerState.w, Probability.average]

omit [IsProbabilityMeasure μ] in
/-- Uniform steps give the unweighted sum of squared nuclear gradient norms. -/
theorem Problem.uniform_cost (P : Problem N m n μ c) {γ : ℝ} (hγ : 0 < γ)
    (hsteps : ∀ i, (c i).γ = γ) (s : States N m n) :
    P.weightedCost γ s = ∑ i, nuclearNorm (P.grad (points s) i) ^ 2 := by
  rw [P.weightedCost_eq]
  simp only [hsteps, div_self (ne_of_gt hγ), one_mul]

end EF21Muon.Distributed
