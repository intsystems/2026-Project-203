import EF21Muon.RandomTies.Generalized
import EF21Muon.Experiments.RandomizedBlocks

/-! Instantiation of the experimental-schedule theorem with the paper's binary
ties. The bit functions may be any measurable functions; independent fair bits
are one instance, given by coinLaw (or its product across rounds). -/

noncomputable section
namespace EF21Muon.RandomTies
set_option maxHeartbeats 800000
set_option backward.isDefEq.respectTransparency false
open MeasureTheory
open LongTails.Algorithms.MatrixPolar
open EF21Muon.Distributed EF21Muon.Stochastic
open EF21Muon.Experiments

variable {N m n : ℕ} {Ω : Type*} [MeasurableSpace Ω]

theorem measurable_random_compressor {B : Type*} [MeasurableSpace B]
    {bits : B → Fin m × Fin n → Bool} (hb : Measurable bits) :
    Measurable (fun z : B × Space m n => compressor (bits z.1) z.2) := by
  have hm : Measurable (fun z : (Fin m × Fin n → Bool) × Space m n =>
      compressor z.1 z.2) := measurable_from_prod_countable_right
        (fun b => measurable_compressor b)
  exact hm.comp ((hb.comp measurable_fst).prodMk measurable_snd)

/-- Replace the fixed scaled-sign compressors by independently sampled ties,
retaining every schedule, momentum, direction-quality and contraction constant. -/
def experimentalControls (c : Trajectory.Controls m n) (compressed : Bool)
    (hCU : c.CU = signCompressor)
    (hCP : c.CP = if compressed then signCompressor else id)
    (hm : 0 < m) (hn : 0 < n)
    (up : ℕ → Fin N → Ω → Fin m × Fin n → Bool)
    (down : ℕ → Ω → Fin m × Fin n → Bool)
    (hup : ∀ t j, Measurable (up t j)) (hdown : ∀ t, Measurable (down t)) :
    RandomizedTrajectory.Controls N m n Ω where
  β := c.β
  θ := c.θ
  qU := c.qU
  qP := c.qP
  B := c.B
  peak := c.peak
  β_pos := c.β_pos
  β_one := c.β_one
  θ_nonneg := c.θ_nonneg
  θ_one := c.θ_one
  qU_nonneg := c.qU_nonneg
  qU_one := c.qU_one
  qP_nonneg := c.qP_nonneg
  qP_one := c.qP_one
  B_nonneg := c.B_nonneg
  peak_nonneg := c.peak_nonneg
  η := c.η
  η_nonneg := c.η_nonneg
  η_peak := c.η_peak
  CU t j ω := compressor (up t j ω)
  CP t ω := if compressed then compressor (down t ω) else id
  measurable_CU t j := measurable_random_compressor (hup t j)
  measurable_CP t := by
    cases compressed with
    | false => exact measurable_snd
    | true => exact measurable_random_compressor (hdown t)
  contracts_CU t j ω x := by
    rw [norm_sub_rev, compressor_error_norm hm hn, ← hCU, norm_sub_rev]
    exact c.contracts_CU x
  contracts_CP t ω x := by
    cases compressed with
    | false => simp
               exact mul_nonneg c.qP_nonneg (norm_nonneg _)
    | true =>
      simp only [↓reduceIte] at hCP ⊢
      rw [norm_sub_rev, compressor_error_norm hm hn, ← hCP, norm_sub_rev]
      exact c.contracts_CP x
  direction := c.direction
  measurable_direction := c.measurable_direction
  direction_bound := c.direction_bound

/-- Fair independent entries within one fresh round; this supplies the concrete
compressor inputs used by emaNext and nestNext. -/
theorem measurable_upBits {p : ℕ} {m n : Fin p → ℕ} (i : Fin p) (j : Fin N) :
    Measurable (fun b : Coins N m n => upBits b i j) :=
  Measurable.of_eval (fun ij => measurable_pi_apply _)

theorem measurable_downBits {p : ℕ} {m n : Fin p → ℕ} (i : Fin p) :
    Measurable (fun b : Coins N m n => downBits b i) :=
  Measurable.of_eval (fun ij => measurable_pi_apply _)

section Constants
variable (c : Trajectory.Controls m n) (compressed : Bool)
    (hCU : c.CU = signCompressor) (hCP : c.CP = if compressed then signCompressor else id)
    (hm : 0 < m) (hn : 0 < n)
    (up : ℕ → Fin N → Ω → Fin m × Fin n → Bool)
    (down : ℕ → Ω → Fin m × Fin n → Bool)
    (hup : ∀ t j, Measurable (up t j)) (hdown : ∀ t, Measurable (down t))

/-- Randomizing ties adds no term to the explicit transient or residual budget. -/
theorem experimental_trackingBudget_eq {μ : Measure Ω} [IsProbabilityMeasure μ]
    (g : Trajectory.Gradients N m n) (O : Trajectory.Innovations N m n μ)
    (R : Trajectory.Roundoff N m n Ω) (x : Space m n) (T : ℕ) :
    RandomizedTrajectory.trackingBudget g
      (experimentalControls c compressed hCU hCP hm hn up down hup hdown) O R x T =
        Trajectory.trackingBudget g c O R x T := rfl
end Constants

/-- Concrete independent fair ties for each experimental layer and round.
The additional sample space Ω may carry the stochastic-gradient samples. -/
def iidExperimentalControls {p : ℕ} {m n : Fin p → ℕ}
    (c : ∀ i, Trajectory.Controls (m i) (n i)) (compressed : Bool)
    (hCU : ∀ i, (c i).CU = signCompressor)
    (hCP : ∀ i, (c i).CP = if compressed then signCompressor else id)
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (i : Fin p) :
    RandomizedTrajectory.Controls N (m i) (n i) (Stream N m n × Ω) :=
  experimentalControls (c i) compressed (hCU i) (hCP i) (hm i) (hn i)
    (fun t j ω => upBits (roundBits ω.1 t) i j)
    (fun t ω => downBits (roundBits ω.1 t) i)
    (fun t j => (measurable_upBits i j).comp ((measurable_roundBits t).comp measurable_fst))
    (fun t => (measurable_downBits i).comp ((measurable_roundBits t).comp measurable_fst))

end EF21Muon.RandomTies
