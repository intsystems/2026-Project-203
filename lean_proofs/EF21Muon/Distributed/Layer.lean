import EF21Muon.Probability.Updates
import EF21Muon.Distributed.Constants
import EF21Muon.Stochastic.Convergence

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

/-- One layer: shared server/broadcast matrices and separate states for every worker. -/
abbrev LayerState (N m n : ℕ) :=
  Space m n × Space m n × (Fin N → Space m n) × (Fin N → Space m n)

namespace LayerState
variable {N m n : ℕ}
def x (s : LayerState N m n) := s.1
def w (s : LayerState N m n) := s.2.1
def M (s : LayerState N m n) := s.2.2.1
def G (s : LayerState N m n) := s.2.2.2
def server (s : LayerState N m n) : EF21Muon.State m n :=
  ⟨s.x, s.w, average s.M, average s.G⟩
def start (x : Space m n) : LayerState N m n := (x, x, 0, 0)
end LayerState

variable {N m n : ℕ}

def nextX (c : Parameters) (s : LayerState N m n) := serverPoint c.γ s.server
def nextW (c : Parameters) (CP : Space m n → Space m n) (s : LayerState N m n) :=
  queryPoint c.γ CP s.server

def update (c : Parameters) (CU CP : Space m n → Space m n)
    (s : LayerState N m n) (reply : Fin N → Space m n) : LayerState N m n :=
  let M := fun j => (1 - c.β) • s.M j + c.β • reply j
  (nextX c s, nextW c CP s, M, fun j => s.G j + CU (M j - s.G j))

def globalError (s : LayerState N m n) (g : Space m n) : ℝ := ‖average s.M - g‖ ^ 2
def localError (s : LayerState N m n) (g : Fin N → Space m n) : ℝ :=
  average (fun j => ‖s.M j - g j‖ ^ 2)
def compressionError (s : LayerState N m n) : ℝ := average (fun j => ‖s.M j - s.G j‖ ^ 2)
def broadcastError (s : LayerState N m n) : ℝ := ‖s.w - s.x‖ ^ 2

theorem average_nonneg {a : Fin N → ℝ} (ha : ∀ j, 0 ≤ a j) : 0 ≤ average a := by
  unfold average
  change 0 ≤ (N : ℝ)⁻¹ * ∑ j, a j
  exact mul_nonneg (inv_nonneg.mpr (Nat.cast_nonneg N)) (Finset.sum_nonneg fun j _ => ha j)

theorem average_mono {a b : Fin N → ℝ} (h : ∀ j, a j ≤ b j) : average a ≤ average b := by
  unfold average
  exact mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun j _ => h j)
    (inv_nonneg.mpr (Nat.cast_nonneg N))

theorem average_mul (a : ℝ) (z : Fin N → ℝ) : average (fun j => a * z j) = a * average z := by
  simp only [average, smul_eq_mul, ← Finset.mul_sum]; ring

theorem average_smul (a : ℝ) (z : Fin N → Space m n) :
    average (fun j => a • z j) = a • average z := by
  simp only [average, ← Finset.smul_sum, smul_smul]; congr 1; ring

theorem integral_average {Ω : Type*} [MeasurableSpace Ω] {p : Measure Ω}
    {a : Fin N → Ω → ℝ} (ha : ∀ j, Integrable (a j) p) :
    (∫ ω, average (fun j => a j ω) ∂p) = average (fun j => ∫ ω, a j ω ∂p) := by
  simp only [average, smul_eq_mul]
  rw [integral_const_mul, integral_finsetSum _ (fun j _ => ha j)]

variable {Ω : Type*} [MeasurableSpace Ω] {p : Measure Ω} [IsProbabilityMeasure p]

/-- Frozen-query assumptions. No finiteness or bounded-support assumption on the noise law. -/
structure Noise (N m n : ℕ) (p : Measure Ω) (query : Fin N → Space m n) (σ : ℝ) where
  value : Fin N → Ω → Space m n
  memLp : ∀ j, MemLp (value j) 2 p
  centered : ∀ j, ∫ ω, value j ω ∂p = 0
  variance : ∀ j, (∫ ω, ‖value j ω‖ ^ 2 ∂p) ≤ σ ^ 2
  independent : ProbabilityTheory.iIndepFun value p

def Noise.replies {q : Fin N → Space m n} {σ : ℝ} (O : Noise N m n p q σ) (ω : Ω) :=
  fun j => q j + O.value j ω

omit [IsProbabilityMeasure p] in
theorem averaged_momentum_update (c : Parameters) (CU CP : Space m n → Space m n)
    (s : LayerState N m n) (q : Fin N → Space m n) {σ : ℝ}
    (O : Noise N m n p q σ) (ω : Ω) :
    average (update c CU CP s (O.replies ω)).M =
      newMomentum c.β (average s.M) (average q) (fun ω => average (fun j => O.value j ω)) ω := by
  change average (fun j => (1 - c.β) • s.M j + c.β • (q j + O.value j ω)) = _
  rw [average_add, average_smul, average_smul, average_add]
  rfl

theorem global_recurrence (c : Parameters) (hN : 0 < N) (CU CP : Space m n → Space m n)
    (s : LayerState N m n) (old new : Space m n) (q : Fin N → Space m n) {σ : ℝ}
    (O : Noise N m n p q σ) {a b : ℝ}
    (ha : ‖old - new‖ ^ 2 ≤ a) (hb : ‖average q - new‖ ^ 2 ≤ b) :
    (∫ ω, globalError (update c CU CP s (O.replies ω)) new ∂p) ≤
      (1 - c.β / 2) * globalError s old + 2 / c.β * a +
        c.β ^ 2 * (1 + 2 / c.β) * b + c.β ^ 2 * σ ^ 2 / N := by
  have hz : (∫ ω, average (fun j => O.value j ω) ∂p) = 0 := by
    simp only [average, integral_smul, integral_finsetSum _ (fun j _ => (O.memLp j).integrable
      (by norm_num)), O.centered, Finset.sum_const_zero, smul_zero]
  have hv := independent_average_variance hN O.memLp O.independent O.centered O.variance
  have h := momentum_recurrence c.β_pos c.β_le_one (average s.M) old new (average q)
    (average_memLp O.memLp) hz hv
  simp only [globalError, averaged_momentum_update]
  have hβ := c.β_pos
  have h1 := mul_le_mul_of_nonneg_left ha (by positivity : 0 ≤ 2 / c.β)
  have h2 := mul_le_mul_of_nonneg_left hb (by positivity : 0 ≤ c.β ^ 2 * (1 + 2 / c.β))
  simp only [div_eq_mul_inv] at h h1 h2 ⊢
  nlinarith only [h, h1, h2]

theorem local_recurrence (c : Parameters) (hN : 0 < N) (CU CP : Space m n → Space m n)
    (s : LayerState N m n) (old new q : Fin N → Space m n) {σ : ℝ}
    (O : Noise N m n p q σ) {a b : ℝ}
    (ha : average (fun j => ‖old j - new j‖ ^ 2) ≤ a)
    (hb : average (fun j => ‖q j - new j‖ ^ 2) ≤ b) :
    (∫ ω, localError (update c CU CP s (O.replies ω)) new ∂p) ≤
      (1 - c.β / 2) * localError s old + 2 / c.β * a +
        c.β ^ 2 * (1 + 2 / c.β) * b + c.β ^ 2 * σ ^ 2 := by
  have hi j := square_integrable ((newMomentum_memLp c.β (s.M j) (q j) (O.memLp j)).sub
    (memLp_const (new j)))
  simp only [Pi.sub_apply] at hi
  change (∫ ω, average (fun j => ‖newMomentum c.β (s.M j) (q j) (O.value j) ω - new j‖ ^ 2) ∂p) ≤ _
  rw [integral_average hi]
  have h := average_mono (fun j => momentum_recurrence c.β_pos c.β_le_one
    (s.M j) (old j) (new j) (q j) (O.memLp j) (O.centered j) (O.variance j))
  simp only [average_add, average_mul, average_const hN] at h
  have hβ := c.β_pos
  have h1 := mul_le_mul_of_nonneg_left ha (by positivity : 0 ≤ 2 / c.β)
  have h2 := mul_le_mul_of_nonneg_left hb (by positivity : 0 ≤ c.β ^ 2 * (1 + 2 / c.β))
  dsimp only [localError]
  linarith only [h, h1, h2]

theorem compression_recurrence (c : Parameters) (hN : 0 < N)
    {CU : Space m n → Space m n} (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (CP : Space m n → Space m n) (s : LayerState N m n)
    (old new q : Fin N → Space m n) {σ : ℝ} (O : Noise N m n p q σ) {a b : ℝ}
    (ha : average (fun j => ‖old j - new j‖ ^ 2) ≤ a)
    (hb : average (fun j => ‖q j - new j‖ ^ 2) ≤ b) :
    (∫ ω, compressionError (update c CU CP s (O.replies ω)) ∂p) ≤
      (1 - c.αU / 2) * compressionError s +
        6 * c.β ^ 2 / c.αU * (localError s old + a + b) + (1 - c.αU) * c.β ^ 2 * σ ^ 2 := by
  have hi j := square_integrable (residual_memLp hm c.αU_pos hc
    ((newMomentum_memLp c.β (s.M j) (q j) (O.memLp j)).sub (memLp_const (s.G j))))
  simp only [Pi.sub_apply] at hi
  have he (ω : Ω) (j : Fin N) :
      ‖(update c CU CP s (O.replies ω)).M j - (update c CU CP s (O.replies ω)).G j‖ ^ 2 =
        ‖CU (newMomentum c.β (s.M j) (q j) (O.value j) ω - s.G j) -
          (newMomentum c.β (s.M j) (q j) (O.value j) ω - s.G j)‖ ^ 2 := by
    rw [norm_sub_rev]
    congr 2
    dsimp [update, Noise.replies, newMomentum, LayerState.M, LayerState.G]; module
  simp only [compressionError, he]
  rw [integral_average hi]
  have h := average_mono (fun j => uplink_recurrence (β := c.β) c.αU_pos c.αU_le_one
    (s.M j) (s.G j) (old j) (new j) (q j) hm hc (O.memLp j) (O.centered j) (O.variance j))
  simp only [average_add, average_mul, average_const hN] at h
  have hU := c.αU_pos
  have hh := mul_le_mul_of_nonneg_left (add_le_add ha hb)
    (by positivity : 0 ≤ 6 * c.β ^ 2 / c.αU)
  dsimp only [localError]
  linarith only [h, hh]

end EF21Muon.Distributed
