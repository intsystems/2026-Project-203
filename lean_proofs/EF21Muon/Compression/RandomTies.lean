import EF21Muon.Distributed.Methods
import Mathlib.Probability.Distributions.Uniform
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.Independence.InfinitePi

/-! Fresh, independent binary ties. The law samples a bit for each worker,
layer and entry, and a separate bit for every downlink entry. Bits at nonzero
entries are ignored. Compression error norms are independent of those bits. -/

noncomputable section

namespace EF21Muon.RandomTies

open MeasureTheory ProbabilityTheory
open LongTails.Algorithms.MatrixPolar
open EF21Muon.Compression EF21Muon.Distributed EF21Muon.Stochastic

def tieSign (b : Bool) (x : ℝ) : ℝ :=
  if x = 0 then (if b then 1 else -1) else signBit x

theorem compatible_tieSign {ι : Type*} (b : ι → Bool) (x : ι → ℝ) :
    CompatibleSigns x (fun i => tieSign (b i) (x i)) := by
  constructor
  · intro i
    by_cases h : x i = 0
    · cases b i <;> simp [tieSign, h]
    · simpa [tieSign, h] using (compatible_signBit x).1 i
  · intro i
    by_cases h : x i = 0
    · simp [h]
    · simpa [tieSign, h] using (compatible_signBit x).2 i

def compressor {m n : ℕ} (b : Fin m × Fin n → Bool) (x : Space m n) : Space m n :=
  WithLp.toLp 2 (scaledSignWith (fun ij => x ij) (fun ij => tieSign (b ij) (x ij)))

theorem measurable_compressor {m n : ℕ} (b : Fin m × Fin n → Bool) :
    Measurable (compressor b) := by
  apply (PiLp.continuous_toLp 2 (fun _ : Fin m × Fin n => ℝ)).measurable.comp
  apply Measurable.of_eval
  intro ij
  have hx (k : Fin m × Fin n) : Measurable (fun x : Space m n => x k) :=
    (PiLp.continuous_apply 2 (fun _ : Fin m × Fin n => ℝ) k).measurable
  have ht : Measurable (fun x : Space m n => tieSign (b ij) (x ij)) := by
    unfold tieSign
    apply Measurable.ite (hx ij (measurableSet_singleton 0))
      measurable_const
    unfold signBit
    exact measurable_const.ite (measurableSet_le measurable_const (hx ij)) measurable_const
  have hl : Measurable (fun x : Space m n => entrywiseL1 (fun k => x k)) :=
    Finset.measurable_sum _ (fun k _ => by simpa only [Real.norm_eq_abs] using (hx k).norm)
  exact (hl.div_const _).mul ht

theorem compressor_error_sq {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (b : Fin m × Fin n → Bool) (x : Space m n) :
    ‖compressor b x-x‖^2 = ‖signCompressor x-x‖^2 := by
  let : NeZero m := ⟨hm.ne'⟩
  let : NeZero n := ⟨hn.ne'⟩
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
  exact (scaledSign_error_identity (fun ij => x ij) _ (compatible_tieSign b _)).trans
    (scaledSign_error_identity (fun ij => x ij) _ (compatible_signBit _)).symm

theorem compressor_error_norm {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (b : Fin m × Fin n → Bool) (x : Space m n) :
    ‖compressor b x-x‖ = ‖signCompressor x-x‖ :=
  (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp (compressor_error_sq hm hn b x)

theorem compressor_contracts {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (b : Fin m × Fin n → Bool) : SquaredContracts (compressor b) (1/(m*n : ℕ)) := by
  intro x
  rw [compressor_error_sq hm hn]
  exact sign_squared_contracts hm hn x

@[simp] theorem compressor_zero {m n : ℕ} (b : Fin m × Fin n → Bool) :
    compressor b (0 : Space m n) = 0 := by
  ext ij
  simp [compressor, scaledSignWith, entrywiseL1]

abbrev Index (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) :=
  Fin (N+1) × (Σ i : Fin p, Fin (m i) × Fin (n i))

abbrev Coins (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) := Index N m n → Bool

def coinLaw (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) : Measure (Coins N m n) :=
  Measure.pi (fun _ => (PMF.uniformOfFintype Bool).toMeasure)

instance coinLaw_probability (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) :
    IsProbabilityMeasure (coinLaw N m n) := by unfold coinLaw; infer_instance

theorem coin_independence (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) :
    iIndepFun (fun k (b : Coins N m n) => b k) (coinLaw N m n) :=
  iIndepFun_pi (X := fun _ => id) (fun _ => aemeasurable_id)

theorem coin_fair (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) (k : Index N m n) (v : Bool) :
    coinLaw N m n {b | b k = v} = 1/2 := by
  have h := (measurePreserving_eval
    (fun _ : Index N m n => (PMF.uniformOfFintype Bool).toMeasure) k).measure_preimage
      (measurableSet_singleton v).nullMeasurableSet
  change coinLaw N m n {b | b k = v} = _ at h
  rw [h, PMF.toMeasure_uniformOfFintype_apply {v} (measurableSet_singleton v)]
  simp

/-- All rounds, workers, links, layers and entries have their own bit. -/
abbrev Stream (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) := (ℕ × Index N m n) → Bool

def streamLaw (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) : Measure (Stream N m n) :=
  Measure.infinitePi (fun _ => (PMF.uniformOfFintype Bool).toMeasure)

instance streamLaw_probability (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) :
    IsProbabilityMeasure (streamLaw N m n) := by unfold streamLaw; infer_instance

theorem stream_independence (N : ℕ) {p : ℕ} (m n : Fin p → ℕ) :
    iIndepFun (fun k (b : Stream N m n) => b k) (streamLaw N m n) :=
  iIndepFun_infinitePi (X := fun _ => id) (fun _ => measurable_id)

theorem stream_fair (N : ℕ) {p : ℕ} (m n : Fin p → ℕ)
    (t : ℕ) (k : Index N m n) (v : Bool) : streamLaw N m n {b | b (t,k) = v} = 1/2 := by
  have h := (measurePreserving_eval_infinitePi
    (fun _ : ℕ × Index N m n => (PMF.uniformOfFintype Bool).toMeasure) (t,k)).measure_preimage
      (measurableSet_singleton v).nullMeasurableSet
  change streamLaw N m n {b | b (t,k) = v} = _ at h
  rw [h, PMF.toMeasure_uniformOfFintype_apply {v} (measurableSet_singleton v)]
  simp

def roundBits {N p : ℕ} {m n : Fin p → ℕ} (b : Stream N m n) (t : ℕ) : Coins N m n :=
  fun k => b (t,k)

theorem measurable_roundBits {N p : ℕ} {m n : Fin p → ℕ} (t : ℕ) :
    Measurable (fun b : Stream N m n => roundBits b t) :=
  Measurable.of_eval (fun k => measurable_pi_apply _)

def downBits {N p : ℕ} {m n : Fin p → ℕ} (b : Coins N m n) (i : Fin p) :=
  fun ij => b (0, ⟨i, ij⟩)

def upBits {N p : ℕ} {m n : Fin p → ℕ} (b : Coins N m n) (i : Fin p) (j : Fin N) :=
  fun ij => b (j.succ, ⟨i, ij⟩)

theorem tieSign_eq_of_ne_zero (b : Bool) {x : ℝ} (hx : x ≠ 0) :
    tieSign b x = signBit x := by simp [tieSign, hx]

theorem compressor_eq_of_no_zeros {m n : ℕ} (b : Fin m × Fin n → Bool)
    (x : Space m n) (hx : ∀ ij, x ij ≠ 0) : compressor b x = signCompressor x := by
  ext ij
  simp [compressor, signCompressor, scaledSignWith, scaledSign, tieSign, hx]

theorem residual_norm {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (b : Fin m × Fin n → Bool) (v g : Space m n) :
    ‖v-(g+compressor b (v-g))‖ = ‖v-(g+signCompressor (v-g))‖ := by
  have he (C : Space m n → Space m n) :
      ‖v-(g+C (v-g))‖ = ‖C (v-g)-(v-g)‖ := by
    rw [norm_sub_rev]
    congr 1
    module
  rw [he, he]
  exact compressor_error_norm hm hn b (v-g)

/-- The same contraction constants and broadcast bound, with arbitrary downlink ties. -/
def matrixScheme (N : ℕ) {p : ℕ} {m n : Fin p → ℕ} (compressed : Bool)
    (β γ L : Fin p → ℝ) (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i)
    (hβ : ∀ i, 0 < β i) (hβ1 : ∀ i, β i ≤ 1) (hγ : ∀ i, 0 < γ i)
    (hL : ∀ i, 0 ≤ L i) (b : Coins N m n) :
    Scheme N m n (matrixControls compressed β γ L hm hn hβ hβ1 hγ hL) where
  CU _ := signCompressor
  CP i := if compressed then compressor (downBits b i) else id
  measurable_CU _ := measurable_signCompressor
  measurable_CP i := by
    cases compressed with
    | false => exact measurable_id
    | true => exact measurable_compressor _
  contracts_CU i := sign_squared_contracts (hm i) (hn i)
  τ i := if compressed then (Real.sqrt ((min (m i) (n i) : ℕ) : ℝ))^2 else 0
  τ_nonneg i := by cases compressed <;> simp only [Bool.false_eq_true, ↓reduceIte] <;> positivity
  broadcast i := by
    cases compressed with
    | false => exact identity_broadcast_bound _
    | true => exact downlink_rec _ (compressor_contracts (hm i) (hn i) _) polar_frobenius_le_sqrt_rank
  dual _ := nuclearNorm_le_sqrt_rank

end EF21Muon.RandomTies
