import EF21Muon.Distributed.Methods

/-! Proposition 2: rotate the round boundary between the paper's
gradient/uplink -> step/downlink loop and EF21-Muon's
step/downlink -> gradient/uplink loop. The statement is pathwise, for arbitrary
numbers of workers and heterogeneous layers, schedules, compressors and gradient
replies. A fixed realization of stochastic replies or compressor ties is allowed.
The server step is abstract, so the same theorem covers any fixed LMO selection. -/

noncomputable section

namespace EF21Muon.Distributed.Correspondence

open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar

variable {N p : ℕ} {m n : Fin p → ℕ}

structure Operations (N : ℕ) (m n : Fin p → ℕ) where
  μ : Fin p → ℝ
  uplink : ℕ → ∀ i, Fin N → Space (m i) (n i) → Space (m i) (n i)
  downlink : ℕ → ∀ i, Space (m i) (n i) → Space (m i) (n i)
  reply : ℕ → Model m n → Fin N → Model m n
  step : ℕ → Model m n → Model m n → Model m n
  step_zero : ∀ t x, step t x 0 = x
  downlink_zero : ∀ t i, downlink t i 0 = 0

/-- The paper's client computation, with EMA momentum mu and beta = 1-mu. -/
def workerPhase (O : Operations N m n) (t : ℕ) (s : States N m n) : States N m n :=
  let W : Model m n := fun i => (s i).w
  fun i =>
    let M := fun j => O.μ i • (s i).M j + (1 - O.μ i) • O.reply t W j i
    ((s i).x, (s i).w, M, fun j => (s i).G j + O.uplink t i j (M j - (s i).G j))

/-- The server LMO step followed by the common downlink refresh. -/
def serverPhase (O : Operations N m n) (t : ℕ) (s : States N m n) : States N m n :=
  let X := O.step t (points s) (fun i => average (s i).G)
  fun i => (X i, (s i).w + O.downlink t i (X i - (s i).w), (s i).M, (s i).G)

def paperRun (O : Operations N m n) (x : Model m n) : ℕ → States N m n
  | 0 => start x
  | t + 1 => serverPhase O t (workerPhase O t (paperRun O x t))

/-- The first step is vacuous. At subsequent upstream rounds the step/downlink
uses the preceding paper round's radius and compressor realization. -/
def frameworkRun (O : Operations N m n) (x : Model m n) : ℕ → States N m n
  | 0 => start x
  | t + 1 => workerPhase O t (serverPhase O (t - 1) (frameworkRun O x t))

theorem serverPhase_start (O : Operations N m n) (t : ℕ) (x : Model m n) :
    serverPhase O t (start x) = start x := by
  have hz : (fun i => average ((start x : States N m n) i).G) = 0 := by
    funext i
    simp [start, LayerState.start, LayerState.G, average]
  simp only [serverPhase, hz, show points (start x : States N m n) = x from rfl,
    O.step_zero]
  funext i
  simp [start, LayerState.start, LayerState.w, LayerState.M,
    LayerState.G, O.downlink_zero]

/-- Full states agree after the client phase; the framework is one client phase
ahead at this round boundary. No smoothness or probabilistic assumptions enter. -/
theorem exact_instance (O : Operations N m n) (x : Model m n) (t : ℕ) :
    frameworkRun O x (t + 1) = workerPhase O t (paperRun O x t) := by
  induction t with
  | zero => simp only [frameworkRun, paperRun, Nat.zero_sub, serverPhase_start]
  | succ t ih =>
    rw [frameworkRun, Nat.add_sub_cancel, ih, paperRun]

/-- In the notation of Proposition 2, X^(t+1) = X_t and W^(t+1) = W_t. -/
theorem models_exact (O : Operations N m n) (x : Model m n) (t : ℕ) :
    points (frameworkRun O x (t + 1)) = points (paperRun O x t) ∧
    (fun i => (frameworkRun O x (t + 1) i).w) =
      (fun i => (paperRun O x t i).w) := by
  rw [exact_instance]
  exact ⟨rfl, rfl⟩

/-- The framework average contains one extra copy of the initial model. -/
theorem observable_sum (O : Operations N m n) (x : Model m n)
    (F : Model m n → ℝ) (T : ℕ) :
    (∑ t ∈ Finset.range (T + 1), F (points (frameworkRun O x t))) =
      F x + ∑ t ∈ Finset.range T, F (points (paperRun O x t)) := by
  rw [Finset.sum_range_succ']
  simp only [(models_exact O x _).1]
  change (∑ t ∈ Finset.range T, F (points (paperRun O x t))) + F x = _
  ring

/-- Transfer a nonnegative stationarity bound across the one-step index shift. -/
theorem average_transfer (O : Operations N m n) (x : Model m n)
    (F : Model m n → ℝ) (hF : 0 ≤ F x) (T : ℕ) (hT : 0 < T) :
    (∑ t ∈ Finset.range T, F (points (paperRun O x t))) / T ≤
      ((T + 1 : ℕ) : ℝ) / T *
        ((∑ t ∈ Finset.range (T + 1), F (points (frameworkRun O x t))) / (T + 1)) := by
  have hTr : (0 : ℝ) < T := by exact_mod_cast hT
  rw [observable_sum]
  have hp : (T : ℝ) + 1 ≠ 0 := by positivity
  simp only [Nat.cast_add, Nat.cast_one]
  field_simp
  nlinarith

theorem server_state_exact (O : Operations N m n) (x : Model m n) (t : ℕ) :
    serverPhase O t (frameworkRun O x (t + 1)) = paperRun O x (t + 1) := by
  rw [exact_instance]
  rfl

/-- The server's incremental reconstruction is exactly the mean of the updated
worker estimators. This also verifies the aggregate state omitted from the tuple. -/
theorem aggregate_update (O : Operations N m n) (t : ℕ) (s : States N m n) (i : Fin p) :
    average (workerPhase O t s i).G = average (s i).G +
      average (fun j => O.uplink t i j
        (O.μ i • (s i).M j + (1 - O.μ i) • O.reply t (fun k => (s k).w) j i - (s i).G j)) := by
  exact average_add _ _

/-- A scheduled spectral-ball LMO with the convention that a zero estimate takes
the zero step, as required in Proposition 2. -/
def polarStep (η : ℕ → Fin p → ℝ) (t : ℕ) (x g : Model m n) : Model m n :=
  fun i => x i - η t i • polar (g i)

theorem polarStep_zero (η : ℕ → Fin p → ℝ) (t : ℕ) (x : Model m n) :
    polarStep η t x 0 = x := by
  funext i
  simp [polarStep, polar_zero]

/-- Concrete operations for the already formalized stochastic EF21 scheme.
Fixing a noise sequence makes this a pathwise statement, with no distributional
restrictions. The identity and scaled-sign downlinks both preserve zero. -/
def ofScheme {c : Fin p → Parameters} (A : Scheme N m n c)
    (hCP : ∀ i, A.CP i 0 = 0) {Ω : Type*}
    (reply : Model m n → Ω → Fin N → Model m n) (ω : ℕ → Ω) : Operations N m n where
  μ i := 1 - (c i).β
  uplink _ i _ := A.CU i
  downlink _ i := A.CP i
  reply t x := reply x (ω t)
  step _ x g := fun i => x i - ((c i).γ * nuclearNorm (g i)) • polar (g i)
  step_zero _ x := by funext i; simp [polar_zero]
  downlink_zero _ := hCP

/-- The round used by the convergence proof is exactly the framework round
instantiated above, including every worker's momentum and estimator. -/
theorem ofScheme_round {c : Fin p → Parameters} (A : Scheme N m n c)
    (hCP : ∀ i, A.CP i 0 = 0) {Ω : Type*}
    (reply : Model m n → Ω → Fin N → Model m n) (ω : ℕ → Ω)
    (s : States N m n) (t k : ℕ) :
    workerPhase (ofScheme A hCP reply ω) t
      (serverPhase (ofScheme A hCP reply ω) k s) = next A reply s (ω t) := by
  have hb (i : Fin p) : 1 - (1 - (c i).β) = (c i).β := by ring
  funext i
  simp only [workerPhase, serverPhase, ofScheme, hb, next, update, nextW, nextX,
    LayerState.x, LayerState.w, LayerState.M, LayerState.G, LayerState.server,
    points, serverPoint, queryPoint]
  rfl

theorem matrix_downlink_zero (compressed : Bool) {β γ L : Fin p → ℝ}
    (hm : ∀ i, 0 < m i) (hn : ∀ i, 0 < n i) (hβ : ∀ i, 0 < β i)
    (hβ1 : ∀ i, β i ≤ 1) (hγ : ∀ i, 0 < γ i) (hL : ∀ i, 0 ≤ L i) (i : Fin p) :
    (matrixScheme (N := N) compressed β γ L hm hn hβ hβ1 hγ hL).CP i 0 = 0 := by
  cases compressed with
  | false => rfl
  | true =>
    change signCompressor (0 : Space (m i) (n i)) = 0
    ext a
    simp [signCompressor, Compression.entrywiseL1, Compression.scaledSign,
      Compression.scaledSignWith]

end EF21Muon.Distributed.Correspondence
