import EF21Muon.Distributed.Energy
import EF21Muon.Probability.Paths

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]

/-- Matrix geometry and compression assumptions; instantiated for both methods below. -/
structure Scheme (N : ℕ) (m n : Fin p → ℕ) (c : Fin p → Parameters) where
  CU : ∀ i, Space (m i) (n i) → Space (m i) (n i)
  CP : ∀ i, Space (m i) (n i) → Space (m i) (n i)
  measurable_CU : ∀ i, Measurable (CU i)
  measurable_CP : ∀ i, Measurable (CP i)
  contracts_CU : ∀ i, SquaredContracts (CU i) (c i).αU
  τ : Fin p → ℝ
  τ_nonneg : ∀ i, 0 ≤ τ i
  broadcast : ∀ i, BroadcastBound (c i) (CP i) (τ i)
  dual : ∀ i (z : Space (m i) (n i)), nuclearNorm z ≤ (c i).ρ * ‖z‖

/-- Paper assumptions for arbitrary noise laws. Independence is only across workers
at a fixed query/layer; different layers may have correlated noise. -/
structure Problem (N : ℕ) (m n : Fin p → ℕ) (μ : Measure Ω) (c : Fin p → Parameters) where
  workers_pos : 0 < N
  f : Model m n → ℝ
  grad : Model m n → Model m n
  localGrad : Fin N → Model m n → Model m n
  localL : Fin p → Fin N → ℝ
  localL_nonneg : ∀ i j, 0 ≤ localL i j
  fmin : ℝ
  lower_bound : ∀ x, fmin ≤ f x
  derivative : ∀ x, HasFDerivAt f (gradientMap (grad x)) x
  smooth : ∀ x y i, nuclearNorm (grad x i - grad y i) ≤
    (c i).L * spectralNorm (x i - y i)
  localSmooth : ∀ j x y i, nuclearNorm (localGrad j x i - localGrad j y i) ≤
    localL i j * spectralNorm (x i - y i)
  mean_gradient : ∀ x i, average (fun j => localGrad j x i) = grad x i
  σ : Fin p → ℝ
  reply : Model m n → Ω → Fin N → Model m n
  measurable_reply : Measurable (Function.uncurry reply)
  noise_memLp : ∀ x i j, MemLp (fun ω => reply x ω j i - localGrad j x i) 2 μ
  noise_centered : ∀ x i j, (∫ ω, reply x ω j i - localGrad j x i ∂μ) = 0
  noise_variance : ∀ x i j, (∫ ω, ‖reply x ω j i - localGrad j x i‖ ^ 2 ∂μ) ≤ σ i ^ 2
  worker_independence : ∀ x i, ProbabilityTheory.iIndepFun
    (fun j ω => reply x ω j i - localGrad j x i) μ

def points (s : States N m n) : Model m n := fun i => (s i).x
def queries {c : Fin p → Parameters} (A : Scheme N m n c) (s : States N m n) : Model m n :=
  fun i => nextW (c i) (A.CP i) (s i)
def serverPoints (c : Fin p → Parameters) (s : States N m n) : Model m n :=
  fun i => nextX (c i) (s i)

def next {c : Fin p → Parameters} (A : Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n) (s : States N m n) (ω : Ω) : States N m n :=
  fun i => update (c i) (A.CU i) (A.CP i) (s i) (fun j => reply (queries A s) ω j i)

def Problem.ell2 {μ : Measure Ω} {c : Fin p → Parameters} (P : Problem N m n μ c) (i : Fin p) : ℝ :=
  average (fun j => P.localL i j ^ 2)

theorem Problem.ell2_nonneg {μ : Measure Ω} {c : Fin p → Parameters}
    (P : Problem N m n μ c) (i : Fin p) : 0 ≤ P.ell2 i := average_nonneg (fun _ => sq_nonneg _)

def Problem.frozenNoise {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}
    (P : Problem N m n μ c) (A : Scheme N m n c) (s : States N m n) (i : Fin p) :
    Noise N (m i) (n i) μ (fun j => P.localGrad j (queries A s) i) (P.σ i) where
  value j ω := P.reply (queries A s) ω j i - P.localGrad j (queries A s) i
  memLp := P.noise_memLp (queries A s) i
  centered := P.noise_centered (queries A s) i
  variance := P.noise_variance (queries A s) i
  independent := P.worker_independence (queries A s) i

theorem next_frozen {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}
    (P : Problem N m n μ c) (A : Scheme N m n c) (s : States N m n) (ω : Ω) (i : Fin p) :
    next A P.reply s ω i = update (c i) (A.CU i) (A.CP i) (s i)
      ((P.frozenNoise A s i).replies ω) := by
  have he : (P.frozenNoise A s i).replies ω = (fun j => P.reply (queries A s) ω j i) := by
    funext j
    dsimp [Noise.replies, Problem.frozenNoise]
    module
  rw [he]
  rfl

omit [MeasurableSpace Ω] in
theorem points_next {c : Fin p → Parameters} (A : Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n) (s : States N m n) (ω : Ω) :
    points (next A reply s ω) = serverPoints c s := rfl

theorem measurable_average {B E : Type*} [MeasurableSpace B]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [MeasurableSpace E] [BorelSpace E]
    [SecondCountableTopology E] {z : B → Fin N → E} (hz : ∀ j, Measurable (fun b => z b j)) :
    Measurable (fun b => average (z b)) :=
  (Finset.measurable_sum _ (fun j _ => hz j)).const_smul _

theorem measurable_points : Measurable (points : States N m n → Model m n) :=
  Measurable.of_eval (fun i => (measurable_pi_apply i).fst)

theorem measurable_serverPoints (c : Fin p → Parameters) :
    Measurable (serverPoints c : States N m n → Model m n) := by
  apply Measurable.of_eval
  intro i
  have hx : Measurable (fun s : States N m n => (s i).x) := (measurable_pi_apply i).fst
  have hG : Measurable (fun s : States N m n => average (s i).G) :=
    measurable_average (fun j => (measurable_pi_apply j).comp (measurable_pi_apply i).snd.snd.snd)
  exact hx.sub ((measurable_const.mul (continuous_nuclearNorm.measurable.comp hG)).smul
    ((measurable_polar (m i) (n i)).comp hG))

theorem measurable_queries {c : Fin p → Parameters} (A : Scheme N m n c) :
    Measurable (queries A : States N m n → Model m n) := by
  apply Measurable.of_eval
  intro i
  have hx : Measurable (fun s : States N m n => serverPoints c s i) :=
    (measurable_pi_apply i).comp (measurable_serverPoints c (N := N) (m := m) (n := n))
  have hw : Measurable (fun s : States N m n => (s i).w) := (measurable_pi_apply i).snd.fst
  exact hw.add ((A.measurable_CP i).comp (hx.sub hw))

theorem measurable_next {c : Fin p → Parameters} (A : Scheme N m n c)
    {reply : Model m n → Ω → Fin N → Model m n} (hr : Measurable (Function.uncurry reply)) :
    Measurable (Function.uncurry (next A reply)) := by
  have hreply : Measurable (fun z : States N m n × Ω => reply (queries A z.1) z.2) :=
    hr.comp (((measurable_queries A).comp measurable_fst).prodMk measurable_snd)
  apply Measurable.of_eval
  intro i
  have hx : Measurable (fun z : States N m n × Ω => serverPoints c z.1 i) :=
    ((measurable_pi_apply i).comp (measurable_serverPoints c (N := N) (m := m) (n := n))).comp measurable_fst
  have hw : Measurable (fun z : States N m n × Ω => queries A z.1 i) :=
    ((measurable_pi_apply i).comp (measurable_queries A)).comp measurable_fst
  have hM j : Measurable (fun z : States N m n × Ω =>
      (1 - (c i).β) • (z.1 i).M j + (c i).β • reply (queries A z.1) z.2 j i) := by
    have hold : Measurable (fun z : States N m n × Ω => (z.1 i).M j) :=
      (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst).snd.snd.fst
    have hrep : Measurable (fun z : States N m n × Ω => reply (queries A z.1) z.2 j i) :=
      (measurable_pi_apply i).comp ((measurable_pi_apply j).comp hreply)
    simpa only [Pi.smul_def, Pi.add_def] using (hold.const_smul (1 - (c i).β)).add
      (hrep.const_smul (c i).β)
  have hG j : Measurable (fun z : States N m n × Ω => (z.1 i).G j) :=
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst).snd.snd.snd
  exact hx.prodMk (hw.prodMk ((Measurable.of_eval hM).prodMk (Measurable.of_eval
    (fun j => (hG j).add ((A.measurable_CU i).comp ((hM j).sub (hG j)))))))

end EF21Muon.Distributed
