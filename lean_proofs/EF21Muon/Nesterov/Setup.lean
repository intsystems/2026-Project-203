import EF21Muon.Nesterov.SmoothEnergy
import EF21Muon.Distributed.Convergence

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]

def core (s : States N m n) : Distributed.States N m n := fun i => (s i).1
def points (s : States N m n) : Model m n := Distributed.points (core s)
def serverPoints (c : Fin p → Parameters) (s : States N m n) : Model m n :=
  Distributed.serverPoints c (core s)
def queries {c : Fin p → Parameters} (A : Scheme N m n c) (s : States N m n) : Model m n :=
  Distributed.queries A (core s)
def initial (x : Model m n) : States N m n := fun i => start (x i)

def next {c : Fin p → Parameters} (A : Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n) (s : States N m n) (ω : Ω) : States N m n :=
  fun i => update (c i) (A.CU i) (A.CP i) (s i) (fun j => reply (queries A s) ω j i)

theorem points_next {c : Fin p → Parameters} (A : Scheme N m n c)
    (reply : Model m n → Ω → Fin N → Model m n) (s : States N m n) (ω : Ω) :
    points (next A reply s ω) = serverPoints c s := rfl

def frozenNoise {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}
    (P : Problem N m n μ c) (A : Scheme N m n c) (s : States N m n) (i : Fin p) :=
  P.frozenNoise A (core s) i

theorem next_frozen {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}
    (P : Problem N m n μ c) (A : Scheme N m n c) (s : States N m n) (ω : Ω) (i : Fin p) :
    next A P.reply s ω i = update (c i) (A.CU i) (A.CP i) (s i)
      ((frozenNoise P A s i).replies ω) := by
  have he : (frozenNoise P A s i).replies ω = (fun j => P.reply (queries A s) ω j i) := by
    funext j
    dsimp [frozenNoise, Noise.replies, Problem.frozenNoise, queries]
    module
  rw [he]; rfl

theorem measurable_core : Measurable (core : States N m n → Distributed.States N m n) :=
  Measurable.of_eval (fun i => (measurable_pi_apply i).fst)

theorem measurable_points : Measurable (points : States N m n → Model m n) :=
  Distributed.measurable_points.comp measurable_core

theorem measurable_serverPoints (c : Fin p → Parameters) :
    Measurable (serverPoints c : States N m n → Model m n) :=
  (Distributed.measurable_serverPoints c).comp measurable_core

theorem measurable_queries {c : Fin p → Parameters} (A : Scheme N m n c) :
    Measurable (queries A : States N m n → Model m n) :=
  (Distributed.measurable_queries A).comp measurable_core

theorem measurable_next {c : Fin p → Parameters} (A : Scheme N m n c)
    {reply : Model m n → Ω → Fin N → Model m n} (hr : Measurable (Function.uncurry reply)) :
    Measurable (Function.uncurry (next A reply)) := by
  have hreply : Measurable (fun z : States N m n × Ω => reply (queries A z.1) z.2) :=
    hr.comp (((measurable_queries A).comp measurable_fst).prodMk measurable_snd)
  apply Measurable.of_eval
  intro i
  have hx : Measurable (fun z : States N m n × Ω => serverPoints c z.1 i) :=
    ((measurable_pi_apply i).comp (measurable_serverPoints c)).comp measurable_fst
  have hw : Measurable (fun z : States N m n × Ω => queries A z.1 i) :=
    ((measurable_pi_apply i).comp (measurable_queries A)).comp measurable_fst
  have hold j : Measurable (fun z : States N m n × Ω => (z.1 i).1.M j) :=
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst).fst.snd.snd.fst
  have hrep j : Measurable (fun z : States N m n × Ω => reply (queries A z.1) z.2 j i) :=
    (measurable_pi_apply i).comp ((measurable_pi_apply j).comp hreply)
  have hM j : Measurable (fun z : States N m n × Ω =>
      (1-(c i).β) • (z.1 i).1.M j+(c i).β • reply (queries A z.1) z.2 j i) := by
    simpa only [Pi.smul_def, Pi.add_def] using
      ((hold j).const_smul (1-(c i).β)).add ((hrep j).const_smul (c i).β)
  have hV j : Measurable (fun z : States N m n × Ω =>
      (1-(c i).β) • ((1-(c i).β) • (z.1 i).1.M j+(c i).β • reply (queries A z.1) z.2 j i)+
        (c i).β • reply (queries A z.1) z.2 j i) := by
    simpa only [Pi.smul_def, Pi.add_def] using
      ((hM j).const_smul (1-(c i).β)).add ((hrep j).const_smul (c i).β)
  have hG j : Measurable (fun z : States N m n × Ω => (z.1 i).1.G j) :=
    (measurable_pi_apply j).comp ((measurable_pi_apply i).comp measurable_fst).fst.snd.snd.snd
  exact (hx.prodMk (hw.prodMk ((Measurable.of_eval hM).prodMk (Measurable.of_eval
    (fun j => (hG j).add ((A.measurable_CU i).comp ((hV j).sub (hG j)))))))).prodMk
      (Measurable.of_eval hold)

set_option maxHeartbeats 1200000 in
theorem measurable_energy {B : Type*} [MeasurableSpace B] {i : Fin p}
    {s : B → Layer N (m i) (n i)} {localG : B → Fin N → Space (m i) (n i)}
    (hs : Measurable s) (hlocal : ∀ j, Measurable (fun b => localG b j))
    (c : Fin p → Parameters) (ell2 : ℝ) :
    Measurable (fun b => energy (c i) ell2 (s b) (localG b)) := by
  have hM j : Measurable (fun b => (s b).1.M j) :=
    (measurable_pi_apply j).comp hs.fst.snd.snd.fst
  have hG j : Measurable (fun b => (s b).1.G j) :=
    (measurable_pi_apply j).comp hs.fst.snd.snd.snd
  have hprev j : Measurable (fun b => (s b).2 j) := (measurable_pi_apply j).comp hs.snd
  have hV j : Measurable (fun b => target (c i) (s b) j) := by
    simpa only [target, Pi.smul_def, Pi.sub_def] using
      ((hM j).const_smul (2-(c i).β)).sub ((hprev j).const_smul (1-(c i).β))
  have hF := measurable_average (fun j => ((hM j).sub (hlocal j)).norm.pow_const 2)
  have hH := measurable_average (fun j => ((hV j).sub (hG j)).norm.pow_const 2)
  have hJ := measurable_average (fun j => ((hM j).sub (hprev j)).norm.pow_const 2)
  have hP := (hs.fst.snd.fst.sub hs.fst.fst).norm.pow_const 2
  exact ((((hF.const_mul _).add (hH.const_mul _)).add (hJ.const_mul _)).add
    (hP.const_mul _)).const_mul _

end EF21Muon.Nesterov
