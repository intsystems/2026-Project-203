import EF21Muon.Distributed.Setup

noncomputable section

namespace EF21Muon.Distributed

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

variable {N p : ℕ} {m n : Fin p → ℕ} {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {c : Fin p → Parameters}

def Problem.potential (P : Problem N m n μ c) (s : States N m n) : ℝ :=
  P.f (points s) - P.fmin + ∑ i, energy (c i) (P.ell2 i) (s i)
    (P.grad (points s) i) (fun j => P.localGrad j (points s) i)

def Problem.cost (P : Problem N m n μ c) (s : States N m n) : ℝ :=
  ∑ i, (c i).γ / 4 * nuclearNorm (P.grad (points s) i) ^ 2

def Problem.noiseTotal (P : Problem N m n μ c) : ℝ :=
  ∑ i, (c i).γ * noise (c i) N * P.σ i ^ 2

omit [IsProbabilityMeasure μ] in
theorem Problem.potential_nonneg (P : Problem N m n μ c) (s : States N m n) :
    0 ≤ P.potential s :=
  add_nonneg (sub_nonneg.mpr (P.lower_bound _)) (Finset.sum_nonneg (fun i _ =>
    energy_nonneg (c i) (P.ell2_nonneg i) _ _ _))

omit [IsProbabilityMeasure μ] in
theorem Problem.cost_nonneg (P : Problem N m n μ c) (s : States N m n) : 0 ≤ P.cost s :=
  Finset.sum_nonneg (fun i _ => mul_nonneg (div_nonneg (c i).γ_pos.le (by norm_num)) (sq_nonneg _))

theorem measurable_energy {B : Type*} [MeasurableSpace B] {i : Fin p}
    {s : B → LayerState N (m i) (n i)} {g : B → Space (m i) (n i)}
    {localG : B → Fin N → Space (m i) (n i)}
    (hs : Measurable s) (hg : Measurable g) (hlocal : ∀ j, Measurable (fun b => localG b j))
    (ell2 : ℝ) : Measurable (fun b => energy (c i) ell2 (s b) (g b) (localG b)) := by
  have hM j : Measurable (fun b => (s b).M j) := (measurable_pi_apply j).comp hs.snd.snd.fst
  have hG j : Measurable (fun b => (s b).G j) := (measurable_pi_apply j).comp hs.snd.snd.snd
  have hE := ((measurable_average hM).sub hg).norm.pow_const 2
  have hF := measurable_average (fun j => ((hM j).sub (hlocal j)).norm.pow_const 2)
  have hH := measurable_average (fun j => ((hM j).sub (hG j)).norm.pow_const 2)
  have hP := (hs.snd.fst.sub hs.fst).norm.pow_const 2
  exact (((((hE.const_mul _).add (hF.const_mul _)).add (hH.const_mul _)).add
    (hP.const_mul _))).const_mul _

omit [IsProbabilityMeasure μ] in
theorem Problem.grad_measurable (P : Problem N m n μ c) : Measurable P.grad :=
  (gradient_continuous (fun i => (c i).L_nonneg) P.smooth).measurable

omit [IsProbabilityMeasure μ] in
theorem Problem.localGrad_measurable (P : Problem N m n μ c) (j : Fin N) : Measurable (P.localGrad j) :=
  (gradient_continuous (fun i => P.localL_nonneg i j) (P.localSmooth j)).measurable

omit [IsProbabilityMeasure μ] in
theorem Problem.energy_measurable (P : Problem N m n μ c) (i : Fin p) :
    Measurable (fun s : States N m n => energy (c i) (P.ell2 i) (s i)
      (P.grad (points s) i) (fun j => P.localGrad j (points s) i)) :=
  measurable_energy (measurable_pi_apply i)
    ((measurable_pi_apply i).comp (P.grad_measurable.comp measurable_points))
    (fun j => (measurable_pi_apply i).comp ((P.localGrad_measurable j).comp measurable_points)) _

omit [IsProbabilityMeasure μ] in
theorem Problem.potential_measurable (P : Problem N m n μ c) : Measurable P.potential := by
  have hf : Continuous P.f := continuous_iff_continuousAt.mpr (fun x => (P.derivative x).continuousAt)
  exact ((hf.measurable.comp measurable_points).sub measurable_const).add
    (Finset.measurable_sum _ (fun i _ => P.energy_measurable i))

omit [IsProbabilityMeasure μ] in
theorem Problem.cost_measurable (P : Problem N m n μ c) : Measurable P.cost :=
  Finset.measurable_sum _ (fun i _ => ((continuous_nuclearNorm.measurable.comp
    ((measurable_pi_apply i).comp (P.grad_measurable.comp measurable_points))).pow_const 2).const_mul _)

theorem gradient_difference_sq {i : Fin p} {g : Model m n → Model m n} {L d : ℝ}
    (hL : 0 ≤ L) (hlip : ∀ x y, nuclearNorm (g x i - g y i) ≤ L * spectralNorm (x i - y i))
    (x y : Model m n) (hd : spectralNorm (x i - y i) ≤ d) :
    ‖g x i - g y i‖ ^ 2 ≤ L ^ 2 * d ^ 2 := by
  have h := (frobenius_le_nuclear _).trans ((hlip x y).trans (mul_le_mul_of_nonneg_left hd hL))
  have hs := mul_self_le_mul_self (norm_nonneg _) h
  nlinarith only [hs]

theorem Problem.layer_drift (P : Problem N m n μ c) (A : Scheme N m n c)
    (hsmall : ∀ i, 2 * (c i).L * (c i).γ + 4 * zeta (c i) (P.ell2 i) (A.τ i) * (c i).γ ^ 2 ≤ 1)
    (s : States N m n) (i : Fin p) :
    inner ℝ (P.grad (points s) i) (nextX (c i) (s i) - (s i).x) +
      (c i).L / 2 * spectralNorm (nextX (c i) (s i) - (s i).x) ^ 2 +
      (∫ ω, energy (c i) (P.ell2 i) (next A P.reply s ω i)
        (P.grad (serverPoints c s) i) (fun j => P.localGrad j (serverPoints c s) i) ∂μ) +
        (c i).γ / 4 * nuclearNorm (P.grad (points s) i) ^ 2 ≤
      energy (c i) (P.ell2 i) (s i) (P.grad (points s) i)
        (fun j => P.localGrad j (points s) i) + (c i).γ * noise (c i) N * P.σ i ^ 2 := by
  have hmov : spectralNorm (serverPoints c s i - points s i) ≤
      (c i).γ * nuclearNorm (average (s i).G) := movement_spectral (c i) (s i).server
  have hbro : spectralNorm (queries A s i - serverPoints c s i) ≤
      ‖nextW (c i) (A.CP i) (s i) - nextX (c i) (s i)‖ := spectralNorm_le_frobenius _
  have ha := gradient_difference_sq (c i).L_nonneg (fun x y => P.smooth x y i) _ _ hmov
  rw [norm_sub_rev] at ha
  have hb := gradient_difference_sq (c i).L_nonneg (fun x y => P.smooth x y i) _ _ hbro
  have haJ := average_mono (fun j => gradient_difference_sq (P.localL_nonneg i j)
    (fun x y => P.localSmooth j x y i) _ _ hmov)
  have hbJ := average_mono (fun j => gradient_difference_sq (P.localL_nonneg i j)
    (fun x y => P.localSmooth j x y i) _ _ hbro)
  simp only [norm_sub_rev (P.localGrad _ (serverPoints c s) i)] at haJ
  have haJ' : average (fun j => ‖P.localGrad j (points s) i - P.localGrad j (serverPoints c s) i‖ ^ 2) ≤
      P.ell2 i * (c i).γ ^ 2 * nuclearNorm (average (s i).G) ^ 2 := by
    convert haJ using 1
    simp only [mul_pow, mul_assoc, mul_comm _ ((c i).γ ^ 2 * _), average_mul, Problem.ell2]
  have hbJ' : average (fun j => ‖P.localGrad j (queries A s) i - P.localGrad j (serverPoints c s) i‖ ^ 2) ≤
      P.ell2 i * ‖nextW (c i) (A.CP i) (s i) - nextX (c i) (s i)‖ ^ 2 := by
    convert hbJ using 1
    simp only [mul_comm _ (‖_‖ ^ 2), average_mul, Problem.ell2]
  simp only [next_frozen P A s]
  apply layer_descent (c i) P.workers_pos (P.ell2_nonneg i) (A.measurable_CU i)
    (A.contracts_CU i) (A.broadcast i) (A.dual i) (hsmall i) (s i)
    (P.grad (points s) i) (P.grad (serverPoints c s) i)
    (fun j => P.localGrad j (points s) i) (fun j => P.localGrad j (serverPoints c s) i)
    (fun j => P.localGrad j (queries A s) i) (P.frozenNoise A s i)
  · simpa only [mul_pow, mul_assoc] using ha
  · simpa only [P.mean_gradient] using hb
  · exact haJ'
  · exact hbJ'

theorem Problem.energy_next_integrable (P : Problem N m n μ c) (A : Scheme N m n c)
    (s : States N m n) (i : Fin p) :
    Integrable (fun ω => energy (c i) (P.ell2 i) (next A P.reply s ω i)
      (P.grad (serverPoints c s) i) (fun j => P.localGrad j (serverPoints c s) i)) μ := by
  simp only [next_frozen P A s]
  exact energy_integrable (c i) _ (A.measurable_CU i) (A.contracts_CU i) _ _ _ _ _ (P.frozenNoise A s i)

theorem Problem.potential_next_integrable (P : Problem N m n μ c) (A : Scheme N m n c)
    (s : States N m n) : Integrable (fun ω => P.potential (next A P.reply s ω)) μ := by
  simp only [Problem.potential, points_next]
  exact (integrable_const _).add (integrable_finsetSum _ (fun i _ => P.energy_next_integrable A s i))

theorem Problem.potential_descent (P : Problem N m n μ c) (A : Scheme N m n c)
    (hsmall : ∀ i, 2 * (c i).L * (c i).γ + 4 * zeta (c i) (P.ell2 i) (A.τ i) * (c i).γ ^ 2 ≤ 1)
    (s : States N m n) :
    (∫ ω, P.potential (next A P.reply s ω) ∂μ) + P.cost s ≤ P.potential s + P.noiseTotal := by
  have hd := smooth_upper P.derivative P.smooth (points s) (serverPoints c s)
  have h := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => P.layer_drift A hsmall s i)
  simp only [Problem.potential, points_next]
  rw [integral_add (integrable_const _) (integrable_finsetSum _
    (fun i _ => P.energy_next_integrable A s i)), integral_const, probReal_univ, one_smul,
    integral_finsetSum _ (fun i _ => P.energy_next_integrable A s i)]
  simp only [Finset.sum_add_distrib] at h hd
  dsimp only [Problem.cost, Problem.noiseTotal, points, serverPoints] at *
  linarith only [h, hd]

/-- Expected stationarity along the actual synchronous, multiworker, multilayer paths. -/
theorem Problem.finite_time (P : Problem N m n μ c) (A : Scheme N m n c)
    (hsmall : ∀ i, 2 * (c i).L * (c i).γ + 4 * zeta (c i) (P.ell2 i) (A.τ i) * (c i).γ ^ 2 ≤ 1)
    (T : ℕ) (s : States N m n) :
    Integrable (fun ξ => ((run (next A P.reply) T s ξ).map P.cost).sum) (pathMeasure μ T) ∧
    (∫ ξ, ((run (next A P.reply) T s ξ).map P.cost).sum ∂pathMeasure μ T) ≤
      P.potential s + (T : ℝ) * P.noiseTotal := by
  have h := totalPotential_integral_bound μ (measurable_next A P.measurable_reply)
    P.cost_measurable P.potential_measurable P.cost_nonneg P.potential_nonneg
    (P.potential_next_integrable A) (P.potential_descent A hsmall) T s
  have hm : Measurable (fun ξ : Samples Ω T => ((run (next A P.reply) T s ξ).map P.cost).sum) := by
    have ht : ∀ k, Measurable (Function.uncurry (terminal (next A P.reply) k)) := by
      intro k
      induction k with
      | zero => exact measurable_fst
      | succ k ihk => exact ihk.comp (((measurable_next A P.measurable_reply).comp
          (measurable_fst.prodMk measurable_snd.fst)).prodMk measurable_snd.snd)
    have hpair : Measurable (fun ξ : Samples Ω T => (s, ξ)) := measurable_const.prodMk measurable_id
    have hr : Measurable (fun ξ : Samples Ω T => totalPotential (next A P.reply) P.cost P.potential T s ξ) :=
      (totalPotential_measurable (measurable_next A P.measurable_reply)
        P.cost_measurable P.potential_measurable T).comp hpair
    have hterminal : Measurable (fun ξ : Samples Ω T => P.potential (terminal (next A P.reply) T s ξ)) :=
      P.potential_measurable.comp ((ht T).comp hpair)
    simpa only [Pi.sub_def, totalPotential_eq, add_sub_cancel_right] using hr.sub hterminal
  have hnonneg (ξ : Samples Ω T) : 0 ≤ ((run (next A P.reply) T s ξ).map P.cost).sum :=
    List.sum_nonneg (by intro r hr; obtain ⟨a, _, rfl⟩ := List.mem_map.mp hr; exact P.cost_nonneg a)
  have hle (ξ : Samples Ω T) : ((run (next A P.reply) T s ξ).map P.cost).sum ≤
      totalPotential (next A P.reply) P.cost P.potential T s ξ := by
    rw [totalPotential_eq]; exact le_add_of_nonneg_right (P.potential_nonneg _)
  have hi := h.1.mono' hm.aestronglyMeasurable (Filter.Eventually.of_forall fun ξ =>
    (Real.norm_of_nonneg (hnonneg ξ)).trans_le (hle ξ))
  exact ⟨hi, (integral_mono hi h.1 hle).trans h.2⟩

end EF21Muon.Distributed
