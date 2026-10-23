import EF21Muon.Nesterov.AveragedTracking
import EF21Muon.Nesterov.SmoothEnergy

/-! Split the old local weights between global and local errors. Their sums
are the old weights, so the old step-size condition remains sufficient. -/

noncomputable section

namespace EF21Muon.Nesterov.Averaged

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

def jWeight (c : Parameters) : ℝ := 4*hWeight c/c.αU
def rWeight (c : Parameters) : ℝ := 16*hWeight c/c.αU+jWeight c
def eWeight (c : Parameters) : ℝ := 2*dWeight c/c.β+6*c.β*dWeight c
def lWeight (c : Parameters) : ℝ := 6*c.β*rWeight c
def noise (c : Parameters) (N : ℕ) : ℝ :=
  ((eWeight c+dWeight c)/(N:ℝ)+lWeight c+rWeight c)*c.β^2

theorem weight_sums (c : Parameters) :
    eWeight c+lWeight c = Nesterov.fWeight c ∧
    dWeight c+rWeight c = Nesterov.rWeight c := by
  dsimp [eWeight, lWeight, rWeight, jWeight, Nesterov.fWeight,
    Nesterov.rWeight, Nesterov.jWeight]
  constructor <;> ring

theorem weights_nonneg (c : Parameters) :
    0 ≤ dWeight c ∧ 0 ≤ hWeight c ∧ 0 ≤ jWeight c ∧
    0 ≤ rWeight c ∧ 0 ≤ eWeight c ∧ 0 ≤ lWeight c := by
  have hb := c.β_pos
  have hu := c.αU_pos
  dsimp [dWeight, hWeight, jWeight, rWeight, eWeight, lWeight]
  exact ⟨by positivity, by positivity, by positivity, by positivity,
    by positivity, by positivity⟩

theorem coefficients (c : Parameters) :
    eWeight c*(1-c.β/2)+3*c.β^2*dWeight c = eWeight c-dWeight c ∧
    lWeight c*(1-c.β/2)+3*c.β^2*rWeight c = lWeight c ∧
    hWeight c*(1-c.αU/2) = hWeight c-dWeight c := by
  have hb := ne_of_gt c.β_pos
  have hu := ne_of_gt c.αU_pos
  dsimp [eWeight, lWeight, hWeight]
  constructor
  · field_simp <;> ring
  constructor
  · ring
  · field_simp <;> ring

variable {N m n : ℕ}

def energy (c : Parameters) (ell2 : ℝ) (s : Layer N m n)
    (g : Space m n) (localG : Fin N → Space m n) : ℝ :=
  c.γ*(eWeight c*globalError s.1 g+lWeight c*localError s.1 localG+
    hWeight c*compression c s+dWeight c*meanIncrement s+
    jWeight c*increment s+pWeight c ell2*broadcastError s.1)

theorem energy_nonneg (c : Parameters) {ell2 : ℝ} (hell : 0 ≤ ell2)
    (s : Layer N m n) (g : Space m n) (localG : Fin N → Space m n) :
    0 ≤ energy c ell2 s g localG := by
  rcases weights_nonneg c with ⟨hd, hh, hj, _, he, hl⟩
  have hp := (Nesterov.weights_nonneg c hell).2.2.2.2.2.2
  have hF : 0 ≤ localError s.1 localG := average_nonneg (fun _ => sq_nonneg _)
  have hH : 0 ≤ compression c s := average_nonneg (fun _ => sq_nonneg _)
  have hJ : 0 ≤ increment s := average_nonneg (fun _ => sq_nonneg _)
  have hE : 0 ≤ globalError s.1 g := sq_nonneg _
  have hK : 0 ≤ meanIncrement s := sq_nonneg _
  have hP : 0 ≤ broadcastError s.1 := sq_nonneg _
  have hg := c.γ_pos.le
  unfold energy
  positivity

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

theorem energy_integrable (c : Parameters) (ell2 : ℝ) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU) (CP : Space m n → Space m n)
    (s : Layer N m n) (g : Space m n) (localG q : Fin N → Space m n) {σ : ℝ}
    (O : Noise N m n μ q σ) :
    Integrable (fun ω => energy c ell2 (update c CU CP s (O.replies ω)) g localG) μ := by
  have hiE := (global_integrable c CU CP s O g).const_mul (eWeight c)
  have hiF := (local_integrable c CU CP s O localG).const_mul (lWeight c)
  have hiH := (compression_integrable c hm hc CP s O).const_mul (hWeight c)
  have hiK := (meanIncrement_integrable c CU CP s O).const_mul (dWeight c)
  have hiJ := (increment_integrable c CU CP s O).const_mul (jWeight c)
  exact (((((hiE.add hiF).add hiH).add hiK).add hiJ).add
    (integrable_const (pWeight c ell2*‖nextW c CP s.1-nextX c s.1‖^2))).const_mul c.γ

theorem direction_bound (c : Parameters) (hN : 0 < N)
    (hdual : ∀ z : Space m n, nuclearNorm z ≤ c.ρ*‖z‖)
    (s : Layer N m n) (g : Space m n) :
    inner ℝ g (nextX c s.1-s.1.x) ≤
      c.γ*dWeight c*(globalError s.1 g+compression c s+meanIncrement s)-
        c.γ/4*nuclearNorm g^2-c.γ/4*nuclearNorm (average s.1.G)^2 := by
  let G := average s.1.G
  let a := nuclearNorm G
  let e := nuclearNorm (G-g)
  have hi := (le_abs_self (inner ℝ (G-g) (polar G))).trans (inner_le_nuclear_mul_spectral _ _)
  have hi' := mul_le_mul_of_nonneg_left (polar_mem G) (nuclearNorm_nonneg (G-g))
  rw [inner_sub_left, polar_pairing_eq_nuclear] at hi
  have hp : a-e ≤ inner ℝ g (polar G) := by dsimp [a,e]; linarith only [hi,hi']
  have ht := nuclearNorm_add_le G (g-G)
  rw [add_sub_cancel] at ht
  have hr : nuclearNorm (g-G) = e := by dsimp [e]; rw [← neg_sub G g, nuclearNorm_neg]
  rw [hr] at ht
  have hg2 := mul_self_le_mul_self (nuclearNorm_nonneg g) ht
  have he2 := mul_self_le_mul_self (nuclearNorm_nonneg _) (hdual (G-g))
  have he3 := mul_le_mul_of_nonneg_left (mean_error_sq c hN s g) (sq_nonneg c.ρ)
  have heb : e^2 ≤ 3*c.ρ^2*(globalError s.1 g+compression c s+meanIncrement s) := by
    dsimp [e,G] at *; nlinarith only [he2,he3]
  have hγ := c.γ_pos.le
  have hγe := mul_le_mul_of_nonneg_left heb hγ
  have hγg := mul_le_mul_of_nonneg_left hg2 hγ
  have hpair := mul_le_mul_of_nonneg_left hp (mul_nonneg hγ (nuclearNorm_nonneg G))
  have hs := mul_nonneg hγ (sq_nonneg (a/2-e))
  have hs2 := mul_nonneg hγ (sq_nonneg (a-e))
  have hsum : 0 ≤ globalError s.1 g+compression c s+meanIncrement s :=
    add_nonneg (add_nonneg (sq_nonneg _)
      (average_nonneg (fun _ => sq_nonneg _))) (sq_nonneg _)
  have hpos := mul_nonneg (mul_nonneg hγ (sq_nonneg c.ρ)) hsum
  have hstep : nextX c s.1-s.1.x = -(c.γ*a) • polar G := by
    dsimp [nextX, serverPoint, LayerState.server, LayerState.x, LayerState.G, a,G]; module
  rw [hstep, inner_smul_right]
  dsimp only [dWeight]
  change _ ≤ _-c.γ/4*nuclearNorm g^2-c.γ/4*a^2
  nlinarith only [hγe,hγg,hpair,hs,hs2,hpos]

omit [IsProbabilityMeasure μ] in
theorem integral_add6 {a b c d e f : Ω → ℝ}
    (ha : Integrable a μ) (hb : Integrable b μ) (hc : Integrable c μ)
    (hd : Integrable d μ) (he : Integrable e μ) (hf : Integrable f μ) :
    (∫ ω, a ω+b ω+c ω+d ω+e ω+f ω ∂μ) =
      (∫ ω, a ω ∂μ)+(∫ ω, b ω ∂μ)+(∫ ω, c ω ∂μ)+
        (∫ ω, d ω ∂μ)+(∫ ω, e ω ∂μ)+(∫ ω, f ω ∂μ) := by
  have h2 : Integrable (fun ω => a ω+b ω) μ := ha.add hb
  have h3 : Integrable (fun ω => a ω+b ω+c ω) μ := h2.add hc
  have h4 : Integrable (fun ω => a ω+b ω+c ω+d ω) μ := h3.add hd
  have h5 : Integrable (fun ω => a ω+b ω+c ω+d ω+e ω) μ := h4.add he
  rw [integral_add h5 hf, integral_add h4 he, integral_add h3 hd,
    integral_add h2 hc, integral_add ha hb]

set_option maxHeartbeats 2400000 in
theorem layer_descent (c : Parameters) (hN : 0 < N) {ell2 τ σ : ℝ}
    (hell : 0 ≤ ell2) {CU CP : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (hdown : BroadcastBound c CP τ)
    (hdual : ∀ z : Space m n, nuclearNorm z ≤ c.ρ*‖z‖)
    (hsmall : 2*c.L*c.γ+4*zeta c ell2 τ*c.γ^2 ≤ 1)
    (s : Layer N m n) (old new : Space m n) (oldJ newJ q : Fin N → Space m n)
    (hmean : average oldJ = old) (hnew : average newJ = new) (O : Noise N m n μ q σ)
    (ha : average (fun j => ‖oldJ j-newJ j‖^2) ≤
      ell2*c.γ^2*nuclearNorm (average s.1.G)^2)
    (hb : average (fun j => ‖q j-newJ j‖^2) ≤
      ell2*‖nextW c CP s.1-nextX c s.1‖^2) :
    inner ℝ old (nextX c s.1-s.1.x)+c.L/2*spectralNorm (nextX c s.1-s.1.x)^2+
      (∫ ω, energy c ell2 (update c CU CP s (O.replies ω)) new newJ ∂μ)+
        c.γ/4*nuclearNorm old^2 ≤ energy c ell2 s old oldJ+c.γ*noise c N*σ^2 := by
  rcases weights_nonneg c with ⟨hd0,hh0,hj0,hr0,he0,hl0⟩
  rcases Nesterov.weights_nonneg c hell with ⟨_,_,_,_,_,hi0,hp0⟩
  have hγ := c.γ_pos
  have haG : ‖old-new‖^2 ≤ ell2*c.γ^2*nuclearNorm (average s.1.G)^2 := by
    rw [← hmean, ← hnew, ← average_sub]
    exact (average_sq_le hN (fun j => oldJ j-newJ j)).trans ha
  have hbG : ‖average q-new‖^2 ≤ ell2*‖nextW c CP s.1-nextX c s.1‖^2 := by
    rw [← hnew, ← average_sub]
    exact (average_sq_le hN (fun j => q j-newJ j)).trans hb
  have hd := direction_bound c hN hdual s old
  have hs := mul_self_le_mul_self (show 0 ≤ spectralNorm (nextX c s.1-s.1.x) from norm_nonneg _)
    (movement_spectral c s.1.server)
  have hs' := mul_le_mul_of_nonneg_left hs (div_nonneg c.L_nonneg (by norm_num : (0:ℝ) ≤ 2))
  have he := mul_le_mul_of_nonneg_left (global_recurrence c hN CU CP s old new q O haG hbG)
    (mul_nonneg hγ.le he0)
  have hf := mul_le_mul_of_nonneg_left (local_recurrence c hN CU CP s oldJ newJ q O ha hb)
    (mul_nonneg hγ.le hl0)
  have hh := mul_le_mul_of_nonneg_left (compression_recurrence c hm hc CP s O)
    (mul_nonneg hγ.le hh0)
  have hk := mul_le_mul_of_nonneg_left (meanIncrement_recurrence c hN CU CP s old new q O haG hbG)
    (mul_nonneg hγ.le hd0)
  have hj := mul_le_mul_of_nonneg_left (increment_recurrence c hN CU CP s oldJ newJ q O ha hb)
    (mul_nonneg hγ.le hr0)
  have hp := mul_le_mul_of_nonneg_left (hdown s.1.server)
    (mul_nonneg hγ.le (add_nonneg hi0 hp0))
  rcases coefficients c with ⟨hE,hF,hH⟩
  have hP := (Nesterov.coefficients c ell2).2.2.2
  have hE' := congrArg (fun r : ℝ => c.γ*r*globalError s.1 old) hE
  have hF' := congrArg (fun r : ℝ => c.γ*r*localError s.1 oldJ) hF
  have hH' := congrArg (fun r : ℝ => c.γ*r*compression c s) hH
  have hJ' : c.γ*(4*hWeight c/c.αU)*increment s = c.γ*jWeight c*increment s := rfl
  have hP' := congrArg (fun r : ℝ => c.γ*r*broadcastError s.1) hP
  have hdrop : 0 ≤ c.γ*(c.αP/2)*injection c ell2*broadcastError s.1 :=
    mul_nonneg (mul_nonneg (mul_nonneg hγ.le (by linarith [c.αP_pos])) hi0) (sq_nonneg _)
  have hR : c.γ*rWeight c*(∫ ω, increment (update c CU CP s (O.replies ω)) ∂μ) =
      c.γ*(16*hWeight c/c.αU+jWeight c)*(∫ ω, increment (update c CU CP s (O.replies ω)) ∂μ) := rfl
  have hI : injection c ell2 =
      (eWeight c+lWeight c)*c.β^2*(1+2/c.β)*ell2+
        3*c.β^2*(dWeight c+rWeight c)*ell2 := by
    rw [(weight_sums c).1, (weight_sums c).2]
    rfl
  have hI' := congrArg (fun r : ℝ => c.γ*r*‖nextW c CP s.1-nextX c s.1‖^2) hI
  have hZ : c.γ^3*zeta c ell2 τ*nuclearNorm (average s.1.G)^2 =
      c.γ^3*((eWeight c+lWeight c)*2*ell2/c.β+3*c.β^2*(dWeight c+rWeight c)*ell2+
      (injection c ell2+pWeight c ell2)*2*τ/c.αP)*nuclearNorm (average s.1.G)^2 := by
    rw [(weight_sums c).1, (weight_sums c).2]
    rfl
  have hN : c.γ*noise c N*σ^2 =
      c.γ*eWeight c*(c.β^2*σ^2/N)+c.γ*dWeight c*(c.β^2*σ^2/N)+
        c.γ*lWeight c*c.β^2*σ^2+c.γ*rWeight c*c.β^2*σ^2 := by
    unfold noise
    ring
  have hstep := mul_le_mul_of_nonneg_left hsmall
    (by positivity : 0 ≤ c.γ/4*nuclearNorm (average s.1.G)^2)
  have hiE := global_integrable c CU CP s O new
  have hiF := local_integrable c CU CP s O newJ
  have hiH := compression_integrable c hm hc CP s O
  have hiK := meanIncrement_integrable c CU CP s O
  have hiJ := increment_integrable c CU CP s O
  have hbp (ω : Ω) : broadcastError (update c CU CP s (O.replies ω)).1 =
      ‖nextW c CP s.1-nextX c s.1‖^2 := rfl
  simp only [energy, hbp, integral_const_mul]
  have hEint := hiE.const_mul (eWeight c)
  have hFint := hiF.const_mul (lWeight c)
  have hHint := hiH.const_mul (hWeight c)
  have hKint := hiK.const_mul (dWeight c)
  have hJint := hiJ.const_mul (jWeight c)
  rw [integral_add6 hEint hFint hHint hKint hJint
    (integrable_const (pWeight c ell2*‖nextW c CP s.1-nextX c s.1‖^2))]
  simp only [integral_const_mul, integral_const, probReal_univ, one_smul]
  change c.γ*(injection c ell2+pWeight c ell2)*‖nextW c CP s.1-nextX c s.1‖^2 ≤
    c.γ*(injection c ell2+pWeight c ell2)*((1-c.αP/2)*broadcastError s.1+
      2*τ/c.αP*c.γ^2*nuclearNorm (average s.1.G)^2) at hp
  change c.L/2*(spectralNorm (nextX c s.1-s.1.x)*spectralNorm (nextX c s.1-s.1.x)) ≤
    c.L/2*(c.γ*nuclearNorm (average s.1.G)*(c.γ*nuclearNorm (average s.1.G))) at hs'
  simp only [div_eq_mul_inv] at he hf hh hk hj hp hE' hF' hH' hJ' hP' hR hI' hZ hN hstep ⊢
  nlinarith only [hd,hs',he,hf,hh,hk,hj,hp,hE',hF',hH',hJ',hP',hdrop,hR,hI',hZ,hN,hstep]

end EF21Muon.Nesterov.Averaged
