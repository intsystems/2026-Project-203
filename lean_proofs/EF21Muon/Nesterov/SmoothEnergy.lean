import EF21Muon.Nesterov.Tracking

noncomputable section

namespace EF21Muon.Nesterov

open MeasureTheory hiding average average_add average_sub average_const
open EF21Muon.Probability EF21Muon.Stochastic EF21Muon.Distributed
open LongTails.Algorithms.MatrixPolar
open scoped BigOperators

def dWeight (c : Parameters) : ℝ := 8*c.ρ^2
def hWeight (c : Parameters) : ℝ := 2*dWeight c/c.αU
def jWeight (c : Parameters) : ℝ := dWeight c+4*hWeight c/c.αU
def rWeight (c : Parameters) : ℝ := 16*hWeight c/c.αU+jWeight c
def fWeight (c : Parameters) : ℝ := 2*dWeight c/c.β+6*c.β*rWeight c
def injection (c : Parameters) (ell2 : ℝ) : ℝ :=
  fWeight c*c.β^2*(1+2/c.β)*ell2+3*c.β^2*rWeight c*ell2
def pWeight (c : Parameters) (ell2 : ℝ) : ℝ := 2*injection c ell2/c.αP
def zeta (c : Parameters) (ell2 τ : ℝ) : ℝ :=
  fWeight c*2*ell2/c.β+3*c.β^2*rWeight c*ell2+
    (injection c ell2+pWeight c ell2)*2*τ/c.αP
def noise (c : Parameters) : ℝ := (fWeight c+rWeight c)*c.β^2

theorem weights_nonneg (c : Parameters) {ell2 : ℝ} (hell : 0 ≤ ell2) :
    0 ≤ dWeight c ∧ 0 ≤ hWeight c ∧ 0 ≤ jWeight c ∧ 0 ≤ rWeight c ∧
    0 ≤ fWeight c ∧ 0 ≤ injection c ell2 ∧ 0 ≤ pWeight c ell2 := by
  have hb := c.β_pos
  have hu := c.αU_pos
  have hp := c.αP_pos
  dsimp [dWeight, hWeight, jWeight, rWeight, fWeight, injection, pWeight]
  exact ⟨by positivity, by positivity, by positivity, by positivity,
    by positivity, by positivity, by positivity⟩

theorem coefficients (c : Parameters) (ell2 : ℝ) :
    fWeight c*(1-c.β/2)+3*c.β^2*rWeight c = fWeight c-dWeight c ∧
    hWeight c*(1-c.αU/2) = hWeight c-dWeight c ∧
    4*hWeight c/c.αU = jWeight c-dWeight c ∧
    (injection c ell2+pWeight c ell2)*(1-c.αP/2) =
      pWeight c ell2-c.αP/2*injection c ell2 := by
  have hb := ne_of_gt c.β_pos
  have hu := ne_of_gt c.αU_pos
  have hp := ne_of_gt c.αP_pos
  dsimp [fWeight, hWeight, jWeight, pWeight]
  constructor
  · field_simp <;> ring
  constructor
  · field_simp <;> ring
  constructor
  · ring
  · field_simp <;> ring

variable {N m n : ℕ}

def energy (c : Parameters) (ell2 : ℝ) (s : Layer N m n)
    (localG : Fin N → Space m n) : ℝ :=
  c.γ*(fWeight c*localError s.1 localG+hWeight c*compression c s+
    jWeight c*increment s+pWeight c ell2*broadcastError s.1)

theorem energy_nonneg (c : Parameters) {ell2 : ℝ} (hell : 0 ≤ ell2)
    (s : Layer N m n) (localG : Fin N → Space m n) : 0 ≤ energy c ell2 s localG := by
  rcases weights_nonneg c hell with ⟨_, hh, hj, _, hf, _, hp⟩
  have hF : 0 ≤ localError s.1 localG := average_nonneg (fun _ => sq_nonneg _)
  have hH : 0 ≤ compression c s := average_nonneg (fun _ => sq_nonneg _)
  have hJ : 0 ≤ increment s := average_nonneg (fun _ => sq_nonneg _)
  have hP : 0 ≤ broadcastError s.1 := sq_nonneg _
  exact mul_nonneg c.γ_pos.le (add_nonneg (add_nonneg (add_nonneg
    (mul_nonneg hf hF) (mul_nonneg hh hH)) (mul_nonneg hj hJ)) (mul_nonneg hp hP))

theorem direction_bound (c : Parameters) (hN : 0 < N)
    (hdual : ∀ z : Space m n, nuclearNorm z ≤ c.ρ*‖z‖)
    (s : Layer N m n) (g : Space m n) (localG : Fin N → Space m n)
    (hmean : average localG = g) :
    inner ℝ g (nextX c s.1-s.1.x) ≤
      c.γ*dWeight c*(localError s.1 localG+compression c s+increment s)-
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
  have he3 := mul_le_mul_of_nonneg_left (mean_error_sq c hN s g localG hmean) (sq_nonneg c.ρ)
  have heb : e^2 ≤ 3*c.ρ^2*(localError s.1 localG+compression c s+increment s) := by
    dsimp [e,G] at *; nlinarith only [he2,he3]
  have hγ := c.γ_pos.le
  have hγe := mul_le_mul_of_nonneg_left heb hγ
  have hγg := mul_le_mul_of_nonneg_left hg2 hγ
  have hpair := mul_le_mul_of_nonneg_left hp (mul_nonneg hγ (nuclearNorm_nonneg G))
  have hs := mul_nonneg hγ (sq_nonneg (a/2-e))
  have hs2 := mul_nonneg hγ (sq_nonneg (a-e))
  have hsum : 0 ≤ localError s.1 localG+compression c s+increment s :=
    add_nonneg (add_nonneg (average_nonneg (fun _ => sq_nonneg _))
      (average_nonneg (fun _ => sq_nonneg _))) (average_nonneg (fun _ => sq_nonneg _))
  have hpos := mul_nonneg (mul_nonneg hγ (sq_nonneg c.ρ)) hsum
  have hstep : nextX c s.1-s.1.x = -(c.γ*a) • polar G := by
    dsimp [nextX, serverPoint, LayerState.server, LayerState.x, LayerState.G, a,G]; module
  rw [hstep, inner_smul_right]
  dsimp only [dWeight]
  change _ ≤ _-c.γ/4*nuclearNorm g^2-c.γ/4*a^2
  nlinarith only [hγe,hγg,hpair,hs,hs2,hpos]

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

theorem energy_integrable (c : Parameters) (ell2 : ℝ) {CU : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU) (CP : Space m n → Space m n)
    (s : Layer N m n) (localG q : Fin N → Space m n) {σ : ℝ} (O : Noise N m n μ q σ) :
    Integrable (fun ω => energy c ell2 (update c CU CP s (O.replies ω)) localG) μ := by
  have hiF := local_integrable c CU CP s O localG
  have hiH := compression_integrable c hm hc CP s O
  have hiJ := increment_integrable c CU CP s O
  exact ((((hiF.const_mul _).add (hiH.const_mul _)).add (hiJ.const_mul _)).add
    (integrable_const (pWeight c ell2*‖nextW c CP s.1-nextX c s.1‖^2))).const_mul _

set_option maxHeartbeats 1600000 in
theorem layer_descent (c : Parameters) (hN : 0 < N) {ell2 τ σ : ℝ}
    (hell : 0 ≤ ell2) {CU CP : Space m n → Space m n}
    (hm : Measurable CU) (hc : SquaredContracts CU c.αU)
    (hdown : BroadcastBound c CP τ)
    (hdual : ∀ z : Space m n, nuclearNorm z ≤ c.ρ*‖z‖)
    (hsmall : 2*c.L*c.γ+4*zeta c ell2 τ*c.γ^2 ≤ 1)
    (s : Layer N m n) (old : Space m n) (oldJ newJ q : Fin N → Space m n)
    (hmean : average oldJ = old) (O : Noise N m n μ q σ)
    (ha : average (fun j => ‖oldJ j-newJ j‖^2) ≤
      ell2*c.γ^2*nuclearNorm (average s.1.G)^2)
    (hb : average (fun j => ‖q j-newJ j‖^2) ≤
      ell2*‖nextW c CP s.1-nextX c s.1‖^2) :
    inner ℝ old (nextX c s.1-s.1.x)+c.L/2*spectralNorm (nextX c s.1-s.1.x)^2+
      (∫ ω, energy c ell2 (update c CU CP s (O.replies ω)) newJ ∂μ)+
        c.γ/4*nuclearNorm old^2 ≤ energy c ell2 s oldJ+c.γ*noise c*σ^2 := by
  rcases weights_nonneg c hell with ⟨hd0,hh0,hj0,hr0,hf0,hi0,hp0⟩
  have hγ := c.γ_pos
  have hd := direction_bound c hN hdual s old oldJ hmean
  have hs := mul_self_le_mul_self (show 0 ≤ spectralNorm (nextX c s.1-s.1.x) from norm_nonneg _)
    (movement_spectral c s.1.server)
  have hs' := mul_le_mul_of_nonneg_left hs (div_nonneg c.L_nonneg (by norm_num : (0:ℝ) ≤ 2))
  have hf := mul_le_mul_of_nonneg_left (local_recurrence c hN CU CP s oldJ newJ q O ha hb)
    (mul_nonneg hγ.le hf0)
  have hh := mul_le_mul_of_nonneg_left (compression_recurrence c hm hc CP s O)
    (mul_nonneg hγ.le hh0)
  have hj := mul_le_mul_of_nonneg_left (increment_recurrence c hN CU CP s oldJ newJ q O ha hb)
    (mul_nonneg hγ.le hr0)
  have hp := mul_le_mul_of_nonneg_left (hdown s.1.server)
    (mul_nonneg hγ.le (add_nonneg hi0 hp0))
  rcases coefficients c ell2 with ⟨hF,hH,hJ,hP⟩
  have hF' := congrArg (fun r : ℝ => c.γ*r*localError s.1 oldJ) hF
  have hH' := congrArg (fun r : ℝ => c.γ*r*compression c s) hH
  have hJ' := congrArg (fun r : ℝ => c.γ*r*increment s) hJ
  have hP' := congrArg (fun r : ℝ => c.γ*r*broadcastError s.1) hP
  have hdrop : 0 ≤ c.γ*(c.αP/2)*injection c ell2*broadcastError s.1 :=
    mul_nonneg (mul_nonneg (mul_nonneg hγ.le (by linarith [c.αP_pos])) hi0) (sq_nonneg _)
  have hR : c.γ*rWeight c*(∫ ω, increment (update c CU CP s (O.replies ω)) ∂μ) =
      c.γ*(16*hWeight c/c.αU+jWeight c)*(∫ ω, increment (update c CU CP s (O.replies ω)) ∂μ) := rfl
  have hI : injection c ell2 = fWeight c*c.β^2*(1+2/c.β)*ell2+3*c.β^2*rWeight c*ell2 := rfl
  have hI' := congrArg (fun r : ℝ => c.γ*r*‖nextW c CP s.1-nextX c s.1‖^2) hI
  have hZ : c.γ^3*zeta c ell2 τ*nuclearNorm (average s.1.G)^2 =
      c.γ^3*(fWeight c*2*ell2/c.β+3*c.β^2*rWeight c*ell2+
      (injection c ell2+pWeight c ell2)*2*τ/c.αP)*nuclearNorm (average s.1.G)^2 := rfl
  have hN : c.γ*noise c*σ^2 = c.γ*(fWeight c+rWeight c)*c.β^2*σ^2 := by
    dsimp [noise]; ring
  have hstep := mul_le_mul_of_nonneg_left hsmall
    (by positivity : 0 ≤ c.γ/4*nuclearNorm (average s.1.G)^2)
  have hiF := local_integrable c CU CP s O newJ
  have hiH := compression_integrable c hm hc CP s O
  have hiJ := increment_integrable c CU CP s O
  have hbp (ω : Ω) : broadcastError (update c CU CP s (O.replies ω)).1 =
      ‖nextW c CP s.1-nextX c s.1‖^2 := rfl
  simp only [energy, hbp, integral_const_mul]
  rw [integral_add4 (hiF.const_mul _) (hiH.const_mul _) (hiJ.const_mul _) (integrable_const _)]
  simp only [integral_const_mul, integral_const, probReal_univ, one_smul]
  change c.γ*(injection c ell2+pWeight c ell2)*‖nextW c CP s.1-nextX c s.1‖^2 ≤
    c.γ*(injection c ell2+pWeight c ell2)*((1-c.αP/2)*broadcastError s.1+
      2*τ/c.αP*c.γ^2*nuclearNorm (average s.1.G)^2) at hp
  change c.L/2*(spectralNorm (nextX c s.1-s.1.x)*spectralNorm (nextX c s.1-s.1.x)) ≤
    c.L/2*(c.γ*nuclearNorm (average s.1.G)*(c.γ*nuclearNorm (average s.1.G))) at hs'
  simp only [div_eq_mul_inv] at hf hh hj hp hF' hH' hJ' hP' hR hI' hZ hN hstep ⊢
  nlinarith only [hd,hs',hf,hh,hj,hp,hF',hH',hJ',hP',hdrop,hR,hI',hZ,hN,hstep]

end EF21Muon.Nesterov
