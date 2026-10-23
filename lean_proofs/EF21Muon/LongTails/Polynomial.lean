import EF21Muon.LongTails.Tuning
import Mathlib.Topology.Order.IntermediateValue

/-! Retain all momentum powers in the finite-time bound. The optimizer below
minimizes this scalar certificate, not the algorithm's true performance. -/

noncomputable section

namespace EF21Muon.LongTails

structure Profile where
  a : ℝ
  h : ℝ
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  a_nonneg : 0 ≤ a
  h_nonneg : 0 ≤ h
  s₁_nonneg : 0 ≤ s₁
  s₂_nonneg : 0 ≤ s₂
  s₃_nonneg : 0 ≤ s₃

def Profile.bound (P : Profile) (T : ℕ) (β : ℝ) : ℝ :=
  4 * (P.a / (β * T) + P.h * β / T +
    P.s₁ * β + P.s₂ * β ^ 2 + P.s₃ * β ^ 3)

def Profile.root₁ (P : Profile) (ε : ℝ) : ℝ := 24 * P.s₁ / ε ^ 2
def Profile.root₂ (P : Profile) (ε : ℝ) : ℝ := Real.sqrt (24 * P.s₂ / ε ^ 2)
def Profile.root₃ (P : Profile) (ε : ℝ) : ℝ := (24 * P.s₃ / ε ^ 2) ^ (1 / 3 : ℝ)

def Profile.balancedBeta (P : Profile) (ε : ℝ) : ℝ :=
  1 / (1 + P.root₁ ε + P.root₂ ε + P.root₃ ε)

theorem Profile.roots_nonneg (P : Profile) (ε : ℝ) :
    0 ≤ P.root₁ ε ∧ 0 ≤ P.root₂ ε ∧ 0 ≤ P.root₃ ε := by
  dsimp [root₁, root₂, root₃]
  exact ⟨div_nonneg (by linarith [P.s₁_nonneg]) (sq_nonneg ε),
    Real.sqrt_nonneg _, Real.rpow_nonneg
      (div_nonneg (mul_nonneg (by norm_num) P.s₃_nonneg) (sq_nonneg ε)) _⟩

theorem Profile.balancedBeta_pos (P : Profile) (ε : ℝ) : 0 < P.balancedBeta ε := by
  rcases P.roots_nonneg ε with ⟨h₁, h₂, h₃⟩
  dsimp [balancedBeta]
  positivity

theorem Profile.balancedBeta_le_one (P : Profile) (ε : ℝ) : P.balancedBeta ε ≤ 1 := by
  rcases P.roots_nonneg ε with ⟨h₁, h₂, h₃⟩
  dsimp [balancedBeta]
  apply (div_le_one (by positivity)).mpr
  linarith

theorem Profile.beta_mul_roots (P : Profile) (ε : ℝ) :
    P.balancedBeta ε * P.root₁ ε ≤ 1 ∧
    P.balancedBeta ε * P.root₂ ε ≤ 1 ∧
    P.balancedBeta ε * P.root₃ ε ≤ 1 := by
  rcases P.roots_nonneg ε with ⟨h₁, h₂, h₃⟩
  have hd : 0 < 1 + P.root₁ ε + P.root₂ ε + P.root₃ ε := by positivity
  simp only [balancedBeta, one_div_mul_eq_div]
  exact ⟨(div_le_one hd).mpr (by linarith),
    (div_le_one hd).mpr (by linarith), (div_le_one hd).mpr (by linarith)⟩

theorem Profile.root_powers (P : Profile) {ε : ℝ} (hε : 0 < ε) :
    P.root₁ ε * ε ^ 2 = 24 * P.s₁ ∧
    P.root₂ ε ^ 2 * ε ^ 2 = 24 * P.s₂ ∧
    P.root₃ ε ^ 3 * ε ^ 2 = 24 * P.s₃ := by
  have he : ε ^ 2 ≠ 0 := ne_of_gt (sq_pos_of_pos hε)
  have h₂ : 0 ≤ 24 * P.s₂ / ε ^ 2 :=
    div_nonneg (mul_nonneg (by norm_num) P.s₂_nonneg) (sq_nonneg ε)
  have h₃ : 0 ≤ 24 * P.s₃ / ε ^ 2 :=
    div_nonneg (mul_nonneg (by norm_num) P.s₃_nonneg) (sq_nonneg ε)
  have hc : P.root₃ ε ^ 3 = 24 * P.s₃ / ε ^ 2 := by
    dsimp [root₃]
    rw [← Real.rpow_mul_natCast h₃]
    norm_num
  exact ⟨div_mul_cancel₀ _ he,
    by rw [root₂, Real.sq_sqrt h₂, div_mul_cancel₀ _ he],
    by rw [hc, div_mul_cancel₀ _ he]⟩

theorem Profile.balanced_noise (P : Profile) {ε : ℝ} (hε : 0 < ε) :
    4 * (P.s₁ * P.balancedBeta ε + P.s₂ * P.balancedBeta ε ^ 2 +
      P.s₃ * P.balancedBeta ε ^ 3) ≤ ε ^ 2 / 2 := by
  rcases P.roots_nonneg ε with ⟨hr₁, hr₂, hr₃⟩
  rcases P.beta_mul_roots ε with ⟨h₁, h₂, h₃⟩
  rcases P.root_powers hε with ⟨he₁, he₂, he₃⟩
  have hb := (P.balancedBeta_pos ε).le
  have hs₂ : (P.balancedBeta ε * P.root₂ ε) ^ 2 ≤ 1 :=
    pow_le_one₀ (mul_nonneg hb hr₂) h₂
  have hs₃ : (P.balancedBeta ε * P.root₃ ε) ^ 3 ≤ 1 :=
    pow_le_one₀ (mul_nonneg hb hr₃) h₃
  have ht₁ := mul_le_mul_of_nonneg_right h₁ (sq_nonneg ε)
  have ht₂ := mul_le_mul_of_nonneg_right hs₂ (sq_nonneg ε)
  have ht₃ := mul_le_mul_of_nonneg_right hs₃ (sq_nonneg ε)
  have e₁ : P.balancedBeta ε * P.root₁ ε * ε ^ 2 =
      24 * (P.s₁ * P.balancedBeta ε) := by rw [mul_assoc, he₁]; ring
  have e₂ : (P.balancedBeta ε * P.root₂ ε) ^ 2 * ε ^ 2 =
      24 * (P.s₂ * P.balancedBeta ε ^ 2) := by rw [mul_pow, mul_assoc, he₂]; ring
  have e₃ : (P.balancedBeta ε * P.root₃ ε) ^ 3 * ε ^ 2 =
      24 * (P.s₃ * P.balancedBeta ε ^ 3) := by rw [mul_pow, mul_assoc, he₃]; ring
  rw [e₁] at ht₁
  rw [e₂] at ht₂
  rw [e₃] at ht₃
  linarith only [ht₁, ht₂, ht₃]

def Profile.balancedBudget (P : Profile) (ε : ℝ) : ℕ :=
  max 1 ⌈8 * (P.a + P.h) / ε ^ 2 *
    (1 + P.root₁ ε + P.root₂ ε + P.root₃ ε)⌉₊

theorem Profile.balancedBudget_pos (P : Profile) (ε : ℝ) :
    0 < P.balancedBudget ε := lt_of_lt_of_le (by norm_num) (le_max_left _ _)

theorem Profile.balancedBudget_eq (P : Profile) {ε : ℝ} (hε : 0 < ε) :
    P.balancedBudget ε = Distributed.stochasticBudget (P.a + P.h) (P.balancedBeta ε) ε := by
  unfold balancedBudget Distributed.stochasticBudget balancedBeta
  congr 2
  field_simp

theorem Profile.bound_at_balancedBudget (P : Profile) {ε : ℝ} (hε : 0 < ε) :
    P.bound (P.balancedBudget ε) (P.balancedBeta ε) ≤ ε ^ 2 := by
  have hb := P.balancedBeta_pos ε
  have hb1 := P.balancedBeta_le_one ε
  have ht : (0 : ℝ) < P.balancedBudget ε := by exact_mod_cast P.balancedBudget_pos ε
  have hsq : P.balancedBeta ε ^ 2 ≤ 1 := pow_le_one₀ hb.le hb1
  have hi : P.a / (P.balancedBeta ε * P.balancedBudget ε) +
      P.h * P.balancedBeta ε / P.balancedBudget ε ≤
      (P.a + P.h) / (P.balancedBeta ε * P.balancedBudget ε) := by
    apply (le_div_iff₀ (mul_pos hb ht)).mpr
    field_simp
    nlinarith [mul_le_mul_of_nonneg_left hsq P.h_nonneg]
  have hbudget := Distributed.budget_transient (Ψ := P.a + P.h) hb hε
  rw [← P.balancedBudget_eq hε] at hbudget
  simp only [mul_div_assoc] at hbudget
  have hn := P.balanced_noise hε
  dsimp [bound]
  linarith only [hi, hbudget, hn]

def scalarBound (a b c d β : ℝ) : ℝ := a / β + b * β + c * β ^ 2 + d * β ^ 3

def rootPolynomial (b c d β : ℝ) : ℝ := b * β ^ 2 + 2 * c * β ^ 3 + 3 * d * β ^ 4

theorem root_exists {a b c d : ℝ} (ha : 0 < a) (ht : a < b + 2 * c + 3 * d) :
    ∃ β : ℝ, 0 < β ∧ β ≤ 1 ∧ rootPolynomial b c d β = a := by
  have hc : Continuous (rootPolynomial b c d) := by unfold rootPolynomial; fun_prop
  have hr : a ∈ Set.Icc (rootPolynomial b c d 0) (rootPolynomial b c d 1) := by
    simpa [rootPolynomial] using (show a ∈ Set.Icc (0 : ℝ) (b + 2*c + 3*d) from ⟨ha.le, ht.le⟩)
  obtain ⟨β, hβ, he⟩ := intermediate_value_Icc (by norm_num : (0 : ℝ) ≤ 1) hc.continuousOn hr
  refine ⟨β, lt_of_le_of_ne hβ.1 ?_, hβ.2, he⟩
  intro hz
  rw [← hz] at he
  simp [rootPolynomial] at he
  linarith

def optimalBeta (a b c d : ℝ) : ℝ :=
  if h : 0 < a ∧ a < b + 2*c + 3*d then Classical.choose (root_exists h.1 h.2) else 1

theorem optimalBeta_spec {a b c d : ℝ} (ha : 0 < a) :
    0 < optimalBeta a b c d ∧ optimalBeta a b c d ≤ 1 ∧
    (optimalBeta a b c d = 1 ∧ b + 2*c + 3*d ≤ a ∨
      rootPolynomial b c d (optimalBeta a b c d) = a) := by
  unfold optimalBeta
  split_ifs with h
  · obtain ⟨hb, hb1, he⟩ := Classical.choose_spec (root_exists h.1 h.2)
    exact ⟨hb, hb1, Or.inr he⟩
  · exact ⟨by norm_num, le_refl _, Or.inl ⟨rfl, le_of_not_gt (fun ht => h ⟨ha, ht⟩)⟩⟩

theorem scalarBound_root_minimum {a b c d r β : ℝ}
    (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (hr : 0 < r) (hβ : 0 < β)
    (he : rootPolynomial b c d r = a) :
    scalarBound a b c d r ≤ scalarBound a b c d β := by
  have hid : scalarBound a b c d β - scalarBound a b c d r =
      (β-r)^2/β * (b + c*(β+2*r) + d*(β^2+2*β*r+3*r^2)) := by
    rw [← he]
    unfold scalarBound rootPolynomial
    field_simp
    ring
  have hn : 0 ≤ (β-r)^2/β * (b + c*(β+2*r) + d*(β^2+2*β*r+3*r^2)) := by positivity
  linarith

theorem scalarBound_endpoint_minimum {a b c d β : ℝ}
    (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (hβ : 0 < β) (hβ1 : β ≤ 1)
    (ha : b + 2*c + 3*d ≤ a) : scalarBound a b c d 1 ≤ scalarBound a b c d β := by
  have hm := scalarBound_root_minimum hb hc hd (by norm_num : (0 : ℝ) < 1) hβ
    (show rootPolynomial b c d 1 = b + 2*c + 3*d by simp [rootPolynomial])
  have hi : 0 ≤ 1/β-1 := by rw [sub_nonneg]; exact (one_le_div hβ).mpr hβ1
  have he : scalarBound a b c d β - scalarBound a b c d 1 =
      scalarBound (b+2*c+3*d) b c d β - scalarBound (b+2*c+3*d) b c d 1 +
      (a-(b+2*c+3*d))*(1/β-1) := by unfold scalarBound; ring
  have hn := mul_nonneg (sub_nonneg.mpr ha) hi
  linarith

theorem optimalBeta_minimizes {a b c d : ℝ} (ha : 0 < a)
    (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) {β : ℝ} (hβ : 0 < β) (hβ1 : β ≤ 1) :
    scalarBound a b c d (optimalBeta a b c d) ≤ scalarBound a b c d β := by
  rcases optimalBeta_spec ha with ⟨hr, _, he | he⟩
  · rw [he.1]
    exact scalarBound_endpoint_minimum hb hc hd hβ hβ1 he.2
  · exact scalarBound_root_minimum hb hc hd hr hβ he

def Profile.optimalBeta (P : Profile) (T : ℕ) : ℝ :=
  LongTails.optimalBeta (4*P.a/T) (4*(P.h/T+P.s₁)) (4*P.s₂) (4*P.s₃)

theorem Profile.bound_eq_scalar (P : Profile) (T : ℕ) (β : ℝ) :
    P.bound T β = scalarBound (4*P.a/T) (4*(P.h/T+P.s₁)) (4*P.s₂) (4*P.s₃) β := by
  unfold bound scalarBound
  ring

theorem Profile.optimalBeta_minimizes (P : Profile) {T : ℕ} (hT : 0 < T)
    (ha : 0 < P.a) {β : ℝ} (hβ : 0 < β) (hβ1 : β ≤ 1) :
    P.bound T (P.optimalBeta T) ≤ P.bound T β := by
  have ht : (0 : ℝ) < T := by exact_mod_cast hT
  rw [P.bound_eq_scalar, P.bound_eq_scalar]
  exact LongTails.optimalBeta_minimizes (by positivity)
    (by have := P.h_nonneg; have := P.s₁_nonneg; positivity)
    (by linarith [P.s₂_nonneg]) (by linarith [P.s₃_nonneg]) hβ hβ1

theorem Profile.optimalBeta_valid (P : Profile) {T : ℕ} (hT : 0 < T) (ha : 0 < P.a) :
    0 < P.optimalBeta T ∧ P.optimalBeta T ≤ 1 := by
  have ht : (0 : ℝ) < T := by exact_mod_cast hT
  exact (optimalBeta_spec (a := 4*P.a/T) (b := 4*(P.h/T+P.s₁))
    (c := 4*P.s₂) (d := 4*P.s₃) (by positivity)).imp_right And.left

theorem Profile.feasible_budget_exists (P : Profile) (ha : 0 < P.a) {ε : ℝ} (hε : 0 < ε) :
    ∃ T : ℕ, 0 < T ∧ P.bound T (P.optimalBeta T) ≤ ε ^ 2 := by
  exact ⟨P.balancedBudget ε, P.balancedBudget_pos ε,
    (P.optimalBeta_minimizes (P.balancedBudget_pos ε) ha
      (P.balancedBeta_pos ε) (P.balancedBeta_le_one ε)).trans (P.bound_at_balancedBudget hε)⟩

/-- Smallest positive horizon certified by the exact scalar bound over all
constant momenta in (0,1], with the safe proportionality constant g fixed.
The fallback handles degenerate inputs; optimality is stated for a > 0. -/
def Profile.optimalBudget (P : Profile) (ε : ℝ) : ℕ :=
  if h : 0 < P.a ∧ 0 < ε then Nat.find (P.feasible_budget_exists h.1 h.2)
  else P.balancedBudget ε

theorem Profile.optimalBudget_spec (P : Profile) (ha : 0 < P.a) {ε : ℝ} (hε : 0 < ε) :
    0 < P.optimalBudget ε ∧
    P.bound (P.optimalBudget ε) (P.optimalBeta (P.optimalBudget ε)) ≤ ε ^ 2 := by
  simp only [optimalBudget, dite_eq_left (show 0 < P.a ∧ 0 < ε from ⟨ha, hε⟩)]
  exact Nat.find_spec (P.feasible_budget_exists ha hε)

theorem Profile.optimalBudget_minimal (P : Profile) (ha : 0 < P.a) {ε : ℝ} (hε : 0 < ε)
    {T : ℕ} (hT : 0 < T) {β : ℝ} (hβ : 0 < β) (hβ1 : β ≤ 1)
    (hbound : P.bound T β ≤ ε ^ 2) : P.optimalBudget ε ≤ T := by
  simp only [optimalBudget, dite_eq_left (show 0 < P.a ∧ 0 < ε from ⟨ha, hε⟩)]
  exact Nat.find_min' (P.feasible_budget_exists ha hε)
    ⟨hT, (P.optimalBeta_minimizes hT ha hβ hβ1).trans hbound⟩

theorem Profile.optimalBudget_le_balanced (P : Profile) (ha : 0 < P.a) {ε : ℝ} (hε : 0 < ε) :
    P.optimalBudget ε ≤ P.balancedBudget ε :=
  P.optimalBudget_minimal ha hε (P.balancedBudget_pos ε) (P.balancedBeta_pos ε)
    (P.balancedBeta_le_one ε) (P.bound_at_balancedBudget hε)

theorem Profile.bound_le_collapsed (P : Profile) {T : ℕ} (hT : 0 < T)
    {β : ℝ} (hβ : 0 < β) (hβ1 : β ≤ 1) :
    P.bound T β ≤ 4*(P.a+P.h)/(β*T) + 4*β*(P.s₁+P.s₂+P.s₃) := by
  have ht : (0 : ℝ) < T := by exact_mod_cast hT
  have hb2 : β^2 ≤ 1 := pow_le_one₀ hβ.le hβ1
  have hb2' : β^2 ≤ β := by nlinarith
  have hb3 : β^3 ≤ β := by nlinarith [mul_le_mul_of_nonneg_left hb2 hβ.le]
  have hi : P.a/(β*T)+P.h*β/T ≤ (P.a+P.h)/(β*T) := by
    apply (le_div_iff₀ (mul_pos hβ ht)).mpr
    field_simp
    nlinarith [mul_le_mul_of_nonneg_left hb2 P.h_nonneg]
  have h₂ := mul_le_mul_of_nonneg_left hb2' P.s₂_nonneg
  have h₃ := mul_le_mul_of_nonneg_left hb3 P.s₃_nonneg
  unfold bound
  simp only [mul_div_assoc] at hi ⊢
  linarith only [hi, h₂, h₃]

def Profile.collapsedBudget (P : Profile) (ε : ℝ) : ℕ :=
  accuracyBudget (P.a+P.h) 1 (P.s₁+P.s₂+P.s₃) ε

theorem Profile.bound_at_collapsedBudget (P : Profile) {ε : ℝ} (hε : 0 < ε) :
    P.bound (P.collapsedBudget ε) (accuracyBeta (P.s₁+P.s₂+P.s₃) ε) ≤ ε ^ 2 := by
  unfold collapsedBudget
  have hs : 0 ≤ P.s₁+P.s₂+P.s₃ := by linarith [P.s₁_nonneg, P.s₂_nonneg, P.s₃_nonneg]
  have hb := accuracyBeta_pos hs hε
  have hi := P.bound_le_collapsed (accuracyBudget_pos (P.a+P.h) 1 (P.s₁+P.s₂+P.s₃) ε)
    hb (accuracyBeta_le_one hs hε)
  apply bound_at_accuracyBudget (g := 1) (V := P.a+P.h)
    (noise := accuracyBeta (P.s₁+P.s₂+P.s₃) ε ^ 2 * (P.s₁+P.s₂+P.s₃))
    (by norm_num) hs hε (le_refl _) (by ring_nf; exact le_refl _) ?_
  convert hi using 1 <;> (try field_simp) <;> ring

theorem Profile.optimalBudget_le_collapsed (P : Profile) (ha : 0 < P.a) {ε : ℝ} (hε : 0 < ε) :
    P.optimalBudget ε ≤ P.collapsedBudget ε := by
  have hs : 0 ≤ P.s₁+P.s₂+P.s₃ := by linarith [P.s₁_nonneg, P.s₂_nonneg, P.s₃_nonneg]
  exact P.optimalBudget_minimal ha hε (accuracyBudget_pos _ _ _ _)
    (accuracyBeta_pos hs hε) (accuracyBeta_le_one hs hε) (P.bound_at_collapsedBudget hε)

/-- A limitation of this upper-bound certificate, not a lower bound for the
algorithm: positive linear noise forces epsilon^(-4) under constant tuning. -/
theorem Profile.certificate_barrier (P : Profile) {T : ℕ} (hT : 0 < T)
    {β ε : ℝ} (hβ : 0 < β) (hbound : P.bound T β ≤ ε ^ 2) :
    64 * P.a * P.s₁ ≤ ε ^ 4 * T := by
  have ht : (0 : ℝ) < T := by exact_mod_cast hT
  let x := 4 * P.a / (β*T)
  let y := 4 * P.s₁ * β
  have hx : 0 ≤ x := div_nonneg (mul_nonneg (by norm_num) P.a_nonneg) (by positivity)
  have hy : 0 ≤ y := mul_nonneg (mul_nonneg (by norm_num) P.s₁_nonneg) hβ.le
  have hr : 0 ≤ P.h*β/T + P.s₂*β^2 + P.s₃*β^3 := by
    have := P.h_nonneg
    have := P.s₂_nonneg
    have := P.s₃_nonneg
    positivity
  have hsum : x+y ≤ ε^2 := by
    dsimp [x, y]
    unfold bound at hbound
    simp only [mul_div_assoc] at hr hbound ⊢
    linarith only [hbound, hr]
  have hs := pow_le_pow_left₀ (add_nonneg hx hy) hsum 2
  have hp : 4*x*y ≤ ε^4 := by nlinarith only [sq_nonneg (x-y), hs]
  have he : (4*x*y)*T = 64*P.a*P.s₁ := by dsimp [x, y]; field_simp; ring
  have hh := mul_le_mul_of_nonneg_right hp ht.le
  rw [he] at hh
  exact hh

end EF21Muon.LongTails
