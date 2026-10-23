import EF21Muon.Compression.RandomTies
import EF21Muon.Nesterov.AveragedConvergence
import EF21Muon.Nesterov.GeneralizedDrift

noncomputable section
namespace EF21Muon.RandomTies
open MeasureTheory
open LongTails.Algorithms.MatrixPolar
open EF21Muon.Distributed EF21Muon.Stochastic

variable {N m n : ℕ}

/-- Recompute the new worker memories using each worker's own fresh tie bits. -/
def refresh (b : Fin N → Fin m × Fin n → Bool)
    (oldG v : Fin N → Space m n) (s : LayerState N m n) : LayerState N m n :=
  (s.x,s.w,s.M,fun j => oldG j+compressor (b j) (v j-oldG j))

theorem measurable_refresh {Z : Type*} [MeasurableSpace Z]
    (b : Fin N → Fin m × Fin n → Bool)
    {oldG v : Z → Fin N → Space m n} {s : Z → LayerState N m n}
    (hg : Measurable oldG) (hv : Measurable v) (hs : Measurable s) :
    Measurable (fun z => refresh b (oldG z) (v z) (s z)) :=
  hs.fst.prodMk (hs.snd.fst.prodMk (hs.snd.snd.fst.prodMk (Measurable.of_eval
    fun j => ((measurable_pi_apply j).comp hg).add ((measurable_compressor _).comp
      (((measurable_pi_apply j).comp hv).sub ((measurable_pi_apply j).comp hg))))))

theorem refresh_norm (hm : 0 < m) (hn : 0 < n)
    (b : Fin N → Fin m × Fin n → Bool) (oldG v : Fin N → Space m n)
    (s : LayerState N m n) (hG : ∀ j, s.G j = oldG j+signCompressor (v j-oldG j))
    (j : Fin N) : ‖v j-(refresh b oldG v s).G j‖ = ‖v j-s.G j‖ := by
  change ‖v j-(oldG j+compressor (b j) (v j-oldG j))‖ = _
  rw [hG j]
  exact residual_norm hm hn (b j) _ _

theorem refresh_energy (hm : 0 < m) (hn : 0 < n) (c : Parameters) (ell2 : ℝ)
    (b : Fin N → Fin m × Fin n → Bool) (oldG : Fin N → Space m n)
    (s : LayerState N m n) (hG : ∀ j, s.G j = oldG j+signCompressor (s.M j-oldG j))
    (g : Space m n) (localG : Fin N → Space m n) :
    energy c ell2 (refresh b oldG s.M s) g localG = energy c ell2 s g localG := by
  have he : compressionError (refresh b oldG s.M s) = compressionError s := by
    unfold compressionError
    congr 1
    funext j
    change ‖s.M j-(refresh b oldG s.M s).G j‖^2 = _
    rw [refresh_norm hm hn b oldG s.M s hG]
  change c.γ*(_+_+c.uplinkWeight*compressionError (refresh b oldG s.M s)+_) = _
  rw [he]
  rfl

theorem refresh_radiusEnergy (hm : 0 < m) (hn : 0 < n) (c : Parameters)
    (b : Fin N → Fin m × Fin n → Bool) (oldG : Fin N → Space m n)
    (s : LayerState N m n) (hG : ∀ j, s.G j = oldG j+signCompressor (s.M j-oldG j))
    (g Z : Space m n) (localG : Fin N → Space m n) :
    radiusEnergy c (refresh b oldG s.M s) g Z localG = radiusEnergy c s g Z localG := by
  have he : firstCompression (refresh b oldG s.M s) = firstCompression s := by
    unfold firstCompression
    congr 1
    funext j
    change ‖s.M j-(refresh b oldG s.M s).G j‖ = _
    exact refresh_norm hm hn b oldG s.M s hG j
  change _+_+radiusSWeight c*firstCompression (refresh b oldG s.M s)+_ = _
  rw [he]
  rfl

def nestRefresh (c : Parameters) (b : Fin N → Fin m × Fin n → Bool)
    (oldG : Fin N → Space m n) (s : Nesterov.Layer N m n) : Nesterov.Layer N m n :=
  (refresh b oldG (Nesterov.target c s) s.1, s.2)

theorem nestRefresh_energy (hm : 0 < m) (hn : 0 < n) (c : Parameters) (ell2 : ℝ)
    (b : Fin N → Fin m × Fin n → Bool) (oldG : Fin N → Space m n)
    (s : Nesterov.Layer N m n)
    (hG : ∀ j, s.1.G j = oldG j+signCompressor (Nesterov.target c s j-oldG j))
    (g : Space m n) (localG : Fin N → Space m n) :
    Nesterov.Averaged.energy c ell2 (nestRefresh c b oldG s) g localG =
      Nesterov.Averaged.energy c ell2 s g localG := by
  have he : Nesterov.compression c (nestRefresh c b oldG s) = Nesterov.compression c s := by
    unfold Nesterov.compression
    congr 1
    funext j
    change ‖Nesterov.target c s j-(refresh b oldG (Nesterov.target c s) s.1).G j‖^2 = _
    rw [refresh_norm hm hn b oldG (Nesterov.target c s) s.1 hG]
  unfold Nesterov.Averaged.energy
  rw [he]
  rfl

theorem nestRefresh_radiusEnergy (hm : 0 < m) (hn : 0 < n) (c : Parameters)
    (b : Fin N → Fin m × Fin n → Bool) (oldG : Fin N → Space m n)
    (s : Nesterov.Layer N m n)
    (hG : ∀ j, s.1.G j = oldG j+signCompressor (Nesterov.target c s j-oldG j))
    (g Z : Space m n) (localG : Fin N → Space m n) :
    Nesterov.radiusEnergy c (nestRefresh c b oldG s) g Z localG =
      Nesterov.radiusEnergy c s g Z localG := by
  have he : Nesterov.firstCompression c (nestRefresh c b oldG s) = Nesterov.firstCompression c s := by
    unfold Nesterov.firstCompression
    congr 1
    funext j
    exact refresh_norm hm hn b oldG (Nesterov.target c s) s.1 hG j
  unfold Nesterov.radiusEnergy
  rw [he]
  rfl

theorem measurable_nestRefresh {Z : Type*} [MeasurableSpace Z]
    (c : Parameters) (b : Fin N → Fin m × Fin n → Bool)
    {oldG : Z → Fin N → Space m n} {s : Z → Nesterov.Layer N m n}
    (hg : Measurable oldG) (hs : Measurable s) :
    Measurable (fun z => nestRefresh c b (oldG z) (s z)) := by
  have hv : Measurable (fun z => Nesterov.target c (s z)) := Measurable.of_eval fun j =>
    (((measurable_pi_apply j).comp hs.fst.snd.snd.fst).const_smul (2-c.β)).sub
      (((measurable_pi_apply j).comp hs.snd).const_smul (1-c.β))
  exact (measurable_refresh b hg hv hs.fst).prodMk hs.snd

end EF21Muon.RandomTies
