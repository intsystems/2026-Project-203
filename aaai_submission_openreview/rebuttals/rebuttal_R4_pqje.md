Dear Reviewer, thank you for the careful review.

1. Re covered experiments: As submitted, Theorem 5 covers no EF21 run: it requires gradient-dependent or constant steps, while our runs use cosine schedules. Following your review, we extended EF21-Muon(U)Sign convergence to constant/cosine rates and fixed EMA/Nesterov, covering Tables 1–2 for exact polar updates but not nanoGPT's momentum warmup. We will add a coverage table.
2. Re Table 2: The submitted logs (5 seeds) give MuonSign 82.94% versus SignSGD 81.44%, as in A8. We corrected Table 2, its comparison and Table 1.
3. Re our contribution: We respectfully disagree that our contribution is limited. SignMuon is concurrent: Mishra et al. is a May 2026 preprint developed independently. The EF21 rates are inherited, but Theorems 1–4 are new. We also proved SignMuon convergence under uniform alignment of signed updates with momentum.
4. Re nanoGPT: nanoGPT applies the optimizer after full-precision gradient averaging, so it tests centralized optimizers; Sec. 5.3 will state this. Federated CNN2 tests distributed compression: with our new heterogeneous clients, even at this scale SignMuon falls to 42.96%, while EF21-Muon(U)Sign keep about 84% (see 7S1b, Q2).
5. Re presentation: We will simplify the presentation as suggested. Sec. 4.4 will say that EF21-MuonSign updates the server model X, workers compute gradients at its broadcast model W, and the guarantee bounds X.

We hope these answers warrant reconsidering the score.
