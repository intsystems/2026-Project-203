Dear Reviewer, thank you for your questions.

1. Re covered experiments: As submitted, no EF21 run is fully covered: Theorem 5 requires gradient-dependent or constant steps, but our runs use cosine schedules. Based on your review, we extended EF21-Muon(U)Sign convergence to constant/cosine rates and fixed EMA/Nesterov. This covers Tables 1–2 for exact polar updates, but not nanoGPT's momentum warmup (Table 3). We will add a coverage table.
2. Re Table 2: The submitted logs (5 seeds) give MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%, agreeing with A8. We corrected Table 2, its comparison and Table 1.
3. Re our contribution: SignMuon is concurrent work: Mishra et al. is a May 2026 preprint, developed independently. The EF21 rates are inherited; Theorems 1–4 are our contribution. We also proved SignMuon convergence under uniform alignment between signed updates and momentum.
4. Re nanoGPT: nanoGPT applies the optimizer after full-precision gradient averaging, so it tests centralized optimizers; Sec. 5.3 will state this. Federated CNN2 tests distributed compression: with our new heterogeneous clients, even at this scale SignMuon falls to 42.96%, while EF21-Muon(U)Sign keep about 84% (our response to 7S1b, Q2).
5. Re presentation: We will simplify the presentation as suggested. Sec. 4.4 will say that EF21-MuonSign updates the server model X, workers compute gradients at its broadcast model W, and the guarantee bounds X.

We hope these answers warrant reassessing the score.
