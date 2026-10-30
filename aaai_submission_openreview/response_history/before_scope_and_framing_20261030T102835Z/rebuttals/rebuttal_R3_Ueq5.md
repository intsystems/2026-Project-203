Thank you for identifying the reporting and rate issues.

1. Re the inconsistent tables: The submitted federated CNN2/CIFAR-10 logs (11 IID clients, 5 seeds) reproduce A8: MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%. We corrected Table 2 and its reversed comparison in the revision. See our response to 7S1b, Q2 for the new heterogeneous federated results.
2. Re the extended guarantees: Based on your review, we extended EF21-MuonUSign and EF21-MuonSign convergence to fixed EMA/Nesterov and deterministic learning rates, including constant/cosine schedules. We verified the tracking and convergence proofs in Lean. The bounds quantify initialization, stochastic noise and orthogonalization accuracy. These EF21 bounds retain the submitted assumption that layers are separable; our separate SignMuon theorem allows interacting layers under joint smoothness and uniform alignment.
3. Re the two convergence rates: For EF21-MuonUSign and EF21-MuonSign with exact polar updates, our constant/cosine theorem bounds the learning-rate-weighted average expected nuclear gradient norm. Choosing the peak learning rate and fixed momentum for $T$ iterations gives an $O(T^{-1/4})$ rate. We separately proved an $O(T^{-1/2})$ average expected squared nuclear gradient norm bound with gradient-dependent steps.

If these corrections resolve your reservations, please consider raising your score.
