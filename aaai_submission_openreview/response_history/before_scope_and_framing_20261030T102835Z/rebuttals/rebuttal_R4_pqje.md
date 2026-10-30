Thank you for the questions.

1. Re the extended guarantees: Based on your review, we extended EF21-Muon(U)Sign convergence to constant/cosine rates and fixed EMA/Nesterov, with proofs verified in Lean. The bounds quantify noise and orthogonalization error. We separately proved the squared-gradient rate with gradient-dependent steps.
2. Re Table 2: The submitted CNN2/CIFAR-10 logs (11 IID clients, 5 seeds) give MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%, agreeing with A8. We corrected Table 2 and its comparison.
3. Re our contribution: We explore Muon with sign compression and error feedback. SignMuon leads on centralized CIFAR-10; heterogeneous federation favors EF21-Muon(U)Sign (see our response to 7S1b, Q2). Based on your review, we proved SignMuon convergence under joint smoothness, unbiased bounded-variance noise and uniform alignment between signed updates and momentum.
4. Re nanoGPT: nanoGPT applies the optimizer after full-precision gradient averaging (SignMuon loss 3.2881; SignSGD 3.4049). Federated CNN2 compresses client messages before aggregation. Our extension covers fixed Nesterov; nanoGPT uses momentum warmup.
5. Re the two models: Theorem 5 and Corollary 1 in the submitted supplement bound the expected squared nuclear gradient norm at EF21-MuonSign’s server model X; workers compute gradients at its broadcast model W (Eq. 13).

If these answers and revisions resolve your concerns, please consider raising your score.
