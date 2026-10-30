Thank you for the technical checks.

1. Re our theory: After submission, we extended EF21-Muon(U)Sign convergence to fixed EMA/Nesterov and deterministic rates, with proofs verified in Lean. The bounds use separable layers and unbiased bounded-variance noise, and quantify numerical orthogonalization error.
2. Re the two models: Theorem 5 and Corollary 1 in the submitted supplement bound the expected squared nuclear gradient norm at EF21-MuonSign’s server model X. Gradients are computed at broadcast model W; Table 3 reports losses 3.3213 for W and 5.5198 for X in the same run.
3. Re our contribution: Our comparison isolates sign and error-feedback placement. We prove all 3 plain placements can ascend and establish gradient-side EF convergence. SignSGD uses EMA in our comparison; a matched-gradient-budget SSVR experiment remains open.
4. Re heterogeneous data: We completed learning-rate tuning on CNN2/CIFAR-10 with 11 Dirichlet(0.1) clients, 2000 rounds, no local updates and 5 final seeds. See our response to 7S1b, Q2.
5. Re the SignMuon condition: We also proved SignMuon convergence for interacting layers under joint smoothness, bounded noise and uniform alignment of signed updates with momentum.
6. Re the reported measurements: The 0.0021 nanoGPT gap needs more seeds (1 run/method). CNN2’s 1.087 bits/parameter includes auxiliaries and EF scales. A5 gives SignMuon 16.6 s/epoch on centralized ResNet18/CIFAR-10. MuonSign and SGD reach 80% in all 5 IID seeds.
