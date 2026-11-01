We answer the technical questions below.

1. Re our theory: The noise assumption is Assumption 3 (A.10); we will move it to the main text. We acknowledge that layer-wise smoothness forces separable layers; the EF21 bounds retain it, while our new SignMuon theorem allows interacting layers. The EF21 theorems assume exact polar factors; we have not bounded the Newton–Schulz error.
2. Re the two models: Theorem 5 and Corollary 1 bound the gradient at EF21-MuonSign's server model X, but gradients are computed at broadcast model W. Table 3 reports losses 3.3213 for W and 5.5198 for X in the same run.
3. Re our contribution: We agree that the EF21 results instantiate EF21-Muon and will cite Modoranu et al. SignSGD uses EMA in our comparison; a matched-gradient-budget SSVR experiment remains open, and we will cite Jiang et al.
4. Re heterogeneous data: We will restrict Sec. 5.2 to IID clients. With 11 Dirichlet(0.1) clients, only EF21-Muon(U)Sign retain over 80% (7S1b, Q2).
5. Re suggestions: We will add your method matrix, move Sec. 4.1 to Sec. 5 and restrict "unrepairable" to the analyzed recursion. We also proved SignMuon convergence under uniform alignment of signed updates with momentum.
6. Re measurements: A.10 takes the zero step at zero input. The 0.0021 nanoGPT gap needs more seeds. By A5, SignMuon reaches 90% in about 128 s vs 238–260 s for SignSGD, Adam and SGD. CNN2's 1.087 bits/parameter includes auxiliaries and scales. MuonSign and SGD reach 80% in all 5 IID seeds.
