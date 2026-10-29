Thank you for the detailed checks.

1. Re our theory: We generalized EF21 theorems to fixed EMA and Nesterov momentum with deterministic learning rates. The core proofs are verified in Lean. The bounds retain separable layers and account for unbiased bounded-variance noise and orthogonalization error.
2. Re the two models: EF21-MuonSign queries broadcast W; its theorem concerns server X. Their losses in the same nanoGPT run are 3.3213 and 5.5198.
3. Re our contribution: We credit EF21, Modoranu and concurrent SignMuon. Our examples prove ascent for all 3 sign placements and divergence of output-level error feedback. SignSGD uses EMA. SSVR needs matched gradient-query budgets and remains untested; we will qualify the geometry attribution.
4. Re the experiments: Dirichlet(0.1) transfer at IID rates gives EF21-MuonSign 83.23% versus SignMuon 35.92%; retuning is pending. Clients take no local updates. nanoGPT tests centralized updates after full-precision averaging.
5. Re the presentation: We will add a coverage table, clarify assumptions and move implementation to setup. SignMuon’s guarantee permits interacting layers when signed updates stay sufficiently aligned with momentum; this condition is unmeasured. Theorem 4 does not rule out decreasing steps.
6. Re details: We will define polar(0)=0 and count EF scales. nanoGPT has 1 run/method; its 0.0021 gap does not establish a ranking. A5’s 16.6 s/epoch is not network timing. MuonSign and SGD reach the target in 5/5 seeds.
