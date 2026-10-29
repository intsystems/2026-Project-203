Thank you for the detailed checks.

1. Re our theory: After submission, we extended EF21 to fixed EMA/Nesterov and deterministic learning rates, with core proofs verified in Lean. The bounds retain separable layers and include unbiased bounded-variance noise and orthogonalization error.
2. Re the two models: EF21-MuonSign queries broadcast W; its theorem concerns server X. Their losses in the same nanoGPT run are 3.3213 and 5.5198.
3. Re our contribution: We prove all 3 sign placements can ascend and output-level error feedback can diverge. Heterogeneous tests show why EF placement matters. SignSGD already uses EMA. SSVR remains untested and needs matched gradient-query budgets; we will qualify the geometry attribution.
4. Re heterogeneity: After submission, we added Dirichlet(0.1) tests: gradient-side EF leads at IID-selected rates (table in our reply to 7S1b, Q2). Clients take no local updates; retuning is pending. nanoGPT averages full-precision gradients.
5. Re the presentation: We will add the method/coverage table and move implementation to setup. Our post-submission SignMuon theorem allows interacting layers under a uniform positive alignment with momentum; this is unmeasured. Our new output-EF example defeats power decay and budget-tuned constant parameters.
6. Re details: We will define polar(0)=0 and count EF scales. nanoGPT has 1 run/method; its 0.0021 gap cannot rank methods. A5’s 16.6 s/epoch is not network timing. MuonSign and SGD reach 80% in 5/5 seeds.
