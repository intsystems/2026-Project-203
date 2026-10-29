Thank you for the detailed technical checks.

Re guarantees: A.10 states unbiased bounded-variance noise; we will add it to the main statement. One max-spectral norm on all layers removes separability. Our new deterministic-schedule bound allows fixed EMA/Nesterov and either EF21 downlink, but not finite polar solvers, changing momentum or auxiliary Adam.
Re X/W: EF21-MuonSign losses are 5.5198 at server X and 3.3213 at broadcast W. The guarantee concerns X; success at W does not validate it.
Re attribution: We credit EF21-Muon and Modoranu et al.; the submitted novelty is the negative sign-placement/output-EF analysis. SignSGD uses momentum; SSVR is untested. We will cite Jiang et al. and remove exclusive attribution to geometry.
Re scope: Federation uses 11 IID clients, no local updates. nanoGPT averages full-precision gradients; no distributed one-bit gains are demonstrated.
Re clarity: We will add the method/coverage table and move implementation details to the experiments. A sufficient SignMuon condition is $[U^\top\mathrm{sign}(UV^\top)V]_{jj}\ge\delta>0$ for $M=U\Sigma V^\top$; it implies alignment $\ge\delta\|M\|_*$. “Unrepairable” will be limited to the specified output-EF counterexample; decreasing schedules and other repairs are not ruled out.
Re details: We will define polar(0)=0. The 0.0021 gap is inconclusive. A5 reports seconds/epoch; we drop the broad 2x claim. Payloads include FP32 auxiliaries/EF scales. Corrected MuonSign/SGD runs reach 80% in all 5 seeds.
