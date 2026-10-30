Thank you for your questions.

1. Re why SignMuon works: After your review, we proved SignMuon convergence when signed updates remain uniformly aligned with momentum, under smoothness and unbiased bounded-variance noise. Layers may interact; network alignment is unmeasured.
2. Re heterogeneous clients: We tuned LRs on federated CNN2/CIFAR-10: 11 Dirichlet(0.1) clients, 2000 rounds, no local updates. Rates use validation seed 0; momentum 0.9; mean±SD over 5 final seeds:

|Method|Accuracy (%)|
|---|---:|
|**EF21-MuonSign**|**84.03±0.52**|
|**EF21-MuonUSign**|**83.84±0.32**|
|SGD|79.71±2.03|
|Adam|75.16±1.07|
|EF21-SignMuon|49.73±6.69|
|Muon|46.02±8.55|
|MuonUSign|44.23±3.28|
|SignMuon|42.96±5.94|
|MuonSign|40.75±0.98|
|SignSGD|36.24±4.64|

**EF21-Muon(U)Sign are the only methods above 80% after tuning.**

3. Re our analysis: After your review, we proved EF21-Muon(U)Sign convergence for constant/cosine rates and fixed EMA/Nesterov, and verified the proofs in Lean. These bounds control expected gradient norms.
4. Re choosing a method: SignMuon is effective on IID CNN2/CIFAR-10; for label skew, use EF21-MuonUSign for uplink-only compression or EF21-MuonSign for both links. Messages use 1.087 bits/parameter, including auxiliaries and scales.
5. Re nanoGPT: nanoGPT validation losses are 3.2881 for SignMuon and 3.4049 for SignSGD (1 run each). The 0.0021 gap to EF21-SignMuon cannot rank methods.

If these clarifications address your concerns, please consider raising your score.
