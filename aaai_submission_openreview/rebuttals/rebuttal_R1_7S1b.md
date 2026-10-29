Thank you for your questions.

1. Re why SignMuon works: After your review, we proved SignMuon convergence when signed updates remain uniformly aligned with momentum, under smoothness and unbiased bounded-variance noise. Layers may interact; network alignment is unmeasured.
2. Re heterogeneous clients: We added federated CNN2/CIFAR-10 tests: 11 clients, Dirichlet(0.1) label splits, 2000 rounds, no local updates. IID-selected rates are reused; mean±SD, 5 seeds:

|Method|Accuracy (%)|
|---|---:|
|**EF21-MuonSign**|**83.23±0.36**|
|**EF21-MuonUSign**|**82.99±0.46**|
|SGD|81.03±0.62|
|Adam|71.17±7.16|
|Muon|46.02±8.55|
|EF21-SignMuon|45.46±7.66|
|SignMuon|35.92±8.49|
|SignSGD|26.42±3.23|
|MuonUSign|26.21±2.75|
|MuonSign|25.63±3.98|

**EF21-Muon(U)Sign are the only Muon variants that work well here (>80%).**

3. Re our analysis: After your review, we extended gradient-side EF21 to fixed EMA/Nesterov and constant/cosine learning rates. Core convergence proofs are verified in Lean; separable layers remain assumed.
4. Re choosing a method: SignMuon is effective on IID CNN2/CIFAR-10; for label skew, use EF21-MuonUSign for uplink-only compression or EF21-MuonSign for both links. Messages use 1.087 bits/parameter, including auxiliaries and scales.
5. Re nanoGPT: nanoGPT validation losses are 3.2881 for SignMuon and 3.4049 for SignSGD (1 run each). The 0.0021 gap to EF21-SignMuon cannot rank methods.

If these clarifications address your concerns, please consider raising your score.
