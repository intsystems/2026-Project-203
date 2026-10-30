Thank you for your questions.

1. Re why SignMuon works: Based on your review, we proved SignMuon convergence for interacting layers under joint smoothness, unbiased bounded-variance noise and uniform alignment of signed updates with momentum. Testing network alignment remains open.
2. Re heterogeneous clients: We tuned rates for CNN2/CIFAR-10: 11 Dirichlet(0.1) clients, 2000 rounds, no local updates. Momentum 0.9; validation seed 0; mean±SD over 5 final seeds:

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

Only EF21-MuonSign and EF21-MuonUSign exceed 80%.

3. Re our analysis: Based on your review, we proved expected gradient-norm bounds for EF21-Muon(U)Sign with constant/cosine rates and fixed EMA/Nesterov, verified in Lean.
4. Re choosing a method: SignMuon suits centralized deep learning and IID federation. For heterogeneous federated learning, EF21-MuonUSign compresses uploads and EF21-MuonSign both directions. EF messages use 1.087 bits/parameter, including auxiliaries and scales.
5. Re nanoGPT: SignMuon achieves loss 3.2881 versus SignSGD 3.4049 (1 nanoGPT run each). Its 0.0021 gap to EF21-SignMuon needs more seeds to establish an ordering.

If these clarifications address your concerns, please consider raising your score.
