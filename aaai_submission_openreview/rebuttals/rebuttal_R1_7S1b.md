Thank you for your questions.

1. Re why SignMuon works: Since submission, we proved SignMuon converges under smoothness, unbiased bounded-variance noise and sufficient alignment of signed updates with momentum. It allows interacting layers; network tests remain open.
2. Re heterogeneous clients: Following your review, we tested CNN2/CIFAR-10: 11 clients, Dirichlet(0.1), 2000 rounds, no local updates, IID-selected rates. Mean±SD, 5 seeds:

|Method|Accuracy (%)|
|---|---:|
|EF21-MuonSign|83.23±0.36|
|EF21-MuonUSign|82.99±0.46|
|SGD|81.03±0.62|
|Adam|71.17±7.16|
|Muon|46.02±8.55|
|EF21-SignMuon|45.46±7.66|
|SignMuon|35.92±8.49|
|SignSGD|26.42±3.23|
|MuonUSign|26.21±2.75|
|MuonSign|25.63±3.98|

Gradient-side error feedback leads. Retuning is pending.

3. Re our analysis: In response to your review, we extended EF21 to fixed EMA/Nesterov and deterministic rates, retaining separable layers. Its core proofs are verified in Lean. Our submitted proofs show 3 sign placements can ascend and output EF can diverge.
4. Re choosing a method: Use SignMuon for IID data (85.72% vs Muon 85.98%); under heterogeneity, use EF21-MuonUSign for uplink-only compression or EF21-MuonSign for both links. CNN2 uses 1.087 bits/parameter including auxiliaries; EF adds scales and buffers.
5. Re nanoGPT: SignMuon/SignSGD losses are 3.2881/3.4049 (1 run each). A 0.0021 gap to EF21-SignMuon cannot establish a ranking.

If these clarifications address your concerns, please consider raising your score.
