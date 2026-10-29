Thank you for the theory and practical questions.

1. Re why SignMuon works: Under smoothness and unbiased bounded-variance noise, SignMuon converges if signing retains a fixed positive fraction of the update’s alignment with momentum. It allows interacting layers; the condition is unmeasured in our networks.
2. Re heterogeneous clients: With 11 clients and Dirichlet(0.1) labels, reusing IID rates gives EF21-MuonSign 83.23% versus SignMuon 35.92% (5 seeds). The ordering changes; retuning is pending. Clients take no local updates.
3. Re our analysis: The base rates follow EF21. We generalized its analysis to EMA and Nesterov momentum with deterministic learning rates; the tracking and convergence proofs are verified in Lean. We prove all 3 sign placements can point uphill and even error feedback after orthogonalization can diverge.
4. Re choosing a method: SignMuon reaches 85.72%, versus Muon 85.98% and SignSGD 81.44% in 5 submitted runs. For gradient-level error feedback, choose EF21-MuonUSign for the uplink or EF21-MuonSign for both links. CNN2 costs 1.087 bits/parameter including auxiliaries. EF adds scales, buffers and a server model for two-sided compression. The polar runs on clients for SignMuon and the server for gradient EF.
5. Re nanoGPT: nanoGPT losses are 3.2881 for SignMuon and 3.4049 for SignSGD (1 run/method). The 0.0021 gap to EF21-SignMuon does not establish a ranking.

If these clarifications address your concerns, please consider raising your score.
