# Confidential summary of responses to the main reviewer concerns

Dear AC/SPC,
We develop one-bit Muon variants; our counterexamples cover all three sign placements and output feedback (Theorems 1–4). The main concerns are novelty, the theory–practice gap and inconsistent tables. As clarified to Tx34 (Q1) and pqje, the EF21 rates are inherited; our failure results are new. SignMuon was developed independently of concurrent work by Mishra et al. (May 2026).
For Tx34 (Q2), our new 8×8 convex quadratic family keeps exact-polar EF21-SignMuon's gradient norm ≥1 under horizon-tuned constant rates and fixed momentum, with fixed smoothness and initial gap. Such tuning cannot yield a uniform vanishing rate.
For 7S1b (Q2), we tuned all 10 methods on CNN2/CIFAR-10 with 11 Dirichlet(0.1) clients and 5 final seeds, without local updates. EF21-MuonSign and EF21-MuonUSign retain about 84%, while SignMuon drops from 85.72% (IID) to 42.96%, giving concrete guidance under heterogeneity.
For 7S1b (Q1) and D4r2, we proved SignMuon convergence under uniform alignment of signed updates with momentum; testing this in networks remains open.
For Ueq5 and pqje (Q2), submitted logs confirm A8: MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%. We corrected Tables 1–2 and will remove the invalid first-to-second-moment inference. We also extended EF21 bounds to constant/cosine schedules and fixed EMA/Nesterov (pqje, Q1), with Lean-verified proofs; separable layers and exact polar updates remain required.
