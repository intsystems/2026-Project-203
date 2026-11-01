# Confidential summary of responses to the main reviewer concerns

Dear AC/SPC,

We make significant progress toward state-of-the-art optimization with sign compression, outperforming SignSGD in centralized and federated tests. Theorems 1–4 are original contributions; EF21 rates build on prior work (Tx34, Q1; pqje). SignMuon was developed independently of concurrent work by Mishra et al. (May 2026).

For Tx34 (Q2), an 8×8 convex quadratic family keeps exact-polar EF21-SignMuon's gradient norm ≥1 under horizon-tuned constant rates and fixed momentum, at fixed smoothness and initial gap. This rules out uniform vanishing rates. We also extended Theorems 1–3 to smooth, lower-bounded objectives (D4r2).

For 7S1b (Q2), we tuned 10 methods on CNN2/CIFAR-10 with 11 Dirichlet(0.1) clients, 5 final seeds and no local updates. EF21-MuonSign and EF21-MuonUSign retain about 84%, while SignMuon falls from 85.72% (IID) to 42.96%, guiding method choice.

For 7S1b (Q1) and D4r2, we proved SignMuon convergence under uniform alignment of signed updates with momentum; testing this in networks remains open.

For Ueq5 and pqje (Q1), we extended EF21 bounds to constant/cosine schedules and fixed EMA/Nesterov, with Lean-verified proofs for separable layers and exact polar updates. Submitted logs confirm A8: MuonSign 82.94% versus SignSGD 81.44% (pqje, Q2). We corrected Tables 1–2 and will correct the rate interpretation; the separately proved squared-gradient guarantee remains valid.

We hope you find the novelty sufficient and the paper timely.
