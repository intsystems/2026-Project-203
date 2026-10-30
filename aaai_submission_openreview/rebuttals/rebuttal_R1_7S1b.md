Dear Reviewer, thank you for your questions.

1. Re why SignMuon works: Based on your review, we proved SignMuon convergence under joint smoothness, bounded noise and uniform alignment of signed updates with momentum, which our counterexample violates. Testing it in networks is open.
2. Re heterogeneous clients: The ordering changes. We tuned rates for CNN2/CIFAR-10 with 11 Dirichlet(0.1) clients and no local updates; mean accuracy over 5 seeds:

|Method|IID|Non-IID|
|---|---:|---:|
|EF21-MuonSign|84.0|84.0|
|EF21-MuonUSign|83.6|83.8|
|SGD|81.7|79.7|
|Adam|77.1|75.2|
|EF21-SignMuon|84.7|49.7|
|Muon|86.0|46.0|
|MuonUSign|84.6|44.2|
|SignMuon|85.7|43.0|
|MuonSign|82.9|40.8|
|SignSGD|81.4|36.2|

Methods that sign or orthogonalize client messages before aggregation lose 35–45 points (our federated Muon averages client polar factors). EF21-Muon(U)Sign orthogonalize at the server, and Theorem 5 assumes no bound on client dissimilarity.
3. Re our analysis: The submitted EF21 rates follow from EF21-Muon once Lemma 5 shows that scaled sign is contractive; Theorems 1–4 are new.
4. Re choosing a method: SignMuon suits centralized training and IID federation; for heterogeneous federation, EF21-MuonUSign compresses uploads and EF21-MuonSign both directions. We will add the cost table.
5. Re nanoGPT: SignMuon reaches loss 3.2881 versus SignSGD 3.4049 (1 run each). Its 0.0021 gap to EF21-SignMuon needs more seeds to establish an ordering.

We hope this addresses your main reservations.
