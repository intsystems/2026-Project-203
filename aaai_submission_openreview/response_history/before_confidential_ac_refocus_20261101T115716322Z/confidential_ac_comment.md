# One-bit Muon: why compression and error feedback placement matter

Dear AC/SPC,

Our paper asks how to retain Muon's training performance at roughly one bit per communicated parameter. We compare three ways to combine sign compression with Muon's matrix orthogonalization, plus error feedback, which recovers information lost in compression.

Although sign and orthogonalization each yield descent alone, all three combinations can increase the loss (new Theorems 1-3). Correcting the final update can still fail (Theorem 4); correcting gradients before orthogonalization gives two variants with convergence guarantees. These results identify where a compression remedy works and where it breaks. The EF21 rates build on EF21-Muon; SignMuon was developed independently of concurrent work.

The rebuttal adds practical guidance: on CNN2/CIFAR-10 with different label distributions across 11 clients and no local steps, after tuning all 10 methods and testing 5 seeds, EF21-MuonSign and EF21-MuonUSign retain about 84% accuracy. SignMuon falls from 85.72% with similar client data to 42.96% (7S1b, Q2).

We also prove SignMuon convergence when signed updates stay uniformly aligned with momentum, extend the failure result to constant learning rates chosen for a training budget (Tx34, Q2), and give Lean-verified EF21 proofs for constant/cosine schedules with fixed momentum. Corrected tables are supported by submitted logs.

These results guide the design and choice of compressed Muon methods.
