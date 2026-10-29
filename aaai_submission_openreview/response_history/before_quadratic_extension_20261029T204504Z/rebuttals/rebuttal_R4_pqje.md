Thank you for the detailed questions.

1. Re our guarantees: In response to your review, we generalized EF21 convergence to fixed EMA and Nesterov momentum with deterministic learning rates, including constant and cosine schedules, and verified the core proofs in Lean. The bounds retain separable layers; numerical orthogonalization needs an error bound. nanoGPT varies momentum, and the squared-gradient rate needs gradient-dependent steps.
2. Re Table 2: The submitted 5-seed logs give MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%, agreeing with A8. We will correct Table 2 and the reversed comparison. SignMuon reaches 85.72% versus Muon 85.98%.
3. Re our contribution: After submission, we proved SignMuon converges under smoothness and unbiased bounded-variance noise when signing preserves enough alignment with momentum. This permits interacting layers; the condition is unmeasured in networks. We also added heterogeneous tests: gradient-side EF reaches 83.23% at IID-selected rates (table in our reply to 7S1b, Q2). These results explain why the compression placement matters.
4. Re nanoGPT: nanoGPT tests centralized updates after full-precision averaging: SignMuon loss 3.2881 versus SignSGD 3.4049. CNN2 tests worker voting.
5. Re the presentation: We will simplify the introduction, conclusion and captions, explain the two models, and put assumptions and theorem coverage in the main text.

If these answers and revisions resolve your concerns, please consider raising your score.
