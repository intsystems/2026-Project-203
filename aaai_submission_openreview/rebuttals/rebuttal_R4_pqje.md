Thank you for the detailed technical and presentation feedback.

1. Re our guarantees: We generalized EF21 convergence to EMA and Nesterov momentum with deterministic learning rates, including constant and cosine schedules. Its tracking and convergence proofs are verified in Lean. The bounds retain separable layers and fixed momentum (nanoGPT varies it); numerical orthogonalization needs an error bound. The squared-gradient rate needs gradient-dependent steps.
2. Re Table 2: The submitted 5-seed logs give MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%, agreeing with A8. We will correct Table 2 and reverse that comparison. SignMuon remains 85.72±0.24%, versus Muon 85.98±0.26%.
3. Re our contribution: We also prove SignMuon converges when signing preserves enough alignment with momentum, under smoothness and unbiased bounded-variance noise. This allows interacting layers; the network condition remains unmeasured. We compare 3 sign placements, prove they can move uphill, and show output-level error feedback can diverge. We credit concurrent SignMuon and EF21.
4. Re nanoGPT: nanoGPT tests centralized optimization after full-precision averaging: SignMuon loss 3.2881 versus SignSGD 3.4049. CNN2 tests worker voting.
5. Re the presentation: We will shorten the introduction, conclusion and captions, explain the two models, and put the assumptions and theorem-coverage table in the main text.

If these answers and revisions resolve your concerns, please consider raising your score.
