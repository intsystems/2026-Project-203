Thank you for the technical checks.

1. Re our theory: After submission, we generalized EF21-Muon(U)Sign convergence to fixed EMA/Nesterov and deterministic rates, and verified the proofs in Lean. The bounds use separable-layer smoothness and unbiased bounded-variance noise, with an explicit polar approximation term.
2. Re the two models: EF21-MuonSign queries broadcast W but guarantees server X. Their nanoGPT losses are 3.3213/5.5198 in the same run, after full-precision gradient averaging.
3. Re our contribution: All 3 sign placements can ascend; output-level EF can fail with decreasing steps or budget tuning. SignSGD already uses EMA. We cannot isolate geometry without a matched-query SSVR comparison, which remains untested.
4. Re heterogeneous data: Our new federated CNN2/CIFAR-10 tests use 11 Dirichlet(0.1) clients and IID-selected rates, without local updates. **Only EF21-Muon(U)Sign exceed 80% among Muon variants** (table: 7S1b, Q2).
5. Re the SignMuon condition: Our separate theorem proves SignMuon convergence for interacting layers when signing preserves a uniform fraction of the polar update’s alignment with momentum. The negative result concerns the specified output-EF scheme.
6. Re the reported measurements: nanoGPT has 1 run/method; a 0.0021 gap cannot rank methods. CNN2’s 1.087 bits/parameter includes auxiliaries and EF scales. A5 reports SignMuon 16.6 s/epoch on centralized ResNet18/CIFAR-10. MuonSign and SGD reach 80% in all 5 IID seeds.
