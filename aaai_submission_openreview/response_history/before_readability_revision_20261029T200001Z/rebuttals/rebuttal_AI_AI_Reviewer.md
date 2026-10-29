Thank you for the technical checks.

1. Re guarantees: EF21 retains block separability and unbiased bounded-variance noise. New constant/cosine bounds use exact polar and fixed EMA/Nesterov, with residuals. The conditional SignMuon $O(T^{-1/4})$ mean gradient-norm theorem uses joint smoothness, uniform alignment and horizon tuning. Finite polar/changing momentum remain uncovered.
2. Re X/W: EF21-MuonSign queries broadcast W; the theorem concerns server X. Their losses are 3.3213 and 5.5198 in the same run; we will distinguish them.
3. Re attribution: We credit EF21, Modoranu and concurrent SignMuon; our matched study identifies distinct placement/output-EF failures. SSVR is applicable but needs matched query budgets; it is untested. SignSGD uses EMA. We will narrow geometry attribution.
4. Re scope: Added CNN2 Dirichlet(0.1) transfer at IID rates gives EF21-MuonSign 83.23% versus SignMuon 35.92%; retuning is pending. Still 11 clients, no local updates. nanoGPT’s 3.2881 versus SignSGD 3.4049 tests centralized updates after full-precision averaging.
5. Re clarity: We will add theorem coverage and move implementation details to setup. The network alignment margin remains unmeasured. Theorem 4 excludes universal fixed-step rules, not decreasing schedules/horizon tuning.
6. Re details: We will define polar(0)=0; state 1 nanoGPT run/method; avoid ranking the 0.0021 gap; cite A5 (16.6 s/epoch); qualify timing; count EF scales; correct MuonSign/SGD threshold success to 5/5.
