Thank you for the questions on theory and practical use.

1. Re structure: For oracle input $M=U\Sigma V^\top$, diagonal entries of $U^\top\mathrm{sign}(UV^\top)V$ uniformly above $\delta>0$ give alignment $\ge\delta\|M\|_*$. Joint smoothness, unbiased finite-variance noise, exact polar and horizon tuning give SignMuon an $O(T^{-1/4})$ mean gradient-norm rate. The network margin remains unmeasured.
2. Re federation: New CNN2 Dirichlet(0.1) transfer, 5 seeds at IID rates: EF21-MuonSign 83.23±0.36%, SignMuon 35.92±8.49%. The ordering changes. Retuning is pending; still 11 clients and no local updates.
3. Re analysis: Base EF21 rates are inherited; added fixed EMA/Nesterov and constant/cosine bounds retain separability. Our conditional SignMuon theorem allows interacting layers; the counterexamples motivate its margin.
4. Re choice/cost: For IID accuracy: SignMuon, 85.72% versus Muon 85.98% and SignSGD 81.44% (5 seeds), at one bit both ways. For the analyzed repair: EF21-MuonUSign uplink-only; EF21-MuonSign both ways. CNN2 costs 1.087 bits/parameter including FP32 auxiliaries. EF adds scales, client/server estimators and, if two-sided, a server model. Polar: clients for SignMuon, server for gradient EF.
5. Re nanoGPT: nanoGPT loss: 3.2881 versus SignSGD 3.4049, one run/method. The 0.0021 gap to EF21-SignMuon supports no ranking without compressed-method variance estimates.

If these clarifications address your concerns, please consider raising your score.
