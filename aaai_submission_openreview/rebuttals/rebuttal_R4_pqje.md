Thank you for the precise technical and presentation feedback.

1. Re coverage: The squared-gradient $O(T^{-1/2})$ rate needs gradient-scaled steps. Added ideal EF21 bounds cover fixed EMA/Nesterov and constant/cosine rates with residuals, retaining block separability. Finite polar solvers and nanoGPT’s changing momentum/auxiliary Adam remain outside these bounds. We will add a coverage table.
2. Re tables: The submitted 5-seed logs agree with A8: MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%. We will synchronize Table 2 and reverse that comparison. SignMuon remains 85.72±0.24%, near Muon’s 85.98±0.26% at a one-bit budget.
3. Re structure: For oracle input $M=U\Sigma V^\top$, diagonal entries of $U^\top\mathrm{sign}(UV^\top)V$ uniformly above $\delta>0$ give alignment $\ge\delta\|M\|_*$. Joint smoothness, unbiased finite-variance noise, exact polar and horizon tuning give SignMuon an $O(T^{-1/4})$ mean gradient-norm rate. The network margin remains unmeasured. We credit concurrent SignMuon and EF21; our matched study compares 6 placements/EF variants and their distinct failures.
4. Re nanoGPT: nanoGPT tests centralized optimization after full-precision averaging: loss 3.2881 versus SignSGD 3.4049. Worker voting is tested in CNN2.
5. Re writing: We will lead with gains, define server X/broadcast W, explain scope in the main text, and simplify the introduction, results and captions.

If these answers and revisions resolve your concerns, please consider raising your score.
