# Frozen EF21-SignMuon quadratic certificate

These files snapshot the manuscript addition that appeared while revision 4 of the rebuttals was being prepared. The preparation did not edit the live scientific sources. `theorem.tex` and `certificate.py` preserve the exact bytes checked here, so later author edits do not change the audit record.

Run `python certificate.py` for the exact rational/algebraic certificate. Add `--simulate` for the independent NumPy/SVD trajectory checks. `verification.log` records the passing certificate and 20 numerical trajectories of 2000 steps each. Numerical trajectories corroborate the recurrence; the exact sign identities and the analytic induction establish the theorem for all stated iterations.

For every constant learning rate and fixed EMA/Nesterov coefficient chosen for a finite budget, the constructed 8×8 convex quadratic has Frobenius smoothness 1, initial gap exactly 4/7 and Frobenius gradient norm at least 1 at every iterate. The objective may depend on the chosen parameters. The class has uniform constants. Arbitrary strictly decreasing positive steps tending to zero are covered without momentum; suitable power-decaying schedules are covered for each fixed EMA/Nesterov coefficient. Arbitrary varying momentum and finite polar approximations are not claimed.

This is a separate construction from submitted Theorem 4. The new result is supported by the analytic proof and exact algebraic certificate; it is not part of the existing 45-statement Lean audit. All 128 strict sign inequalities have margin greater than 1/50.
