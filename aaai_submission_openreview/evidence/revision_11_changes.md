# Revision 11: direct answers and the heterogeneity mechanism

The six public replies and the AC/SPC comment were rewritten after the AAAI-27 author-response email confirmed the rules: one response per review at most 1500 characters, one Confidential Comment to the SPC and AC, and no external links.

Each reply now answers the reviewer's literal question first, concedes valid points plainly and labels post-submission results. The 7S1b table pairs submitted IID accuracy with the tuned Dirichlet(0.1) accuracy. The replies explain the drop by where each method applies sign or polar: per client before averaging, or once on the server. They state consistency with Theorem 5, which has no client-dissimilarity bound.

Previously omitted editorial concerns now receive concrete commitments. pqje's writing concerns have their own answer instead of sharing the X/W text. The submitted main Table 1 disagreement with Table A5 is acknowledged alongside Table 2's. Wording that SignMuon "leads" centrally is removed.

Sources and checks are recorded in the Revision 11 section of `02_REBUTTAL_PLAN.md`; hashes are in `revision_11_manifest.json` and the build/check results in `revision_11_validation.json`. No training, proof, addendum or PDF changed. All 64 concern IDs and 79 source mappings remain.
