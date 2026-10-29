# Readability revision and Lean scope

The replies were rewritten from the current author state, including R5 revision 8. They use full sentences, name the method/theorem being discussed and explain the practical meaning. “New proofs” is replaced by explicit ascent and divergence statements. “Horizon tuning,” “with residuals” and compressed lists of theorem properties are replaced by explanations of learning-rate choices and error terms. The framing remains strong sign-based performance plus the comparative placement analysis.

## Lean evidence

- `lean_proofs/comparator/VERIFICATION.json`: 45 frozen statements passed comparison and kernel replay on 2026-10-23. The reference and all three dependency hashes were rechecked and match.
- `lean_proofs/README.md` and `PAPER_AUDIT.md`: original sign-placement ascent and smooth lower-bounded output-EF divergence results have Lean proofs. The final bounded-below objective uses a scalar clipping construction, supporting the existential conclusion while differing from the coordinate modification in the paper.
- Generalized EF21 tracking/convergence arguments cover fixed EMA/Nesterov, deterministic schedules and explicit approximation error/quality assumptions. The multiple-block EF21 trajectory retains separability. Constant/cosine paper specializations use these results; the printed cosine sum is a manual composition, not a separate exported Lean theorem.
- The conditional plain SignMuon theorem and new softplus extensions are not claimed Lean-verified. Their separate analytic proofs remain the evidence.

The full Lean build was not rerun during this prose edit. No scientific source, proof implementation or training result was changed. Exact review passages, all 64 concern IDs, 79 source links, selections and pending flags are preserved. All five human score requests remain active.
