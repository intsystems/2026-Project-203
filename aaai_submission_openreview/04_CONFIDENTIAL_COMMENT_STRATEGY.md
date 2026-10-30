# Optional confidential AC/SPC comment — revision 13

The AAAI-27 email confirms one Confidential Comment to the SPC and AC; it states no length limit, so the configured limit stays `null`. The draft is kept at 1483 UTF-16 characters, so it fits even a 1500-character field.

It makes four points, with reviewer IDs:

1. Agreement: all five human reviewers value Theorems 1–4. The main concerns are novelty beyond EF21-Muon, the theory–practice gap and two inconsistent tables. It notes that SignMuon (Mishra et al.) is a May 2026 preprint developed independently of this work (pqje), without identifying details of our own work.
2. Tables: main Tables 1–2 disagreed with supplement Tables A5/A8 (Ueq5, pqje); the submitted logs reproduce A5/A8, and both main tables and the ranking are corrected.
3. Theory–practice gap: under 11 Dirichlet(0.1) clients the guaranteed methods keep their IID accuracy, while every method that signs or orthogonalizes per client before averaging loses 35–45 points. It explains Muon's 46%: its clients orthogonalize before the server averages. This is consistent with Theorem 5, which has no client-dissimilarity bound.
4. Further theory since submission: lower-bounded Theorems 1–3, the Theorem 4 extension (response to Tx34), EF21 bounds for the rates we run, and the SignMuon condition.

It avoids score arithmetic and claims nothing for the open items. Source: `confidential_comment.json`; export: `rebuttals/confidential_ac_comment.md`. The pre-edit state is in `response_history/before_revision_11_rewrite_20261030T122944Z/`.

Revision 13: the comment follows the committed wording ("Dear AC/SPC, our paper explores how to combine Muon with sign compression…") and keeps the four points above.
