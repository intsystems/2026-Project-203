# Revision 7: completed heterogeneous LR tuning

The frozen fourth results archive contains all ten validation-selected learning rates and five final seeds per method. The independent audit reproduces 167 completed 2000-round jobs, 72 validation trials and 20 final result rows. All original transfer runs are unchanged. MuonSign's two extra validation rates make its unchanged 5e-5 winner interior; every selected rate is now interior.

All six public replies and the AC comment now cite the tuned comparison. The shared table in 7S1b Q2 contains all ten tuned means/sample SDs, ordered by mean, with both gradient-side EF rows bold. Their results are 84.03±0.52% and 83.84±0.32%; the other methods' means are below 80%. The benchmark, validation selection, fixed momentum and five final seeds are identified.

The addendum replaces all pending tuning cells with selected rates and verified results. It documents the complete search and endpoint refinement, keeps the transfer comparison, and explains why the EF advantage persists after LR tuning. Existing mathematical sections and author edits are preserved; the standalone PDF is rebuilt.

Selections, all eight pending flags, seven presentation omissions, original reviews, 64 concern IDs, 79 exact passage mappings and five human score requests are preserved. LR selection is complete; local model steps and partial participation remain untested. No new training or full Lean build was run by this preparation.
