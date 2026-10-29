"""Generate the evidence ledger and author handoff from checked local records."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json

BASE=Path(__file__).resolve().parents[1]
BACKUP=BASE/'response_history/preparation_baseline_20261029T134259Z'


def js(value):
    return json.dumps(value,ensure_ascii=False,indent=2)+'\n'


def main():
    audit=json.loads((BASE/'evidence/source_audit.json').read_text(encoding='utf-8'))
    checked=json.loads((BASE/'evidence/verified_claims.json').read_text(encoding='utf-8'))
    records=json.loads((BASE/'evidence/concern_evidence.json').read_text(encoding='utf-8'))
    responses=json.loads((BASE/'response_fragments.json').read_text(encoding='utf-8'))
    config=json.loads((BASE/'workspace.json').read_text(encoding='utf-8'))
    originals='\n'.join(f'| `{r["path"]}` | {r["bytes"]} | `{r["sha256"]}` |' for r in audit['inputs'])
    counts=Counter(r['status'] for r in audit['code_archive_members'])
    changed='\n'.join(f'- `{r["member"]}` — {r["status"].replace("_"," ")}' for r in audit['code_archive_members'] if r['status']!='identical')
    pdfs='\n'.join(f'| `{r["path"]}` | {r["pages"]} | `{r["sha256"]}` |' for r in checked['article_versions'] if r['path'].endswith('.pdf'))
    source_map=f'''# Submission and current-source map

Prepared 29 October 2026. The reviewers saw the downloaded submission, not the updated repository article. The rebuttals therefore state the actual corrections and new results directly. Nothing was submitted, committed or pushed.

## Authority and immutable inputs

The downloaded main paper (9 pages), technical supplement (35 pages), checklist (2 pages), code/data archive, and review export are the record of what was submitted. All five original hashes were verified against `inputs_manifest.json`, then reverified by `evidence/verify_rebuttal_evidence.py`. Preserve these files unchanged.

| Original file | Bytes | Verified SHA-256 |
| --- | ---: | --- |
{originals}

`evidence/extracted/` contains page-labelled derived text, not replacement source documents. Two-column extraction sometimes interleaves text; PDF pages and TeX were used to resolve formulas/tables.

## Current article correspondence

| Current source | Role | Relevant change / limitation |
| --- | --- | --- |
| `aaai_article/v2_SignMuon_AAAI.tex` → `signmuon_body.tex` | Updated main wrapper/body | Main CIFAR and federated tables now use the values already in submitted A5/A8. Some overbroad claims remain. |
| `v2_SignMuon_AAAI_supp.tex` → `signmuon_appendix.tex` | Updated supplement | Includes expanded convergence material; numbering can differ from the downloaded supplement. |
| `v2_SignMuon_AAAI_full.tex` | Combined wrapper | Same body/appendix; do not confuse it with the reviewers' downloaded main PDF. |
| `ef21_nesterov_convergence.tex` | New positive analysis | Fixed Nesterov tracking; not nanoGPT's changing momentum. |
| `ef21_fixed_radius.tex` | New deterministic-schedule analysis | Constant/cosine first-moment residual/rate statements; fixed peak LR and momentum do not imply vanishing stationarity. |
| `ef21_rate_takeaways.tex` | Separate second-moment argument | Gradient-scaled steps and horizon tuning; no squaring of a first-moment expectation. |
| `signmuon_addendum.tex` / `_body.tex` / `.bib` | Existing author working addendum | Conditional SignMuon, bounded-below plain witnesses and EF gradient-floor scaling. Its deferred experiments are not completed results. Preserved unchanged. |
| `ef21_signmuon_divergence.tex` | Complete Theorem 4 proof | Parameter-dependent objective for fixed eta/mu, exact polar selection and rational limit cycle. |
| `signmuon_mishra_comparison.tex` and main comparison | Still needs correction | Mishra v2 (17 October) now gives conditional actual-vote stochastic theory. The v1-only dismissal is stale. |
| `evidence/EF21_JOINT_SMOOTHNESS.md` | New derivation made during this preparation | Whole-tuple EF21 reduction removes separability; explicit first-moment deterministic-schedule bound for coupled smooth objectives. Not yet inserted into the article. |

Current PDFs are separately versioned artifacts. Neither a newer filename nor a timestamp establishes that a PDF contains all current TeX changes. Their exact hashes/pages and all current TeX/Bib hashes are recorded in `evidence/verified_claims.json`.

| Current PDF | Pages | SHA-256 at audit |
| --- | ---: | --- |
{pdfs}

## Submitted code versus repository

The submitted archive contains 109 non-directory files: {counts['identical']} byte-identical, {counts['different']} different, and {counts['missing_in_repo']} absent in the audited repository snapshot. `evidence/source_audit.json` records every member's original/current hashes. `evidence/submitted_to_repo.diff` contains the text differences. The archive is authoritative for whether evidence was already submitted.

{changed}

Core algorithms were read in the archive and repository. Changes include optional compression diagnostics, tests/docs and result-container metadata; do not infer source identity from matching aggregate tables. More strongly, all **175 federated metrics training histories** in the original nested archive are semantically identical to their current counterparts. This was checked independently of metadata, line endings and ZIP bytes.

Live repository work appeared during preparation in `code/federated/heterogeneous.py`, its test and README files. Those files were left intact. A runner and tests do not constitute a completed heterogeneous experiment; no such new result was available in the audited result directory.

## Which values to use

| Setting | Stale submitted main | Submitted supporting evidence / updated main | Treatment |
| --- | --- | --- | --- |
| Federated MuonSign vs SignSGD | Table 2: 80.68±1.88 vs 81.83±0.43 | A8 + submitted five-seed metrics: **82.94±0.19 vs 81.44±0.15** | Own the reporting error; reverse the affected ranking sentence. |
| Federated SignMuon vs Muon | Table 2: 85.81?0.25 vs 85.99?0.26 | A8/archive: **85.72±0.24 vs 85.98±0.26** | Comparable observed accuracy; no equivalence/significance proof. |
| Centralized SignMuon vs Muon | Table 1: 94.26±0.15 vs 94.26±0.18 | A5 + submitted `article_export/table_cifar.csv`: **94.60±0.15 vs 94.35±0.27** | Replace the whole main table coherently, not just a convenient cell. |
| nanoGPT | Table 3 | Unchanged single-run values; SignMuon3.2881, EF21-SignMuon3.2860, SignSGD3.4049 | No new seeds or statistical ranking. |

Federated values are means across five independently trained seeds of each seed's final-five-evaluation average; SD is the sample SD across those five means. Repeated checkpoints are not independent seeds. The central submitted export has three independent final seeds and rounded aggregate values; its extracted `runs.csv`/`table_cifar.csv` are in `evidence/`. nanoGPT has one independent, unpinned run per compressed method. X/W are two evaluations of the same EF21-MuonSign run.

## Reviews and conference settings

| Workspace | Reviewer | Note ID | Rating / confidence |
| --- | --- | --- | --- |
| R1 | 7S1b | smHroyG4g5 | 6 / 4 |
| R2 | D4r2 | UC752MnbRo | 7 / 4 |
| R3 | Ueq5 | xHT5WIiVi7 | 6 / 2 |
| R4 | pqje | kkv2ydIYff | 5 / 4 |
| R5 | Tx34 | EDZy4k8BK7 | 4 / 3 |
| AI | AI_Reviewer | LKOb5GFqWI | AI review; no score request |

Configured author-supplied rules: **1500 UTF-16 source characters per review**, headings/newlines/math included; no external links; deadline **31 October 2026 AoE**. These settings were retained. The confidential field's limit/readers remain unknown and must be checked in the actual form. No official field rules were inferred from a different venue.

## Preservation and summary migration

Late source update: the live author addendum body/PDF changed after the initial article hash snapshot. The latest 479-line body was read again: its added heterogeneous experiment plan still has pending results, and its EF21 cross-layer extension remains marked open. The prepared rebuttal's mathematical claims remain consistent; the independent J1 derivation is still separate from that addendum. `evidence/final_source_drift.json` records the later hashes and the concurrently changed code README files. Those source files were not edited by this preparation.

`response_history/preparation_baseline_20261029T134259Z/` preserves all pre-preparation JSON, summaries and strategy files. The imported fragments had revision0 and empty answers. All original concern IDs are retained, but placeholder headings/praise/reference entries were reassigned to genuine, fine-grained concerns. The migration affects empty scaffolding, not authored scientific answers. All appended **Original review** blocks remain byte-for-byte equivalent after normal newline decoding; original JSON bytes remain unchanged. Each new concern has one response fragment and exactly one topic. Duplicate mentions link to the same concern. `evidence/concern_evidence.json` records all 64 concrete concerns, evidence, intended answers and pending status.
'''

    plan=r'''# Rebuttal plan and concern-to-evidence ledger

## Strategy

Lead with defensible results. R3/R4's table discrepancy is a real reporting error settled by **already submitted** five-seed logs. R5's quantifier and necessity objections are mathematically correct; narrow the claims while defending the LMO-specific negative constructions. R1/R4's request for a positive SignMuon explanation can now receive a sufficient structural condition and conditional theorem. None of these changes establishes an end-to-end guarantee for the actual network runs.

The updated article is an author resource. Every active reply states its answer directly; it does not ask reviewers to consult an inaccessible revision. Text promises below have concrete proposed wording/tables in this file. New training is never claimed.

## Evidence register

| Tag | Evidence | What it establishes | Limit / remaining action |
| --- | --- | --- | --- |
| E1 | Submitted A8; original nested `code/results/federated_export_results.zip`; `evidence/verified_claims.json` | Independently recomputed five-seed tail means, sample SDs and threshold crossings; all 175 original/current training histories identical | Fix stale main Table 2 and associated ranking. No new training or significance test. |
| E2 | Submitted A.11/A.14/A.15/A.18; federated `algorithms.py`, `main.py`, `methods.py` | CNN2,11 homogeneous fully participating clients; three fixed-weight minibatches of 64 accumulated per client | Zero local parameter steps. No extrapolated ordering under label skew/local training/partial participation. Live heterogeneous runner is not a result. |
| E3 | Submitted Table3/A.16/A.17; `code/results/nanogpt/` | Exact losses, X/W gap, one run/method, full-precision reduce-scatter then local optimizer/all-gather | No variant-specific variance; no measured one-bit network savings; changing momentum and auxiliary Adam. |
| E4 | Submitted A5 and `article_export.tar.gz` → `table_cifar.csv`, `runs.csv` | Updated central accuracy values and existing seconds/epoch | Three final seeds; old generic2x timing claim should be replaced with measured scope. |
| C1 | Submitted communication appendix, federated `methods.py` and estimator allocation | Logical bits, FP32 scales/auxiliaries, polar location and persistent buffers | Single-GPU simulation; no packed-network throughput/latency measurement. |
| N1 | Submitted Theorems1–4; full proofs; rational certificates in `verified_claims.json` | Each plain placement can ascend; specified output-EF can diverge on smooth lower-bounded objectives | Exact polar; Thm4 has fixed parameters and parameter-dependent objective. |
| N2 | Existing author `signmuon_addendum_body.tex` | Softplus strengthens the three plain witnesses to one fixed smooth lower-bounded objective for all positive step and EMA/Nesterov schedules; EF gradient floor scales as c_mu L eta | No arbitrary-schedule impossibility for output EF; its vanishing floor does not refute horizon-tuned rates. |
| G1 | Existing author addendum; OLion geometry | Structural sign/polar alignment criterion and exact-polar conditional SignMuon theorem, including stochastic fixed EMA/Nesterov | Unmeasured uniform geometric margin in our networks; distributed final-vote condition is separate; no performance ranking. |
| T1 | Submitted A.10; updated Nesterov/fixed-radius/rate TeX | Separate sharp second-moment proof; new fixed-momentum deterministic-schedule bounds | Never square a first-moment bound; the squared result needs gradient-scaled radius and horizon tuning. |
| J1 | New `evidence/EF21_JOINT_SMOOTHNESS.md`, Sections1–3 | Whole-tuple norm reduction removes separability. Explicit bound(J) gives first-moment rates/residuals for either downlink, fixed EMA/Nesterov and arbitrary deterministic LR | Exact polar, global joint smoothness, no auxiliary Adam/decoupled decay; not inserted in article. Constants may be poor. |
| P1 | EF21-Muon, Modoranu2024, Mishra2026v2 | Correct prior-work attribution, including concurrent actual-vote conditional stochastic theory | Replace current main/v1 comparison. Do not market the general framework or practical SignMuon as new. |
| P2 | Jiang2024SSVR; Xiao2025; existing SignSGD implementation | Relevant modern sign baselines and theory; existing SignSGD already has momentum | No SSVR comparison completed; no exclusive geometry attribution; SignSGD risk curves do not prove SignMuon properties. |
| W1 | Concrete proposed text and tables below | Reviewable presentation corrections and scope statements | Proposed article changes, not silently applied to author TeX. |

## Precise new positive statements

For (M=USigma V^	op), (Q=UV^	op), (S=mathrm{sign}(Q)), write (c_j=[U^	op S V]_{jj}). Then

[
 langle M,S
angle=sum_jsigma_j c_j.
]

Thus **every (c_jgedelta>0)** is an easily stated sufficient condition for (langle M,S
anglegedelta|M|_*). More generally, let (a=sum_jsigma_j/r), (b=|Q|_1/r), (
ho=|Sigma-aI|_F/(asqrt r)), (arepsilon=|c-bmathbf1|_2/(bsqrt r)). Cauchy–Schwarz gives

[
 langle M,S
anglege a|Q|_1(1-arepsilon
ho).
]

The mean identity is (sum_j c_j=|Q|_1). A uniformly positive **product margin**, not small correlation dispersion alone, is needed. This development is motivated by OLion's conditional geometry and must be credited. Uniform alignment along the *actual momentum signals*, joint smoothness, unbiased finite-variance noise, exact polar, (eta=eta_0T^{-3/4}), and (mu=1-T^{-1/2}) fixed within a run give an average expected first-moment nuclear-gradient rate (O(T^{-1/4})) for EMA/Nesterov. Full proof: the existing author addendum. Do not claim the actual networks satisfy its premise.

For EF21, analyze all layers as one block with norm (max_i|X_i|_{
m op}/lambda_i), dual (mathcal N_lambda(G)=sum_ilambda_i|G_i|_*), and (B^2=sum_ilambda_i^2min(m_i,n_i)). Joint Frobenius smoothness implies the arbitrary-norm framework's smoothness with constant (B^2L); separate scaled-sign compression of layers is still Frobenius contractive. This removes the original block-separability restriction **within the prior framework**, without claiming a new general framework.

The explicit bound(J) in `EF21_JOINT_SMOOTHNESS.md` further proves an LR-weighted expected first-moment guarantee at server X for deterministic schedules, either downlink, and fixed EMA/Nesterov. Fixed peak LR/momentum leave a residual. Constant/cosine schedules with (T^{-3/4}) peak and (1-mu=T^{-1/2}) yield (O(T^{-1/4})). It is not a squared-gradient rate. Joint coupling is handled by the full model displacement in every gradient-drift term. The proof is analytic; the recorded numerical norm checks are only sanity checks, not the proof.

## What each guarantee covers

| Statement | Oracle / update | LR / momentum | Objective/noise | What it controls |
| --- | --- | --- | --- | --- |
| Submitted Thms1–3 | Exact polar with three plain sign compositions | Any positive steps; linear-case EMA/Nesterov | Linear local descent witnesses, unbounded below | Negative gradient–step pairing, not a network ranking |
| New bounded-below plain witnesses | Same exact compositions | Any positive step schedule, any EMA/Nesterov coefficients in[0,1) | One fixed softplus objective per placement, smooth convex/lower-bounded | Gradient norm bounded away from zero for all t; objective diverges if sum of steps diverges |
| Submitted Thm4, output EF | Specified scaled-sign estimate of exact polar outputs | Every fixed eta>0, fixed mu<1; objective may depend on them | Smooth/lower-bounded construction | Divergence; no universal constant-step rule based only on L,mu |
| Submitted/updated sharp EF21 rate | Exact gradient/momentum LMO with scaled radius | Gradient-dependent radius, prescribed horizon-tuned momentum | Explicit smoothness and stochastic variance assumptions | Separate expected **squared** dual-gradient rate O(T^-1/2) at X |
| New J1 bound(J) | Exact polar, gradient-side EF, identity or EF model downlink | Deterministic constant/cosine/etc; fixed EMA/Nesterov | Joint Frobenius smoothness, conditionally centered bounded-variance noise | Weighted **first** moment at X; residual for fixed parameters, tuned O(T^-1/4) |
| New conditional plain SignMuon | Exact polar-sign update | Horizon-tuned steps and fixed EMA/Nesterov | Joint smoothness/noise plus uniform positive alignment | First moment; condition not measured in the networks |
| CIFAR actual implementations | Finite five-iteration polynomial polar; auxiliary Adam group | Tuned cosine; fixed momentum | Real networks; global smoothness not verified | Empirical results, no complete end-to-end guarantee |
| nanoGPT actual implementation | Five Polar Express stages, full-precision reduction; auxiliary Adam | Momentum warmup/cooldown, flat/linear LR | Single training run/method | Central optimizer evidence, no full-run theorem/network benefit |

Finite polynomial stages may not preserve exact-polar signs. The addendum's sign-mismatch penalty is a conditional extension; its smallness is not established for the actual runs. Failure of spectral contractivity is not proof that no other analysis, LR, initialization or compressor repair can work.

## Complete method / systems comparison

Define (P(H)=operatorname{polar}(H)), (s(H)=operatorname{sign}(H)), and (C(R)=operatorname{mean}|R|,s(R)). The descent update subtracts the listed direction. The following compares distributed semantics, not nanoGPT's full-precision transport. Matrix parameters are compressed; auxiliary vector/head parameters use FP32. Raw sign votes are nonlinear and cannot be moved through averaging or the polar operation.

| Method | Sign placement / EF target | Polar location | Matrix uplink | Matrix downlink | Extra persistent matrix-size state beyond common model/momentum | Guarantee |
| --- | --- | --- | --- | --- | --- | --- |
| SignMuon | (s(P(M_j))), worker votes; no EF | Clients | 1 bit/entry | 1 bit/entry | None | Plain-method counterexample; new conditional central alignment result |
| MuonUSign | Server polar after client sign vote; no EF | Server | 1 bit/entry | 32 bits/entry | None | Plain-method counterexample |
| MuonSign | Server polar after client sign vote, sign output; no EF | Server | 1 bit/entry | 1 bit/entry | None | Plain-method counterexample |
| EF21-SignMuon | EF tracks client polar output | Clients | 1 bit/entry + FP32 scale/block | 32 bits/entry | One estimator/client + one server aggregate | Output-EF counterexample already at N=1 |
| EF21-MuonUSign | EF tracks gradient/momentum input to polar | Server | 1 bit/entry + FP32 scale/block | 32 bits/entry | One estimator/client + one server aggregate | Exact-polar EF21 at server X=W, with stated steps/noise |
| EF21-MuonSign | Input EF plus model-downlink EF | Server | 1 bit/entry + FP32 scale/block | 1 bit/entry + FP32 scale/block | One estimator/client + one server aggregate + exact X separate from broadcast W | Exact-polar EF21 at server X; W is queried/communicated |

CNN2: (P_{
m mat}=762560), (P_{
m aux}=2146), 3 matrix blocks. Per-client/per-round per-channel counts:

- Plain compressed: ((P_{
m mat}+32P_{
m aux})/(P_{
m mat}+P_{
m aux})=1.0870) bits/parameter.
- EF compressed: add (32\times3) bits, giving 1.0871 bits/parameter.
- Bidirectionally compressed payload reduction versus FP32: about 29.4x; uplink-only **round-trip** reduction: about 1.93x.
- Each extra FP32 matrix-size buffer has 3,050,240 bytes (about 2.91 MiB). Scratch polar workspaces and common momentum/Adam state are excluded; actual peak memory is not measured by this table.

Practical recommendation: if uplink is limiting and the analyzed repair matters, EF21-MuonUSign keeps the downlink exact; if both channels are constrained, EF21-MuonSign compresses both, with additional state and an X/W gap to monitor. SignMuon is the strongest observed vision-accuracy candidate, with a worst-case failure and only conditional positive theory. No universal ordering or guarantee for the finite solvers is implied.

## Concrete proposed article corrections

### Reporting and empirical claims

> Table 2 in the submission was stale. The submitted five independent 2000-round runs agree with Table A8: MuonSign 82.94±0.19%, SignSGD 81.44±0.15%, SignMuon 85.72±0.24%, and Muon 85.98±0.26% (sample SD). MuonSign therefore exceeds SignSGD by 1.51 percentage points. These values replace the main table and reverse the affected comparison. The central main table is similarly synchronized to submitted Table A5. No additional training was required for these corrections.

> The federated experiment is synchronous distributed CNN2 training with 11 homogeneous fully participating clients. Each client accumulates three minibatches of 64 at the same weights; clients take no local parameter steps. The result does not establish an ordering under statistical heterogeneity, local training or partial participation.

> The nanoGPT experiment averages gradients in full precision by reduce-scatter, then applies a centralized optimizer on owning ranks and gathers parameters. It tests optimizer behavior at language-model scale; it does not test compress-before-aggregation worker voting or demonstrate one-bit network savings. Each compressed method has one independent training run. The 0.0021 loss difference between EF21-SignMuon and SignMuon is inconclusive; no compressed-method variance estimate is available. Upstream Muon variation is not a variance estimate for the other methods.

> EF21-MuonSign's exact server X has validation loss 5.5198 and its broadcast W has 3.3213 in the same run. Gradients are queried at W; the ideal theorem bounds stationarity at X. Successful validation at W does not validate the guarantee at X. Layerwise downlink concentration is observed, but does not prove its cause or that changing compressors is the only remedy.

### Definitions and guarantees

> The sign direction is itself an LMO: (-operatorname{sign}(G)) minimizes (langle G,D
angle) over the unit entrywise infinity-norm ball. Composing it with the polar LMO can nevertheless produce an ascent direction. Plain signs carry no scale; scaled-sign EF transmits one mean-absolute residual scale per block in addition to bits. Auxiliary FP32 parameters are counted separately.

> Centralized placement formulas correspond to a single worker. Distributed variants apply signs or polar factors locally before aggregation, as specified in their algorithms; these operations generally do not commute with averaging. We retain the existing method names to keep code and tables traceable and add an explicit composition-order legend.

> Theorem 4 states: for every fixed (L>0,eta>0,muin[0,1)), there exists an (L)-smooth lower-bounded objective on which the specified output-EF recursion diverges. The objective can depend on the parameters. The result does not establish failure of decreasing schedules or exclude horizon-tuned finite-time rates, other EF repairs or different analyses. Replace “unrepairable”, “unconditional guarantee” and “only changing the compressor” with the precise relevant statement.

> Our convergence statements require explicit conditional unbiasedness and finite variance as well as smoothness and lower boundedness. The squared-gradient rate uses a separate second-moment argument with gradient-scaled steps. A first-moment bound cannot be squared to obtain an expected-square bound. The new whole-tuple formulation permits coupled layers, and the deterministic-schedule theorem gives first-moment residuals/rates for ideal fixed-momentum EF21. Finite polynomial polar, changing nanoGPT momentum and auxiliary Adam remain outside the full-run guarantee.

> The linear examples establish a failure of descent outside the lower-bounded assumption; (f	o-infty) for uncompressed Muon is literal on those linear objectives. New softplus examples preserve each plain placement's failure while satisfying smoothness and lower boundedness. Set the theoretical thin-SVD selection (operatorname{polar}(0)=0); distinguish this from randomized strict-binary sign ties in the implementation.

### Presentation and related work

- State operations and the meaning of gradient–step alignment before referring to numbered theorems. Put the method and coverage tables near the definitions.
- Move practical polar approximation and per-layer scaling from the middle of the theory to experimental setup. Define X as the exact server iterate and W as its compressed broadcast in two short sentences.
- Lead results paragraphs with measured outcomes. Explain target-crossing success counts before reporting target rounds; a fixed target can be affected by residual floors and does not alone identify an asymptotic rate. Corrected MuonSign and SGD both reach 80% in 5/5 submitted runs.
- Replace repeated “document rather than resolve” wording with the actual result. Shorten captions and bring the essential qualifications from A.7/A.10/A.12/A.16/A.17 into the main text.
- Credit prior EF21-Muon for the general positive framework, and Modoranu2024 for compression before full-matrix transforms. Credit Mishra v2's conditional actual-vote theory and OLion's geometric analysis. Do not repeat the current manuscript's v1-only dismissal of concurrent SignMuon.
- Cite Jiang2024SSVR and narrow geometry-only attribution. The existing SignSGD baseline already uses momentum, but no SSVR training comparison was performed. Xiao2025 analyzes SignSGD, not our composed SignMuon update.
- Withdraw the broad 2x wall-clock assertion. Submitted A5 does contain seconds/epoch (SignMuon 16.6, Muon 14.4, SignSGD 12.5, SGD 12.2, Adam 12.7); those are not end-to-end compressed-network timing measurements.

## Primary external sources checked on 29 October 2026

These links are internal evidence only and are excluded from every exported rebuttal. They were used to check attribution and the most recent concurrent theory, not to invent new empirical support.

- [Gruntkowska et al., Error Feedback for Muon and Friends](https://arxiv.org/html/2510.00643v1): arbitrary-norm distributed EF21 framework and its smooth stochastic theorems.
- [Wang, Shen and Zhang, OLion v2](https://arxiv.org/html/2602.01105v2): conditional diagonal-correlation geometry and spectral-spread interaction. Our own network margin is not measured.
- [Mishra et al., SignMuon v2](https://arxiv.org/html/2605.16311v2): conditional actual-vote theory, stochastic momentum and joint-block extension. The favorable condition is not automatic; our negative examples remain compatible with it.
- [Modoranu et al., ICML2024](https://proceedings.mlr.press/v235/modoranu24a.html): error feedback for compressed structured preconditioners.
- [Jiang et al., NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3b7db05c0e383518d789b6e93131f1f0-Abstract-Conference.html): variance-reduced sign methods; no new comparison is available here.
- [Xiao et al., ICML2025](https://proceedings.mlr.press/v267/xiao25c.html): SignSGD risk analysis, not a guarantee for signing a polar factor.

## Every concern: source, finding, limit, concise answer

Exact reviewer quotes live in `annotations.json` and the immutable original-review blocks. The following ledger is generated from the same 64 selectable concerns. “Pending” concerns remain excluded from the active draft and visibly flagged in the editor. A completed limitation statement can be checked even when a separate request for new experiments remains open.
'''
    ledger=[]
    for r in records:
        status='PENDING — excluded; needs evidence' if r['pending'] else ('CHECKED — selected' if r['include'] else 'CHECKED — excluded for space')
        ledger.append(f"### {r['id']} — {r['title']}\n\n**Status:** {status}.\n\n**Reviewer request:** {r['detail']}\n\n**Sources:** {r['evidence']}\n\n**Finding, meaning and limits:** {r['finding']}\n\n**Concise candidate:** {r['export_text']}\n")
    plan+='\n'+'\n'.join(ledger)

    ac_text=('Dear AC/SPC, we highlight two corrections relevant across reviews. Ueq5 and pqje (Q2) correctly identified a stale main table. '
             'The already-submitted five-seed logs independently reproduce Table A8: MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%, '
             'not the reversed main-table ordering. SignMuon remains 85.72±0.24% versus Muon 85.98±0.26%. This is a reporting correction, not new training.\n\n'
             'Tx34 (Q1–2) is right that the general positive EF21 framework is prior work and Theorem 4 does not rule out decreasing schedules or horizon-tuned rates. '
             'Our principal submitted novelty is the explicit sign-placement ascent analysis and smooth lower-bounded output-EF divergence construction. '
             'We will narrow the wording accordingly.\n\n'
             'For 7S1b (Q1) and pqje’s scope concern, we add a structural positive-alignment condition and conditional SignMuon convergence. '
             'We also derive jointly smooth EF21 bounds without layer separability. These ideal results do not certify finite polar solvers, changing nanoGPT momentum or auxiliary Adam. '
             'Additional compressed-method seeds, trajectory diagnostics and broader federation remain open; nanoGPT is a centralized optimizer test after full-precision averaging. '
             'We ask that the verified correction and precise negative contribution be considered with these explicit limits.')
    ac={'version':1,'reviews':{'confidential_ac':{
        'target':'confidential_ac_comment.md','revision':1,'intro':'','closing':'',
        'sections':[{'id':'comment','lead':''}],'reply_topics':[],
        'fragments':[{'id':'C1','text':ac_text,'notes':'Optional scientific synthesis, not a process complaint. Science checked against E1/N1/G1/J1. The actual confidential field limit and readers remain unverified; check them before using this text. Export success only validates local text/mappings.','include':True,'needs_check':False,'joiner':' ','section':'comment'}]}}}
    config['confidential']['references']=[
        {'stem':'review_R3_Ueq5','point':'R3.2','label':'Ueq5 · Table 2/A8 discrepancy'},
        {'stem':'review_R4_pqje','point':'R4.3','label':'pqje · Question2: correct five-seed results'},
        {'stem':'review_R5_Tx34','point':'R5.1','label':'Tx34 · Question1: novelty beyond EF21'},
        {'stem':'review_R5_Tx34','point':'R5.2','label':'Tx34 · Question2: Theorem 4 quantifiers'},
        {'stem':'review_R1_7S1b','point':'R1.1','label':'7S1b · Question1: positive structural condition'},
        {'stem':'review_R4_pqje','point':'R4.1','label':'pqje · Question1: theorem/implementation coverage'}]
    strategy=f'''# Optional confidential AC/SPC comment

## Recommendation

Use only if the actual AC/SPC field permits a short scientific synthesis. There is a concrete cross-review purpose: establish the original-archive basis of the Table 2 correction (Ueq5, pqje Q2), distinguish submitted negative novelty from inherited EF21 theory (Tx34 Q1), and scope Theorem 4 correctly (Tx34 Q2). Mention the added conditional/joint-smooth analysis and the remaining experimental limits without implying all concerns are resolved.

The prepared text is **{len(ac_text.encode('utf-16-le'))//2+1} UTF-16 characters including its exported final newline**. Its factual content is checked. The confidential limit remains `null`; permitted readers and actual posting rules have not been verified. A local “ready” badge does not certify an unknown conference field. This is optional and has not been posted.

## Tone and alternatives

- No score arithmetic, speculation about motives, allegation against reviewers, or hidden claim of completed experiments.
- The acceptance-oriented request is to weigh the verified evidence and precise contribution with explicit limits, rather than to dismiss the negative review.
- If the field is reserved for process issues or its readers/limit are unsuitable, omit the comment. The public replies already contain the relevant scientific answers.
- Local references in `workspace.json` link directly to the six relevant reviewer concerns without adding external URLs to the export.

## Source of the text

`confidential_comment.json` is independently editable. It does not consume a human review's 1500-character allowance. Its baseline empty C1 draft was backed up in `response_history/preparation_baseline_20261029T134259Z/` before replacement. No Git/OpenReview operation is attached to saving.
'''
    docs={'01_SUBMISSION_MAP.md':source_map,'02_REBUTTAL_PLAN.md':plan,
          '04_CONFIDENTIAL_COMMENT_STRATEGY.md':strategy,'confidential_comment.json':js(ac),'workspace.json':js(config)}
    guard=BASE/'evidence/document_outputs.json'
    previous=json.loads(guard.read_text(encoding='utf-8')) if guard.exists() else {}
    for name,value in docs.items():
        path=BASE/name
        if path.exists():
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual==previous.get(name) or path.read_bytes()==(BACKUP/name).read_bytes(), 'Author edit detected: '+name
    for name,value in docs.items():
        with (BASE/name).open('w',encoding='utf-8',newline='\n') as f:
            f.write(value)
    guard.write_text(js({name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in docs}),encoding='utf-8')
    print('Updated source map, full64-concern ledger, optional AC draft and local references.')


if __name__=='__main__':
    main()
