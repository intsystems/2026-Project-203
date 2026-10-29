"""Paper-specific preparation; preserves original reviews and guards author edits.

Run only to regenerate this preparation, not to reset later browser edits. A
hash mismatch against the previous generated output refuses replacement.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import sys

BASE = Path(__file__).resolve().parents[1]
BACKUP = BASE/'response_history/preparation_baseline_20261029T134259Z'
CONFIG = json.loads((BASE/'workspace.json').read_text(encoding='utf-8'))
NOTES = {n['id']: n for n in json.loads((BASE/CONFIG['source']['reviews_json']).read_text(encoding='utf-8'))['notes']}
OUTPUTS = {}
RECORDS = []
RESPONSES = {'version':1, 'reviews':{}}
ANNOTATIONS = {'schema_version':1, 'reviews':[]}


def js(value):
    return json.dumps(value,ensure_ascii=False,indent=2)+'\n'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def prepare(number, overview, intro, closing=''):
    spec = CONFIG['reviews'][number]
    fields = {k: str(v['value'] if isinstance(v,dict) and 'value' in v else v)
              for k,v in NOTES[spec['note_id']]['content'].items()}
    original = (BACKUP/'reviews'/(spec['stem']+'.md')).read_text(encoding='utf-8').split('## Original review\n',1)[1]
    mapping = {'overview':[['review',next(p.strip() for p in re.split(r'\n[ \t]*\n+',fields['review'])
                                        if p.strip() and not p.strip().startswith('**Title:')),0]]}
    review = dict(target=spec['target'],revision=1,intro=intro,closing=closing,
                  sections=[{'id':'answers','lead':''}],fragments=[],reply_topics=[])
    return dict(spec=spec,fields=fields,original=original,mapping=mapping,review=review,
                overview=overview,concerns=[])


def passage(d, field, needle):
    raw = d['fields'][field]
    candidates = [p.strip() for p in re.split(r'\n[ \t]*\n+|\n(?=\d+\.\s)',raw) if needle in p]
    assert len(candidates)==1,(d['spec']['stem'],field,needle)
    return [field,candidates[0],0]


def point(d, identity, title, detail, links, evidence, full_answer,
          text=None, pending=False, include=True):
    assert title.endswith('.') is False
    entries = [passage(d,*link) for link in links]
    d['mapping'][identity] = [entry for i,entry in enumerate(entries) if entry not in entries[:i]]
    d['concerns'].append((identity,title,detail))
    answer = full_answer if text is None else text
    note = f"Evidence: {evidence}\nFinding / complete answer: {full_answer}\n"
    note += ('UNFINISHED: requested additional evidence is not available; do not imply completion.' if pending
             else 'Checked clarification or analytic result. Any stated limitation remains a limitation; no new training is claimed.')
    d['review']['fragments'].append(dict(id=identity,text=answer,notes=note,include=include,
                                         needs_check=pending,joiner=' ',section='answers'))
    RECORDS.append(dict(review=d['spec']['stem'],id=identity,title=title,detail=detail,evidence=evidence,
                        finding=full_answer,pending=pending,include=include,export_text=answer))


def topic(d, identity, heading, field, needle, points):
    entry = passage(d,field,needle)
    d['review']['reply_topics'].append(dict(id=identity,heading=heading,question=entry[1],
                                             source=field,occurrence=0,points=points))


def shared(d, groups):
    """One editable answer can answer several linked concerns without duplication."""
    blocks = []
    if d['review']['intro']:
        blocks.append(dict(kind='intro',before='',after='\n\n'))
    for tid, point_groups in groups:
        blocks.append(dict(kind='heading',topic=tid,before='',after=' '))
        for i, ids in enumerate(point_groups):
            blocks.append(dict(kind='answer',points=ids,before='',after='\n' if i==len(point_groups)-1 else ' '))
    if d['review']['closing']:
        blocks[-1]['after']='\n\n'
        blocks.append(dict(kind='closing',before='',after='\n'))
    d['review']['draft']=blocks


def finish(d, priority, conditional, evidence_request):
    r,spec = d['review'],d['spec']
    r['frame_variants'] = {
        'intro':[dict(id='specific',label='Thanks tied to this review',text=r['intro']),
                 dict(id='brief',label='Brief thanks',text=('Thank you for these detailed checks.' if spec['kind']=='ai'
                                                          else 'Thank you for the careful review.')),
                 dict(id='none',label='Omit opening',text='')],
        'closing':[dict(id='none',label='Omit closing',text=''),
                   dict(id='conditional',label='Conditional reconsideration' if spec['kind']=='human' else 'No score request',text=conditional),
                   dict(id='evidence',label='Request tied to the evidence' if spec['kind']=='human' else 'Evidence summary',text=evidence_request)]}
    RESPONSES['reviews'][spec['stem']]=r
    ANNOTATIONS['reviews'].append({**{k:v for k,v in spec.items() if k!='target'},'points':d['mapping']})
    lines=[f'# {spec["name"]} — {spec["reviewer"]}',
           f'Note ID: `{spec["note_id"]}`. Source: `{CONFIG["source"]["reviews_json"]}`.',
           '## Brief summary',d['overview'],'## Every concern / request']
    for i,(identity,title,detail) in enumerate(d['concerns'],1):
        lines.append(f'{i}. **{identity} — {title}.** {detail}')
    lines.extend(['## Response priorities',priority,'## Original review\n'+d['original']])
    OUTPUTS['reviews/'+spec['stem']+'.md']='\n\n'.join(lines).rstrip()+'\n'


GEOMETRY = ('G1: aaai_article/signmuon_addendum_body.tex, conditional SignMuon theorem and geometry lemma; '
            'OLion v2, Lemma 4.3 / Corollary 4.4; exact-oracle, fixed-within-run momentum, uniformly positive margin.')
JOINT = ('J1: evidence/EF21_JOINT_SMOOTHNESS.md, Sections 1–3 and bound (J), derived in this preparation; '
         'whole-tuple norm instantiates prior EF21; deterministic-schedule first-moment proof is explicit.')
RATES = ('T1: submitted supplement A.10; aaai_article/ef21_nesterov_convergence.tex, '
         'ef21_fixed_radius.tex, ef21_rate_takeaways.tex. Gradient-scaled steps and horizon-tuned momentum '
         'for the separate squared-moment result; never infer it by squaring a first moment.')
FED = ('E1: submitted Table A8 and code/results/federated_export_results.zip: '
       'table_federated.csv and final_*_unit-gain_e2000_fs{0..4}/seed*/metrics.json; '
       'evidence/verified_claims.json independently recomputes means/sample SDs and thresholds.')
SCOPE = ('E2: submitted supplement A.14–A.15 and code/federated/algorithms.py, main.py, methods.py. '
         'CNN2, N=11, homogeneous, full participation; 3 mini-batches of 64 at fixed weights, zero local updates. '
         'New heterogeneous runner exists in live repo, but no completed new result was available.')
LM = ('E3: submitted Table 3 and supplement A.16–A.17; code/results/nanogpt/runs.csv, diagnostics.csv, SUMMARY.md; '
      'one independent unpinned run per compressed method, full-precision gradient reduce-scatter before optimizer.')
COST = ('C1: submitted supplement communication appendix, code/federated/methods.py:communication_bits; '
        'algorithms.py estimator allocation; evidence/verified_claims.json communication calculation. '
        'Counts are simulated payloads, not measured packed-network throughput.')
NEG = ('N1: submitted Theorems 1–4 and proofs; aaai_article/ef21_signmuon_divergence.tex; '
       'evidence/verified_claims.json exact rational ascent/cycle certificates. '
       'Theorem 4 objective depends on fixed eta and mu; no decreasing-schedule impossibility follows.')
SOFT = ('N2: aaai_article/signmuon_addendum_body.tex, lower-bounded softplus witnesses and EF21 gradient-floor corollary. '
        'Only plain placements have one fixed softplus witness for all schedules; do not extend this to output EF21.')
PRIOR = ('P1: Gruntkowska et al. 2025 EF21-Muon, Algorithm 3 / Theorem 19 / Remark 23; '
         'Modoranu et al. ICML 2024; Mishra et al. SignMuon v2 (17 October 2026), Section 4 / Appendix F. '
         'The current main article comparison to Mishra v1 is stale and must be replaced.')


def r1():
    d=prepare(0,'The reviewer values the explicit negative examples, the distinction between EF targets, and the experimental breadth (6, confidence 4). The five questions concern a positive explanation for SignMuon, federation beyond the tested setting, attribution of EF21 results, practical costs, and language-model uncertainty.',
              'Thank you for the theory and practical questions.')
    a=r'For momentum $M=U\Sigma V^\top$, if each diagonal entry of $U^\top\mathrm{sign}(UV^\top)V$ is at least $\delta>0$, then $\langle M,\mathrm{sign}(UV^\top)\rangle\ge\delta\|M\|_*$. A uniform margin, joint smoothness and unbiased finite-variance noise yield a conditional SignMuon rate; the margin is unmeasured in our networks.'
    point(d,'R1.1','Reviewer asks for gradient structure that permits SignMuon descent','Explain a sufficient condition beyond the random-quadratic observation.', [('feedback','more intuition'),('weaknesses','central theory–practice')],GEOMETRY,'The diagonal-correlation condition implies alignment by weighting each diagonal entry by its nonnegative singular value. Uniform alignment, joint smoothness, centered finite-variance noise, exact polar and horizon-tuned fixed EMA/Nesterov yield an expected first-moment nuclear-gradient rate O(T^-1/4). It is a conditional result; actual trajectory verification and distributed-vote alignment remain open.',text=a)
    point(d,'R1.6','Reviewer requests neural-network alignment and spectral measurements','Measure the proposed condition along actual network trajectories.', [('weaknesses','empirically characterize')],GEOMETRY+' '+LM,'No neural-network measurement of the full geometric margin is available. Proposed diagnostics must record momentum/gradient alignment, diagonal correlations and spectral spread for the actual approximate solver and vote.',pending=True,include=False)
    point(d,'R1.2','Reviewer asks whether federation changes preserve the ordering','State what the 11-client experiment supports.', [('feedback','How broadly'),('weaknesses','idealized')],SCOPE,'Neither theory nor the existing experiments predict an ordering under label skew, local updates, partial participation or larger client counts. Three accumulated mini-batches are not local parameter steps.',text='The 11-client IID experiment takes no local updates. We claim no ordering under non-IID data, local training or partial participation.')
    point(d,'R1.7','Reviewer requests heterogeneous and local-training experiments','Broaden the federated evaluation beyond synchronized homogeneous clients.', [('weaknesses','FedAvg-style')],SCOPE,'No completed non-IID/local-training/partial-participation result is available in the audited evidence. A newly added heterogeneous runner is preparation, not experimental evidence.',pending=True,include=False)
    point(d,'R1.3','Reviewer asks which positive results are inherited from EF21','Credit the existing arbitrary-norm framework and isolate extensions.', [('feedback','genuinely new'),('weaknesses','framework')],PRIOR+' '+JOINT+' '+RATES,'Scaled-sign contractivity and the base positive convergence guarantee instantiate prior EF21-Muon. The new rebuttal analysis treats all layers as one arbitrary-norm block and proves joint-smooth deterministic-schedule first-moment bounds for fixed EMA/Nesterov and either downlink.',text='The base positive rates instantiate EF21-Muon. Our added analysis removes layer separability and treats fixed EMA/Nesterov with deterministic schedules; it does not certify finite polar approximations.')
    point(d,'R1.9','Reviewer asks what technical contribution is new','Distinguish the negative construction from positive framework instantiation.', [('weaknesses','methodological novelty'),('feedback','new technical arguments')],NEG+' '+SOFT,'The placement-specific ascent examples and smooth bounded-below output-EF limit cycle are the principal submitted new analysis. The new conditional SignMuon and whole-tuple EF21 derivations are follow-up results; the general EF21 framework is not ours.',text='The principal new submitted proofs are the placement-specific ascent examples and bounded-below output-EF divergence.')
    point(d,'R1.4','Reviewer asks how to choose a communication variant','Recommend choices with explicit evidence limits.', [('feedback','choose among')],COST+' '+FED,'Under an uplink constraint EF21-MuonUSign retains an exact/full-precision downlink; under a bidirectional constraint EF21-MuonSign compresses both sides. SignMuon is the empirical candidate if conditional rather than general guarantees are acceptable. No universal ranking is predicted.',text='Uplink-only: EF21-MuonUSign; both links: EF21-MuonSign for the analyzed repair, or SignMuon for the strongest observed accuracy.')
    point(d,'R1.10','Reviewer requests scales and memory in a comparison table','Account for payloads, precision, estimators and polar location.', [('weaknesses','auxiliary costs')],COST,'The internal method table is prepared in 02_REBUTTAL_PLAN.md. A compressed CNN2 channel costs 1.0870 bits/parameter for plain signs, 1.0871 for EF with FP32 auxiliary parameters and one FP32 scale per matrix. Each gradient EF method adds one client estimator and one global server estimator; two-sided EF additionally stores exact X separately from W.',text='CNN2 compressed payloads are 1.087 bits/parameter including FP32 auxiliaries; EF adds one scale/matrix, client/server estimators, and a separate server model for two-sided EF. Polar: clients for SignMuon, server for gradient EF.')
    point(d,'R1.5','Reviewer asks which nanoGPT differences are reliable','Separate descriptive gaps from statistical evidence.', [('feedback','How confident'),('weaknesses','single-seed')],LM,'EF21-SignMuon vs SignMuon is 3.2860 vs 3.2881, gap .0021. The approximate .003 upstream variation concerns Muon, not these variants. SignMuon vs SignSGD gap is .1168 nats, descriptive only; no variant-specific variance estimate or robust ranking.',text='nanoGPT: the 0.0021 gap is inconclusive. All compressed-method rankings are preliminary single-run observations; even larger gaps lack a variant-specific variance estimate.')
    point(d,'R1.8','Reviewer requests additional language-model seeds','Repeat training rather than repeated validation of one run.', [('weaknesses','additional seeds')],LM,'No additional independent compressed-method training runs are available. Repeated evaluations and upstream Muon runs cannot substitute for them.',pending=True,include=False)
    for i,ids in enumerate((['R1.1','R1.6'],['R1.2','R1.7'],['R1.3','R1.9'],['R1.4','R1.10'],['R1.5','R1.8'])):
        names=['SignMuon structure','federated scope','new analysis','practical choice','nanoGPT uncertainty']
        needles=['more intuition','How broadly','genuinely new','choose among','How confident']
        topic(d,'Q'+str(i+1),str(i+1)+'. Re '+names[i]+':','feedback',needles[i],ids)
    finish(d,'Follow Feedback 1–5. State the structural condition first, separate inherited rates from new negative proofs, and reserve requested new experiments as visible unfinished fragments. The internal table supplies complete costs; the export compresses the practical advice.',
           'If these clarifications address your five questions, we would appreciate your reconsideration.',
           'If the conditional descent result and explicit cost accounting resolve your main reservations, we would appreciate an updated assessment.')


def r2():
    d=prepare(1,'The reviewer sees sufficient experiments to establish the phenomenon and rates the paper 7 (confidence 4). They require concrete writing fixes; larger experiments and a positive explanation would strengthen the contribution but are explicitly optional.',
              'Thank you for distinguishing the required writing fixes from optional extensions.')
    writing=[
      ('R2.1','Reviewer asks to identify sign as an LMO earlier','Define the geometry in Section 2.', [('feedback','third point'),('weaknesses','sign is itself a LMO')], 'W1: submitted Section 2 and current signmuon_body.tex Section 4.1.',r'We will state in Section 2 that $-\mathrm{sign}(G)$ minimizes $\langle G,D\rangle$ over $\|D\|_\infty\le1$.'),
      ('R2.4','Reviewer asks to distinguish raw signs from scaled EF messages','Explain the EF scale when the variants are first introduced.', [('weaknesses','scale of the input')],COST,'Plain signs send bits only; EF sends the residual signs plus its mean absolute value per matrix. FP32 auxiliary parameters are also counted.'),
      ('R2.5','Reviewer asks to distinguish centralized formulas from distributed algorithms','Explain the N=1 correspondence and nonlinear aggregation.', [('weaknesses','as they seem to be centralized')],SCOPE+' '+LM,'Algorithms A2/A4/A5 define centralized placements. At N=1 the distributed definitions reduce to them; at N>1 local sign/LMO and majority voting define different algorithms. We will label both settings explicitly.'),
      ('R2.6','Reviewer questions the linear objective under lower boundedness','Explain why the linear tests do not satisfy Assumption 1.', [('weaknesses','Assumption 1')],NEG+' '+SOFT,r'The linear examples test descent outside Assumption 1; $f\to-\infty$ is literal there. We also proved lower-bounded replacements: $f(X)=\log(1+e^{\langle G,X\rangle})$ retains the plain placements’ ascent directions for every positive step schedule and EMA/Nesterov momentum.'),
      ('R2.7','Reviewer suggests consistent names for sign placements','Make the composition order readable without changing identities.', [('weaknesses','naming')], 'W1: current signmuon_body.tex equations (5)–(6); method table in 02_REBUTTAL_PLAN.md.',r'We will retain labels to keep tables/code traceable and add an operation-order legend: SignMuon = sign(polar(M)); MuonUSign = polar(sign(M)); MuonSign = sign(polar(sign(M))).')]
    for identity,title,detail,links,evidence,text in writing:
        point(d,identity,title,detail,links,evidence,text)
    point(d,'R2.2','Reviewer suggests larger centralized and federated experiments','Additional models and truly distributed language-model training would strengthen the evidence.', [('weaknesses','Scale of numerical')],SCOPE+' '+LM,'No new large-model or distributed-language-model run was completed. nanoGPT evaluates the centralized optimizer after full-precision averaging, so it cannot fill this request.',pending=True,include=False)
    point(d,'R2.3','Reviewer asks for progress on the theory-practice explanation','Give a positive condition without claiming a complete explanation.', [('weaknesses','Lack of resolution'),('feedback','first two')],GEOMETRY,r'With joint smoothness, unbiased finite-variance noise, exact polar and horizon-tuned steps/momentum, we prove SignMuon convergence if $\langle M,\mathrm{sign}(\mathrm{polar}(M))\rangle\ge\delta\|M\|_*$ uniformly. It suffices that all diagonal entries of $U^\top\mathrm{sign}(UV^\top)V$ exceed $\delta>0$ for $M=U\Sigma V^\top$. Verification in our networks remains open.')
    point(d,'R2.8','Reviewer wants a realistic explanation of observed success','Check the proposed sufficient condition on trained networks.', [('weaknesses','resolution')],GEOMETRY,'The conditional proof supplies a mathematical intermediate result. It does not establish that network momentum, approximate polar factors or distributed votes satisfy it.',pending=True,include=False)
    topic(d,'Q1','Re the required writing fixes:','feedback','third point',['R2.1','R2.4','R2.5','R2.6','R2.7'])
    topic(d,'Q2','Re larger experiments:','weaknesses','Scale of numerical',['R2.2'])
    topic(d,'Q3','Re a positive explanation:','weaknesses','Lack of resolution',['R2.3','R2.8'])
    finish(d,'Lead with the requested writing changes, including the naming decision. Defend why the linear examples are meaningful while supplying the new lower-bounded strengthening. Distinguish a conditional mathematical explanation from experimental verification.',
           'If these concrete writing fixes and the added positive condition address your concerns, we would appreciate reconsideration.',
           'We would welcome an updated assessment based on the lower-bounded witnesses and conditional SignMuon result, with broader experiments still open.')


def r3():
    d=prepare(2,'The reviewer finds the placement study useful (6, confidence 2), but identifies a ranking-changing table error, an experimental/theorem mismatch, separable smoothness, and an invalid first-to-second-moment inference. The “N/A” feedback field is not treated as a concern.',
              'Thank you for catching the table error and the invalid squaring argument.')
    point(d,'R3.2','Reviewer identifies the reversed MuonSign versus SignSGD ranking','Correct the main table and every affected comparison.', [('weaknesses','Table 2 reports')],FED,'Table 2 is stale. Five independent submitted runs, each averaged over the final five evaluations at the 2000-round horizon, give MuonSign 82.9424 ± 0.1868 and SignSGD 81.4356 ± 0.1466 sample SD. These agree with A8, reversing the main-table sentence. SignMuon 85.7156 ± .2444 versus Muon 85.9836 ± .2608; no significance claim.',text='Table 2 is stale. Recomputing the five submitted runs gives MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%, agreeing with A8: MuonSign is higher by 1.51 percentage points. SignMuon is 85.72±0.24% and Muon 85.98±0.26%. We will replace the main table and reverse the affected sentence; no new training was needed.')
    point(d,'R3.1','Reviewer asks which learning-rate rule proves the squared-gradient rate','Do not transfer the sharp-stepsize result to cosine schedules.', [('weaknesses','gradient-dependent')],RATES,'The squared dual-gradient rate O(T^-1/2) has a separate second-moment argument with gradient-scaled radii and horizon-tuned momentum. It does not follow for the cosine settings used in the experiments.',text=r'The squared-gradient $O(T^{-1/2})$ result uses gradient-scaled steps and a separate second-moment proof; it does not certify the experimental cosine schedule.')
    point(d,'R3.3','Reviewer identifies the separability forced by layer-wise smoothness','Replace the restriction with an explicit whole-model formulation.', [('weaknesses','forces separability')],JOINT,r'We removed separability by treating the whole model as one block with norm $\max_i\|X_i\|_{\rm op}/\lambda_i$ and dual $\sum_i\lambda_i\|G_i\|_*$, for fixed layer weights $\lambda_i>0$. Joint Frobenius smoothness suffices. The squared-rate formulation then uses one common radius scaled by this dual norm, changing the per-layer rule.')
    point(d,'R3.4','Reviewer rejects squaring an expected first-moment bound','Remove the unsupported inference and state two distinct results.', [('weaknesses','does not follow by squaring')],RATES+' '+JOINT,r'You are right: squaring $\mathbb E\|\nabla f\|$ does not bound $\mathbb E\|\nabla f\|^2$. We will delete that inference. For deterministic constant/cosine radii, our new joint-smooth bound gives a weighted first moment $O(T^{-1/4})$ when peak LR scales as $T^{-3/4}$ and $1-\mu=T^{-1/2}$. Fixed LR/momentum leave a residual bound.')
    point(d,'R3.5','Reviewer asks to distinguish ideal EF21 from the implementations','State the remaining oracle and optimizer qualifications.', [('weaknesses','experimental implementations')],JOINT+' '+LM,'The new deterministic-schedule bound allows fixed EMA/Nesterov and either downlink but assumes exact polar, no decoupled weight decay and all jointly analyzed updates of the specified form. Finite polynomial polar, momentum warmup and auxiliary Adam are not covered.',text='These statements concern exact polar factors, fixed-within-run momentum and the server iterate. Finite polar solvers, nanoGPT momentum warmup and auxiliary Adam remain outside the guarantees.')
    topic(d,'Q1','Re the table discrepancy:','weaknesses','Table 2 reports',['R3.2'])
    topic(d,'Q2','Re the positive guarantee:','weaknesses','gradient-dependent',['R3.1','R3.3','R3.5'])
    topic(d,'Q3','Re the moment inference:','weaknesses','does not follow by squaring',['R3.4'])
    finish(d,'Own the two errors directly. Give the independently reconstructed values first, then separate the sharp second-moment result, new jointly smooth first-moment bound, and remaining experimental gaps. No full-run coverage claim.',
           'If the verified table correction and the separate moment statements resolve these concerns, we would appreciate reconsideration.',
           'We would appreciate an updated assessment in light of the archived five-seed evidence and the removal of separability from the ideal EF21 analysis.')


def r4():
    d=prepare(3,'The reviewer is borderline (5, confidence 4). Questions 1 and 2 prioritize theorem coverage and inconsistent tables. Further concerns ask for positive SignMuon conditions, honest prior-work credit, explicit nanoGPT transport, and a clearer main text.',
              'Thank you for the precise questions on theorem coverage and reporting.')
    coverage=r'No complete network run is certified: all use finite polar solvers; nanoGPT also warms momentum and uses auxiliary Adam. The squared-gradient rate needs gradient-scaled steps. We now prove, under joint smoothness and unbiased bounded-variance noise, a constant/cosine first-moment bound for exact-polar EF21 with either downlink and fixed EMA/Nesterov. Fixed peak LR/momentum leave a residual; $T^{-3/4}$ peak LR and $1-\mu=T^{-1/2}$ give $O(T^{-1/4})$.'
    point(d,'R4.1','Reviewer asks whether the experimental schedules have the proved rate','Identify the distinction between squared-gradient steps and deterministic schedules.', [('feedback','squared-gradient guarantee')],RATES+' '+JOINT,'The original sharp-stepsize squared guarantee does not cover Tables 1–3. New joint-smooth deterministic-schedule proof (J) gives a residual or a horizon-tuned first-moment rate, not the advertised squared rate.',text=coverage)
    point(d,'R4.2','Reviewer asks whether nanoGPT Nesterov momentum is covered','Separate a fixed Nesterov coefficient from momentum warmup.', [('feedback','Nesterov')],JOINT+' '+LM,'Bound (J) permits fixed Nesterov; nanoGPT changes momentum during the run and is not covered.',text=coverage)
    point(d,'R4.3','Reviewer identifies inconsistent main and supplementary results','Reconstruct the five runs and correct the ranking.', [('feedback','I noticed some values')],FED,'The original submitted five-seed archive agrees with A8. Table 2 is stale; final-five-evaluation means give 82.9424 ± .1868 vs 81.4356 ± .1466. The main-table comparison must be reversed.',text='The submitted five-seed logs agree with A8: MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%; Table 2 is stale. We will replace it and reverse the ranking sentence. SignMuon remains 85.72±0.24%, versus Muon 85.98±0.26%.')
    geo=r'For $M=U\Sigma V^\top$, positive diagonal entries of $U^\top\mathrm{sign}(UV^\top)V$ bounded below by $\delta$ imply alignment at least $\delta\|M\|_*$. This yields conditional SignMuon convergence; its margin is unverified in our networks.'
    point(d,'R4.4','Reviewer asks for useful sufficient conditions for SignMuon','Supply a structural condition and a conditional result.', [('weaknesses','under what reasonable assumptions')],GEOMETRY,'The full addendum gives the weaker covariance/spread criterion epsilon*rho<1 and a complete stochastic fixed-momentum first-moment proof. The simpler diagonal lower bound in the export is sufficient and self-contained.',text=geo)
    point(d,'R4.15','Reviewer asks why the favorable quadratic alignment extends to neural networks','Distinguish a sufficient theorem from evidence along actual training trajectories.', [('weaknesses','random quadratic experiments')],GEOMETRY,'The sufficient margin has not been checked on these neural-network trajectories, approximate solvers or distributed votes. The favorable quadratic result does not establish it.',pending=True,include=False)
    point(d,'R4.6','Reviewer questions novelty relative to EF21 and concurrent SignMuon','Credit existing frameworks and the latest concurrent theory.', [('weaknesses','concurrent work')],PRIOR+' '+NEG,'The new submitted contribution is the negative sign-composition and output-EF analysis. The base EF21 theory and practical SignMuon precede this response. Mishra v2 now has conditional actual-vote stochastic momentum results, so our current v1 comparison must be replaced.',text='We credit EF21 and concurrent SignMuon, including Mishra v2’s conditional theory; our main novelty is the negative placement analysis.')
    lm='nanoGPT applies these optimizers after full-precision reduce-scatter. It tests centralized updates, not compressed worker votes or network savings.'
    point(d,'R4.5','Reviewer says nanoGPT does not test the proposed distributed compression','Make the full-precision transport explicit in the main text.', [('weaknesses','nanoGPT experiments do not actually')],LM,'The reviewer is correct. Full-precision gradient averaging precedes the owning-rank optimizer; parameter all-gather follows. Local compress-then-vote semantics and actual one-bit communication gains are not tested by this experiment.',text=lm)
    writing='We will put theorem coverage and X/W definitions in the main text and simplify the introduction, results and captions.'
    issues=[
      ('R4.7','Reviewer asks for definitions before theorem references','Introduce the operations before the introduction cites alignment results.', 'introduction, which refers'),
      ('R4.8','Reviewer objects to broad impossibility wording','Replace “unrepairable” with the specified fixed-parameter output-EF statement.', 'exactly the placement'),
      ('R4.9','Reviewer rejects presenting empirical ordering as a reversed theoretical prediction','The existence of failure examples does not predict method ranking.', 'Experiment orders'),
      ('R4.10','Reviewer asks for a direct conclusion without repetitive caveats','Remove repeated “document rather than resolve” language.', 'also appears repeatedly'),
      ('R4.11','Reviewer asks which of X and W is updated and analyzed','Explain exact server X, compressed broadcast W and their roles.', 'It is difficult to understand'),
      ('R4.12','Reviewer asks to explain target iteration counts before using them','Distinguish finite-target success and stationary error floors from asymptotic rate.', 'fixed-target iteration'),
      ('R4.13','Reviewer asks for a clearer statement of the Section 5.2 results','Lead with measured accuracy instead of theorem qualifications.', 'Sign-after is what'),
      ('R4.14','Reviewer asks to move essential explanations into the main paper','Prepare a coverage table and shorten captions.', 'important explanations')]
    for identity,title,detail,needle in issues:
        full=detail+' Prepared replacement prose and a theorem/implementation table are in 02_REBUTTAL_PLAN.md. X takes exact server updates; W is the compressed broadcast at which gradients are queried. Theorem 4 has parameter-dependent objectives and no decreasing-schedule conclusion. The theory makes no performance-order prediction.'
        point(d,identity,title,detail,[('weaknesses',needle)],NEG+' '+LM+' W1: 02_REBUTTAL_PLAN.md replacement prose.',full,text=writing)
    topic(d,'Q1','1. Re theorem coverage:','feedback','squared-gradient guarantee',['R4.1','R4.2'])
    topic(d,'Q2','2. Re inconsistent tables:','feedback','I noticed some values',['R4.3'])
    topic(d,'Q3','Re scope and novelty:','weaknesses','under what reasonable assumptions',['R4.4','R4.15','R4.6'])
    topic(d,'Q4','Re nanoGPT:','weaknesses','nanoGPT experiments do not actually',['R4.5'])
    topic(d,'Q5','Re presentation:','weaknesses','important explanations',[x[0] for x in issues])
    shared(d,[('Q1',[['R4.1','R4.2']]),('Q2',[['R4.3']]),('Q3',[['R4.4'],['R4.6']]),('Q4',[['R4.5']]),('Q5',[[x[0] for x in issues]])])
    finish(d,'Answer explicit Questions 1–2 first. Do not claim the new theory certifies finite polar solvers or changing nanoGPT momentum. State positive geometry and prior-work credit, then centralized transport. Concrete replacement prose remains available internally for all writing criticisms.',
           'If the verified table correction, explicit scope and added conditional result address your main concerns, we would appreciate reconsideration.',
           'We would appreciate reassessment based on the archived results and the new joint-smooth analysis, while retaining the stated implementation limitations.')


def r5():
    d=prepare(4,'The reviewer rates the paper 4 (confidence 3) and asks two direct questions: what is new beyond EF21, and what Theorem 4 actually rules out. They also correctly reject an inferred performance ranking and treating sufficient compressor conditions as necessary.',
              'Thank you for pressing us on novelty and the exact impossibility claim.')
    new='The new submitted analysis is the explicit ascent constructions and the smooth, lower-bounded output-EF limit cycle. It shows why tracking a polar direction can fail although gradient-side EF succeeds: the polar target can jump by constant order independently of the step size. The base positive rates instantiate EF21-Muon; we will credit that directly.'
    point(d,'R5.1','Reviewer asks what is new beyond the EF21-Muon framework','State the new proofs and credit inherited positive rates.', [('feedback','main new'),('weaknesses','existing EF21')],NEG+' '+PRIOR,'The submitted novelty is the negative sign-placement diagnosis and smooth bounded-below output-EF cycle. The proof separates the discontinuous polar target from the smooth gradient-side target; this is Muon/LMO-specific and not a new arbitrary-norm EF framework.',text=new)
    point(d,'R5.3','Reviewer asks what optimizer-design lesson the counterexamples provide','Explain what is tracked and the consequences of its variation.', [('weaknesses','optimizer design')],NEG+' '+COST,'The example proves that sharing one residual scale while tracking a rapidly changing polar output can create a persistent wrong-sign component. Gradient-level EF avoids this particular tracking obstacle under smoothness; the guarantee still requires its stated assumptions.',text=new)
    quant=r'Theorem 4 says $\forall L,\eta>0,\mu\in[0,1),\ \exists f_{L,\eta,\mu}$ causing divergence for the specified output-EF recursion with fixed parameters. It rules out universal constant-step rules based only on $L,\mu$. It does not cover decreasing schedules or rule out horizon-tuned finite-time rates: the objective depends on the parameters, and its gradient floor scales as $L\eta$.'
    point(d,'R5.2','Reviewer asks whether Theorem 4 includes decreasing learning rates','Write the quantifiers and identify their limit.', [('feedback','counterexample in Theorem 4'),('weaknesses','Fixed-step divergence')],NEG+' '+SOFT,'For each fixed L, eta and mu there is an objective and either momentum variant diverges. The smooth objective may depend on those parameters. The scaled gradient floor is Omega(c_mu L eta), not a horizon-independent positive constant. No decreasing-schedule result is established for output EF.',text=quant)
    point(d,'R5.4','Reviewer says parameter-dependent divergence need not preclude finite-time rates','Acknowledge the dependence of the constructed objective and its constants.', [('weaknesses','problem can depend')],SOFT,'The new floor calculation makes the limitation explicit. It shrinks with eta and does not defeat horizon-tuned convergence theorems whose objective gap and constants are fixed.',text=quant)
    point(d,'R5.5','Reviewer rejects a theoretical prediction of the empirical ranking','Separate worst-case statements from comparative accuracy.', [('weaknesses','theoretical prediction')],GEOMETRY+' '+SOFT,'The theory does not predict which optimizer wins on a task. The positive SignMuon theorem is conditional and compatible with the counterexamples. A new fixed softplus objective proves arbitrary-schedule nonstationarity only for each plain placement; this does not strengthen the output-EF quantifiers.',text=r'The theory predicts no empirical ranking. Our added SignMuon analysis gives a first-moment rate under a uniform positive alignment margin, not a predicted winner. Separately, new fixed softplus witnesses prove failure for every positive step schedule and EMA/Nesterov momentum for the three plain placements; this stronger statement does not extend to output EF.')
    point(d,'R5.6','Reviewer objects to presenting sufficient compressor assumptions as necessary','Retract claims that only changing compressors can help.', [('weaknesses','necessary conditions')],NEG+' '+LM,'Failure of spectral contractivity does not prove nonconvergence under every analysis or parameter choice. c_proj concentration is observational. No initialization/step/compressor ablation proves causation or uniqueness of a remedy.',text='We will remove “unrepairable” and “only changing the compressor”: failing a sufficient condition excludes neither other analyses nor other parameter choices.')
    topic(d,'Q1','1. Re contributions beyond EF21:','feedback','main new',['R5.1','R5.3'])
    topic(d,'Q2','2. Re Theorem 4:','feedback','counterexample in Theorem 4',['R5.2','R5.4'])
    topic(d,'Q3','Re ranking and necessity:','weaknesses','theoretical prediction',['R5.5','R5.6'])
    shared(d,[('Q1',[['R5.1','R5.3']]),('Q2',[['R5.2','R5.4']]),('Q3',[['R5.5'],['R5.6']])])
    finish(d,'Prioritize the two explicit questions. Defend the concrete output-tracking failure as the submitted novelty. Agree with the precise logical objections and avoid using the new plain-method softplus result to overstate the EF result.',
           'If this precise contribution and corrected theorem scope resolve the central objections, we would appreciate reconsideration.',
           'We would appreciate reassessment based on the explicit new negative proofs, with the EF21 attribution and impossibility claims narrowed as above.')


def ai():
    d=prepare(5,'The AI review values the LMO-specific negative examples and output-tracking diagnosis. Its substantive requests concern stochastic and multilayer assumptions, exact-oracle coverage, X versus W, novelty and modern sign baselines, distributed scope, presentation, and five minor reporting details. Headings, praise and bibliography entries are not concerns.',
              'Thank you for the detailed technical checks.')
    theory=r'A.10 states unbiased bounded-variance noise; we will add it to the main statement. One max-spectral norm on all layers removes separability. Our new deterministic-schedule bound allows fixed EMA/Nesterov and either EF21 downlink, but not finite polar solvers, changing momentum or auxiliary Adam.'
    point(d,'AI.1','Reviewer asks for the missing stochastic assumptions','State the noise assumptions where the theorem is stated.', [('review','stated assumptions')],RATES+' '+JOINT,'Unbiased bounded nuclear-variance assumptions are in submitted A.10 but missing from the advertised main statement. The new proof uses conditional zero mean and a joint Frobenius variance bound.',text=theory)
    point(d,'AI.2','Reviewer says the layer smoothness forces separability','Use jointly coupled smoothness instead of the original restrictive assumption.', [('review','stated assumptions')],JOINT,'The whole-tuple max-spectral norm and sum-nuclear dual permit arbitrary inter-layer coupling under ordinary joint Frobenius smoothness. This reduction may worsen dimension constants.',text=theory)
    point(d,'AI.7','Reviewer asks for an inexact polar-oracle guarantee','Do not identify the finite polynomial solver with the exact polar theorem.', [('review','exact-oracle guarantees')],JOINT+' '+GEOMETRY+' '+LM,'There is no trajectory-wide oracle-error bound for the five-step solvers. The new addendum provides a conditional weighted sign-mismatch penalty, not a certificate that it vanishes in the actual experiments. Full-run theory coverage remains open.',text=theory)
    xw='EF21-MuonSign losses are 5.5198 at server X and 3.3213 at broadcast W. The guarantee concerns X; success at W does not validate it.'
    point(d,'AI.3','Reviewer says the guaranteed iterate differs from the queried model','Explain X versus W without transferring the guarantee.', [('review','controls a different iterate')],LM+' '+JOINT,'X takes the exact server update; W tracks it through compressed downlink; workers evaluate gradients at W. The guarantee is stationarity at X, not W.',text=xw)
    point(d,'AI.4','Reviewer identifies the large nanoGPT X/W loss gap','Acknowledge that the practically successful model is not the guaranteed one.', [('review','5.5198')],LM,'The two reported losses are evaluations of the same EF21-MuonSign run, not two independent runs. The gap is 2.1985 nats; c_proj residual concentration is an observation, not a causal initialization/compressor ablation.',text=xw)
    novel='We credit EF21-Muon and Modoranu et al.; the submitted novelty is the negative sign-placement/output-EF analysis.'
    point(d,'AI.5','Reviewer asks to frame positive EF21 results as an instantiation','Identify the paper-specific contribution.', [('review','framework instantiation')],PRIOR+' '+NEG,'The arbitrary-norm framework and general idea of gradient-side EF before a matrix transform are prior work. The new submitted proofs concern sign/polar ascent and output-EF divergence.',text=novel)
    point(d,'AI.18','Reviewer cites gradient-side EF before full-matrix preconditioners','Acknowledge Modoranu et al. explicitly.', [('review','Modoranu et al. (2024)')],PRIOR,'Modoranu et al. 2024 establishes a precedent for EF compression before a structured preconditioner. It does not supply the sign/polar negative construction; cite both facts accurately.',text=novel)
    point(d,'AI.6','Reviewer requests the modern SSVR sign baseline','Compare with Jiang et al. rather than claiming all gains arise from geometry.', [('review','Plain SignSGD is insufficient')], 'P2: Jiang et al., NeurIPS 2024, Efficient Sign-based Optimization; code/common/optimizers.py SignSGD baseline.', 'No SSVR run has been completed. Our SignSGD already uses momentum, but this does not substitute for a variance-reduced baseline; add the citation and remove exclusive causal attribution.',pending=True,include=False)
    point(d,'AI.19','Reviewer rejects attributing gains exclusively to matrix geometry','Explain the existing momentum baseline and the missing comparison.', [('review','exclusively to matrix geometry')], 'P2: code/common/optimizers.py and code/federated/algorithms.py; Jiang et al. 2024.', 'The existing SignSGD baseline uses the same EMA momentum coefficient. Thus the comparison is not against memoryless gradient signs, but it does not isolate geometry against all variance-reduction choices.',text='SignSGD uses momentum; SSVR is untested. We will cite Jiang et al. and remove exclusive attribution to geometry.')
    scope='Federation uses 11 IID clients, no local updates. nanoGPT averages full-precision gradients; no distributed one-bit gains are demonstrated.'
    point(d,'AI.8','Reviewer asks to limit broad federated claims','State the actual data and update regime.', [('review','narrow distributed regime')],SCOPE+' '+LM,'No non-IID/local-update result is completed; nanoGPT uses full-precision gradient transport and centralized optimizer semantics.',text=scope)
    point(d,'AI.9','Reviewer requests a method matrix','Prepare the matrix of operations, payloads, states and guaranteed iterate.', [('review','compact method matrix')],COST+' W1: 02_REBUTTAL_PLAN.md method matrix.', 'A concrete table is prepared internally with all six methods, EF targets, polar location, payloads, memory and theoretical iterate. It should replace the current scattered explanations.',text='We will add the method/coverage table and move implementation details to the experiments.')
    point(d,'AI.11','Reviewer asks to move Section 4.1 implementation details','Restore a clear sequence of definitions and theoretical results.', [('review','theory would read more linearly')], 'W1: 02_REBUTTAL_PLAN.md proposed presentation changes.', 'Move solver and layer-scaling details from Section 4.1 to the experimental setup. Keep theory definitions before counterexamples and EF results.',text='We will add the method/coverage table and move implementation details to the experiments.')
    geo=r'A sufficient SignMuon condition is $[U^\top\mathrm{sign}(UV^\top)V]_{jj}\ge\delta>0$ for $M=U\Sigma V^\top$; it implies alignment $\ge\delta\|M\|_*$.'
    point(d,'AI.10','Reviewer asks for a structural condition explaining positive alignment','Give a sufficient condition with explicit scope.', [('review','intermediate structural condition')],GEOMETRY,'The diagonal lower bound implies alignment by a singular-value-weighted sum. The full covariance criterion and stochastic first-moment theorem are in the author addendum. Credit OLion v2; it is not an empirical explanation for our networks.',text=geo)
    point(d,'AI.20','Reviewer asks to connect the structural condition to realistic training','Measure the margin for the actual polar approximation and vote.', [('review','high-dimensional noise')],GEOMETRY+' '+LM,'The geometric criterion has not been measured on the actual trajectories. Xiao et al. 2025 concerns SignSGD and does not by itself justify the SignMuon condition.',pending=True,include=False)
    minor='We will define polar(0)=0. The 0.0021 gap is inconclusive. A5 reports seconds/epoch; we drop the broad 2x claim. Payloads include FP32 auxiliaries/EF scales. Corrected MuonSign/SGD runs reach 80% in all 5 seeds.'
    point(d,'AI.13','Reviewer asks for the zero-input oracle selection','Define the exact thin-SVD selection and distinguish sign ties.', [('review','selection for')], 'W1: thin-SVD convention in aaai_article/ef21_signmuon_divergence.tex; code/common/optimizers.py.', 'Use polar(0)=0 in the exact theory. For plain strict-binary sign ties at zero, the implementation can randomize; this is distinct from the polar selection.',text=minor)
    point(d,'AI.14','Reviewer questions the single-seed 0.0021 nanoGPT ordering','Do not infer a reliable ranking from that difference.', [('review','selection for')],LM,'The gap EF21-SignMuon 3.2860 vs SignMuon 3.2881 is below an upstream Muon noise reference, not a variant-specific noise estimate. All compressed-method ranks are preliminary.',text=minor)
    point(d,'AI.15','Reviewer asks for timing evidence behind the wall-clock claim','Use submitted timing data with the correct scope.', [('review','selection for')], 'E4: submitted supplement Table A5, seconds/epoch (SignMuon16.6, SGD12.2, SignSGD12.5, Adam12.7); centralized export table_cifar.csv.', 'Elapsed seconds/epoch are already in A5, so timing data are not entirely absent. The old main-table threshold numbers are stale. Withdraw the generic approximately 2x end-to-end speed claim and report per-epoch timing/threshold separately; compilation/validation and network transport differ across settings.',text=minor)
    point(d,'AI.16','Reviewer asks to count scales in one-bit claims','Include one scale per matrix and full-precision auxiliary parameters.', [('review','selection for')],COST,'CNN2 has 762560 matrix and 2146 auxiliary parameters and 3 compressed blocks. Bits/channel are 1.0870 plain and 1.0871 EF, not exactly one per total parameter; uplink-only round-trip gain is 1.93x, not 29.4x.',text=minor)
    point(d,'AI.17','Reviewer flags rounds-to-target conditioned on successful seeds','Correct the stale main table and report successes.', [('review','selection for')],FED,'Corrected submitted archive: MuonSign rounds80 860, SGD1240, both 5/5 successes; SignSGD1140 and SignMuon540, also5/5. Do not use the stale main-table 4/5 and3/5 denominators.',text=minor)
    point(d,'AI.12','Reviewer asks to scope impossibility to the analyzed EF repair','Avoid ruling out all repairs and decreasing schedules.', [('review','Impossibility language')],NEG+' '+SOFT,'Theorem4 is forall fixed L,eta,mu exists parameter-dependent f for the specified scaled-sign output estimator. It does not rule out horizon-tuned finite-time rates, other EF schemes or different analyses.',text='“Unrepairable” will be limited to the specified output-EF counterexample; decreasing schedules and other repairs are not ruled out.')
    topic(d,'Q1','Re guarantees:','review','stated assumptions',['AI.1','AI.2','AI.7'])
    topic(d,'Q2','Re X/W:','review','controls a different iterate',['AI.3','AI.4'])
    topic(d,'Q3','Re attribution:','review','framework instantiation',['AI.5','AI.18','AI.6','AI.19'])
    topic(d,'Q4','Re scope:','review','narrow distributed regime',['AI.8'])
    topic(d,'Q5','Re clarity:','review','compact method matrix',['AI.9','AI.11','AI.10','AI.20','AI.12'])
    topic(d,'Q6','Re details:','review','selection for',['AI.13','AI.14','AI.15','AI.16','AI.17'])
    shared(d,[('Q1',[['AI.1','AI.2','AI.7']]),('Q2',[['AI.3','AI.4']]),('Q3',[['AI.5','AI.18'],['AI.19']]),('Q4',[['AI.8']]),('Q5',[['AI.9','AI.11'],['AI.10'],['AI.12']]),('Q6',[['AI.13','AI.14','AI.15','AI.16','AI.17']])])
    finish(d,'Prioritize valid theorem/implementation qualifications and X/W, then attribution/baselines, scope and factual minor corrections. No score request. Modern SSVR training and trajectory-margin diagnostics remain visibly unfinished.',
           'Thank you for the detailed checks; the remaining experimental gaps are stated explicitly.',
           'The archived seed records settle the reporting corrections; the new analysis retains the implementation qualifications above.')


def main():
    r1();r2();r3();r4();r5();ai()
    OUTPUTS['response_fragments.json']=js(RESPONSES)
    OUTPUTS['annotations.json']=js(ANNOTATIONS)
    OUTPUTS['evidence/concern_evidence.json']=js(RECORDS)
    # Assert no author text/notes or prior revision is being silently replaced.
    baseline=json.loads((BACKUP/'response_fragments.json').read_text(encoding='utf-8'))
    for r in baseline['reviews'].values():
        assert r['revision']==0 and not r['intro'] and not r['closing']
        assert all(not f['text'] for f in r['fragments'])
    # Import the paper's unmodified assembler for exactly the browser's count.
    import os
    os.environ['REBUTTAL_WORKSPACE_ROOT']=str(BASE)
    sys.path.insert(0,str(BASE/'primer/runtime'))
    from assemble_rebuttals import compile_review
    over=[]
    for stem,r in RESPONSES['reviews'].items():
        result=compile_review(r)
        print(stem,result['characters'], 'selected',len(result['included']),'pending',result['pending'])
        if result['issues']:
            over.append((stem,result['issues'],result['body']))
    if over:
        for stem,issues,body in over:
            print(stem,issues,'\n'+body)
        raise SystemExit('Adjust source wording before writing.')
    record_path=BASE/'evidence/preparation_outputs.json'
    previous=json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else {}
    for name,text in OUTPUTS.items():
        target=BASE/name
        if target.exists():
            allowed=previous.get(name)
            initial=BACKUP/name
            assert (allowed and sha(target.read_bytes())==allowed) or (initial.is_file() and target.read_bytes()==initial.read_bytes()), 'Author edit detected: '+name
    if '--write' not in sys.argv:
        print('Dry run. Pass --write to install the checked preparation.')
        return
    for name,text in OUTPUTS.items():
        target=BASE/name;target.parent.mkdir(exist_ok=True,parents=True)
        with target.open('w',encoding='utf-8',newline='\n') as stream:
            stream.write(text)
    record_path.write_text(js({name:sha((BASE/name).read_bytes()) for name in OUTPUTS}),encoding='utf-8')
    print('Prepared',len(RECORDS),'concrete concerns. Original review blocks preserved; baseline backup retained.')


if __name__=='__main__':
    main()
