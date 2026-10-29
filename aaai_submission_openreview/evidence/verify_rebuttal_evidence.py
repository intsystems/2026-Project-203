"""Reproduce rebuttal numbers/certificates without training or altering sources."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import csv
import hashlib
import io
import json
import statistics
import tarfile
import zipfile

import numpy as np
import sympy as s
import fitz

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    audit = json.loads((BASE / "evidence/source_audit.json").read_text())
    for entry in audit["inputs"]:
        assert digest((BASE / entry["path"]).read_bytes()) == entry["sha256"]
    archive = zipfile.ZipFile(next(BASE.glob("*Code And Data Supplement.zip")))
    fed = zipfile.ZipFile(io.BytesIO(archive.read("code/results/federated_export_results.zip")))
    current = zipfile.ZipFile(REPO / "code/results/federated_export_results.zip")
    histories = []
    for name in fed.namelist():
        if not name.endswith("metrics.json"):
            continue
        a, b = json.loads(fed.read(name)), json.loads(current.read(name))
        assert a["history"] == b["history"], name
        histories.append(name)
    records = json.loads((BASE / "evidence/federated_seed_records.json").read_text())
    results = {}
    for method, data in records.items():
        values, crossing = [], []
        for record in data["seed_records"]:
            h = record["metrics"]["history"]
            values.append(statistics.mean(h["test_acc"][-5:]))
            crossing.append(next((t for t, a in zip(h["steps"], h["test_acc"]) if a >= 80), None))
        mean, sd = statistics.mean(values), statistics.stdev(values)
        row = data["aggregate"]
        assert abs(mean - float(row["acc_mean"])) < .000051
        assert abs(sd - float(row["acc_std"])) < .000051
        success = [t for t in crossing if t is not None]
        assert len(success) == int(row["seeds_reaching_target"])
        assert abs(statistics.mean(success) - float(row["rounds_to_target"])) < .00001
        results[method] = dict(seed_tail_means=values, mean=mean, sample_sd=sd,
                               first_crossings=crossing, successful_seeds=len(success),
                               mean_rounds_80=statistics.mean(success))

    # Exact ascent certificate: rational O, rational rank-one term u v^T.
    O = s.Matrix([[101, 20, 2, -2], [-20, 97, 20, -20],
                  [-2, 20, 2, 101], [-2, 20, -101, -2]]) / 103
    u, v = s.Matrix([10, -3, 10, 10]), s.Matrix([10, 3, -10, 10])
    G = 1000 * u * v.T / 309 + O
    assert O.T * O == s.eye(4) and O * v == u
    # O^T G = I + 1000 vv^T/309 is positive definite; polar(G)=O exactly.
    signO = O.applyfunc(s.sign)
    pair1 = s.trace(G.T * signO)
    assert pair1 == -s.Rational(42468, 103)

    S = s.Matrix([[-1,-1,1,1,1], [-1,-1,1,-1,-1], [1,-1,1,1,-1],
                  [1,1,-1,-1,1], [1,1,1,-1,1]])
    A = S.T * S
    # Degree-two spectral interpolation for (S^T S)^(-1/2), exact radicals.
    eigen = [s.Integer(4), (13-3*s.sqrt(17))/2, (13+3*s.sqrt(17))/2]
    roots = [s.Rational(1,2), (s.sqrt(17)+3)/4, (s.sqrt(17)-3)/4]
    invroot = s.zeros(5)
    for i, val in enumerate(eigen):
        projector = s.eye(5)
        for j, other in enumerate(eigen):
            if i != j:
                projector = projector * (A-other*s.eye(5))/(val-other)
        invroot += roots[i]*projector
    P = (S * invroot).applyfunc(s.simplify)
    assert (P.T * P - s.eye(5)).applyfunc(s.simplify) == s.zeros(5)
    assert P[3,1] == -1/s.sqrt(17)
    G5 = S.copy(); G5[3,1] = 100
    pair2 = s.simplify(s.trace(G5.T * P))
    pair3 = s.trace(G5.T * P.applyfunc(s.sign))
    assert pair2 < 0 and pair3 == -76

    # Theorem 4's finite rational tracking calculation, distinct from the
    # smooth interpolation/existence argument in its proof.
    q = s.Rational
    targets = [s.Matrix([[-q(4,5),q(3,5)],[-q(3,5),-q(4,5)]]),
               s.Matrix([[q(3,5),-q(4,5)],[0,0]])]
    targets += [s.Matrix([[q(7,25),(-1)**t*q(24,25)],
                          [(-1)**t*q(24,25),-q(7,25)]]) for t in range(3,11)]
    d, states = s.zeros(2), []
    for target in targets:
        r = target-d
        assert all(x != 0 for x in r)
        alpha = sum(abs(x) for x in r)/4
        d += alpha*r.applyfunc(s.sign)
        states.append(d.copy())
    dB = s.Matrix([[-61,-201],[-61,-61]])/200
    dA = s.Matrix([[131,-9],[131,131]])/200
    for i in range(2,len(states)):
        assert states[i] == (dB if i%2 == 0 else dA)
    assert (dA[1,1]+dB[1,1])/2 == q(7,40)

    # Norm duality, joint-smoothness constants, and the exact EMA variance
    # identity used by EF21_JOINT_SMOOTHNESS.md. These are algebra/sanity
    # checks; the document's proof, rather than these samples, establishes J.
    rng = np.random.default_rng(48060)
    norm_checks = 0
    for _ in range(80):
        shapes = [(2,3),(3,2),(1,4)]
        lam = np.array([.7,1.2,.4])
        xs = [rng.normal(size=z) for z in shapes]
        gs = [rng.normal(size=z) for z in shapes]
        B = np.sqrt(sum(l*l*min(z) for l,z in zip(lam,shapes)))
        pnorm = max(np.linalg.norm(x,2)/l for x,l in zip(xs,lam))
        fro = np.sqrt(sum(np.sum(x*x) for x in xs))
        dual = sum(l*np.linalg.svd(g,compute_uv=False).sum() for g,l in zip(gs,lam))
        gfro = np.sqrt(sum(np.sum(g*g) for g in gs))
        assert lam.min()*pnorm <= fro+1e-12 <= B*pnorm+2e-12
        assert dual <= B*gfro+1e-12
        norm_checks += 1
    beta, theta = s.symbols('beta theta', positive=True)
    mu = 1-beta
    k = beta+mu*theta
    assert s.simplify(k+(1-theta)*mu-1) == 0
    assert s.simplify(beta**2/(1-mu**2)-beta/(2-beta)) == 0

    pmat, paux, layers = 762560, 2146, 3
    ptotal = pmat+paux
    bplain = (pmat+32*paux)/ptotal
    bef = (pmat+32*paux+32*layers)/ptotal
    comm = dict(matrix_parameters=pmat, auxiliary_parameters=paux, layers=layers,
                plain_bits_per_channel=bplain, ef_bits_per_channel=bef,
                both_channel_reduction_plain=32/bplain,
                uplink_only_round_trip_reduction=64/(32+bef),
                bytes_per_extra_fp32_matrix_buffer=4*pmat)

    versions = []
    for path in sorted((REPO/'aaai_article').iterdir()):
        if path.suffix not in ('.tex','.bib','.pdf'):
            continue
        row = dict(path=path.relative_to(REPO).as_posix(), sha256=digest(path.read_bytes()),
                   bytes=path.stat().st_size, modified_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())
        if path.suffix == '.pdf':
            doc = fitz.open(path)
            row.update(pages=len(doc), metadata=doc.metadata,
                       plain_text_sha256=digest(''.join(p.get_text() for p in doc).encode('utf-8')))
        versions.append(row)
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), original_hashes_verified=len(audit['inputs']),
                  federated_results=results, submitted_and_repo_identical_training_histories=len(histories),
                  exact_certificates=dict(signmuon_pairing=str(pair1), muonusign_pairing=str(pair2),
                                          muonusign_pairing_float=float(pair2), muonsign_pairing=str(pair3),
                                          output_ef_cycle_mean_22='7/40'),
                  joint_norm_sanity_samples=norm_checks, communication=comm, article_versions=versions,
                  limits=['No new training runs.', 'Finite rational checks do not replace the smooth interpolation proof.',
                          'Norm samples do not replace the analytic proof of bound J.',
                          'Current PDFs and TeX are separately hashed; PDF freshness is not assumed.'])
    (BASE/'evidence/verified_claims.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS: original hashes, five-seed means/SDs/thresholds,',len(histories),'unchanged training histories, exact ascent/cycle certificates, norm identities.')
    for name,row in results.items():
        print(name, f"{row['mean']:.4f} +/- {row['sample_sd']:.4f}; rounds80={row['mean_rounds_80']:g}; {row['successful_seeds']}/5")


if __name__ == '__main__':
    main()
