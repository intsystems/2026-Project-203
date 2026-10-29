"""Read-only audit of submitted artifacts, repository versions, and result bundles.

Writes derived evidence beside this script. Never extracts executable archive paths.
"""
from pathlib import Path
import csv
import difflib
import hashlib
import io
import json
import statistics
import zipfile

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parent
OUT = BASE / "evidence"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    manifest = json.loads((BASE / "inputs_manifest.json").read_text(encoding="utf-8"))
    inputs = []
    for entry in manifest["files"]:
        data = (BASE / entry["path"]).read_bytes()
        actual = sha(data)
        assert actual == entry["sha256"], entry["path"]
        inputs.append({**entry, "verified_sha256": actual})
    archive = zipfile.ZipFile(next(BASE.glob("*Code And Data Supplement.zip")))
    mapping, changes = [], []
    for member in archive.infolist():
        if member.is_dir():
            continue
        submitted = archive.read(member.filename)
        target = (REPO / member.filename).resolve()
        assert REPO.resolve() in target.parents
        row = {"member": member.filename, "bytes": len(submitted), "submitted_sha256": sha(submitted)}
        if not target.is_file():
            row["status"] = "missing_in_repo"
        else:
            current = target.read_bytes()
            row["repo_sha256"] = sha(current)
            row["status"] = "identical" if submitted == current else "different"
            if row["status"] == "different" and target.suffix in {".py", ".md"}:
                before = submitted.decode("utf-8-sig").splitlines(keepends=True)
                after = current.decode("utf-8-sig").splitlines(keepends=True)
                changes.extend(difflib.unified_diff(before, after, fromfile="submitted/"+member.filename,
                                                   tofile="repo/"+member.filename))
        mapping.append(row)
    write_json("source_audit.json", {"inputs": inputs, "code_archive_members": mapping})
    (OUT / "submitted_to_repo.diff").write_text("".join(changes), encoding="utf-8")

    # Save the already-submitted aggregate and individual seed records as evidence.
    fed = zipfile.ZipFile(io.BytesIO(archive.read("code/results/federated_export_results.zip")))
    subdir = OUT / "submitted_federated"
    subdir.mkdir(exist_ok=True)
    for name in ("SUMMARY.md", "table_federated.csv", "rule_ablation.csv", "communication.csv", "runs.csv"):
        (subdir / name).write_bytes(fed.read("federated_export/" + name))
    rows = list(csv.DictReader(io.StringIO(fed.read("federated_export/table_federated.csv").decode())))
    selected = {r["algorithm"]: r for r in rows}
    verification = {}
    for method in ("muonsign", "signsgd", "signmuon", "muon", "ef21muonusign", "ef21muonsign", "sgd"):
        seed_records = []
        for seed in range(5):
            name = f"federated_export/runs/final_{method}_unit-gain_e2000_fs{seed}/seed{seed}/metrics.json"
            data = json.loads(fed.read(name))
            seed_records.append({"seed": seed, "archive_member": name, "metrics": data})
        verification[method] = {"aggregate": selected[method], "seed_records": seed_records}
    write_json("federated_seed_records.json", verification)

    # Record aggregate equality separately from compressed-container bytes.
    nested = {}
    repo_fed = zipfile.ZipFile(REPO / "code/results/federated_export_results.zip")
    for name in fed.namelist():
        if name.endswith("/"):
            continue
        a = fed.read(name)
        b = repo_fed.read(name) if name in repo_fed.namelist() else None
        nested[name] = {"submitted_sha256": sha(a), "repo_sha256": sha(b) if b is not None else None,
                        "identical_bytes": a == b,
                        "identical_after_newline_normalization": b is not None and a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")}
    write_json("federated_bundle_comparison.json", nested)
    print("Verified original hashes:", len(inputs))
    print("Archive comparison:", {s: sum(r["status"] == s for r in mapping)
                                   for s in ("identical", "different", "missing_in_repo")})
    print("Submitted final federated results:")
    for method, item in verification.items():
        print(method, item["aggregate"]["acc_mean"], item["aggregate"]["acc_std"],
              "seed metric keys:", list(item["seed_records"][0]["metrics"]))


if __name__ == "__main__":
    main()
