# Preparing this directory as anonymous supplementary material

```bash
cd code
python3 -m anonymize --check                        # scan; exits 1 on a hit
python3 -m anonymize --build ../supplement.zip      # clean bundle for submission
python3 -m anonymize --build-dir ../anonymous_code  # the same bundle, unpacked
```

`--build-dir` writes exactly what `--build` would, as a directory to be read
through before it is archived — step 3 below is much easier on a tree than on a
zip, and `test_the_zip_and_the_directory_are_the_same_bundle` pins the two to the
same file list so that the object inspected is the object submitted. It refuses a
destination that is not empty rather than merging into an older build.

`--check` also runs inside `tests/test_code.py`, so a new author name or absolute
path in a docstring fails the test suite rather than reaching a reviewer.

## What the scan looks for

| rule | why |
| :--- | :--- |
| `project-identifier` | An author name, handle, institution or repository belonging to this project |
| `email` | Any email address |
| `home-path` | `/home/<user>`, `/Users/<user>`, `C:\Users\<user>` |
| `project-repo` | A link to this project's own repository |
| `orcid` | An ORCID identifier |

`python3 -m anonymize --list-rules` prints the patterns. The notebook-output
handling in `anonymize.py` is retained but currently inert: the repository holds no
`.ipynb`.

## Two defences, not one

The scan is the gate. The second defence is that the export bundles carry no
absolute path in the first place: a run tree records where it ran (`state.json`
stores a log and a metrics path per job, the exporters stamp their source tree
into a manifest), and on a Linux box those begin with a home directory.
`common/paths.py:repo_relative` rewrites them to `results/...` as they are
written, and `test_export_bundles_carry_no_absolute_path` asserts that nothing
under `results/` carries one. That matters because `--build` excludes `results/`,
so a bundle copied by hand would otherwise bypass the gate entirely.

## What is deliberately *not* removed

* **Upstream URLs and third-party author names.** The modded-nanogpt repository
  this port is built on, the Polar Express implementation it borrows, the
  contributors credited in comments: citations, not self-identification, and
  removing them would misattribute the work. Only *this* project's repository is
  flagged.
* **Hardware and driver strings** in the nanoGPT run logs (`NVIDIA H100 80GB`,
  `Driver Version: …`). The reproducibility checklist asks for exactly this, and
  it identifies a GPU model, not a person.
* **Dates.** They constrain the submission window, not the authors.

If a hit is one of these, add it to `ALLOW` in `anonymize.py` **with a one-line
reason**, so that silencing something stays a visible decision.

## What was found and fixed

The 2026-07-27 run of the scan found four kinds of leak, none of which a checklist
would have caught:

| where | what | fix |
| :--- | :--- | :--- |
| `common/models.py` ×2 | A docstring crediting a colleague by first name | rewritten to describe the models |
| `nanogpt/`, one module | A **filename** containing a first name, plus four references to it | renamed; both files later deleted as superseded |
| a nanoGPT notebook | An output cell holding an absolute home-directory path | stripped at bundle time; no notebooks remain |
| `results/*/MANIFEST` and `state.json` | Generated files quoting an absolute source path | excluded from the bundle, and since 2026-07-31 rewritten at the source by `repo_relative` |

Two are worth dwelling on, being the ones a manual pass misses. A name inside a
**filename** does not match a `\b…\b` regex, since `\b` counts `_` as a word
character; the rule uses letter/digit lookarounds instead. And a name inside a
**notebook output** is invisible to anyone reading the source. Both were found
only by grepping the built zip independently of the tool that built it, which is
why step 3 below is not optional.

The 2026-07-31 sweep found the fourth kind at scale: 2,044 absolute paths across
the results tree and the three export archives, 510 of them in one `state.json`.
All were rewritten, and the exporters now rewrite as they write.

Russian-language comments in `common/models.py` and the nanoGPT scaffolding were
translated at the same time. Not identifying on their own, but they narrow the
author pool for no benefit.

## What the bundle contains

`--build` writes a zip with `code/` at the top level, and a `MANIFEST.md`
recording what was left out, so the omissions are visible rather than silent.

**Excluded**: caches (`__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`),
datasets (`data/`, `data_federated/`, the FineWeb shards), the raw per-run trees
and the unpacked bundles (`results/federated/`, `results/synthetic/`,
`results/article_export/`, `results_old/`), model checkpoints (`*.pt`), and stray
`*.log` and `*.stackdump` files.

**Excluded, and this file with them** (`ANONYMITY_TOOLING` in `anonymize.py`):

| withheld | why |
| :--- | :--- |
| `anonymize.py` | its content *is* the list of names that must not ship |
| `tests/test_anonymize.py` | quotes those rules, and imports the scanner |
| `ANONYMIZATION.md` | a write-up of what leaked, leaks — the table below names a colleague's role, a filename, and the language the comments were written in |

They go as one unit, because each gives away what the others hide. Nothing in the
bundle references them: `test_code.py` adopts the anonymity tests only when the
module is there, the `code/README.md` pointer to this file lives in the repository
README instead, and `test_the_bundle_references_nothing_it_withholds` fails if any
shipped import, link or documented command stops resolving.

**Included, deliberately**, carved out of the blanket `results/**` rule:

* **The three export archives** — `synthetic_results.zip`,
  `article_export.tar.gz`, `federated_export_results.zip`. `REPRODUCE.md` §§3–5
  all say *compute on the GPU box, bring back one archive, plot anywhere*; these
  are the archives, so a reviewer redraws every CIFAR, federated and synthetic
  table and figure without a GPU-night. The archive rather than the tree it came
  from, because an exporter decides what belongs in it — `federated`'s drops runs
  made under a superseded sign convention and records each exclusion with its
  reason, and `results/federated/` holds 66 such runs.
* **`results/nanogpt/`** — eight logs that cost an 8×H100 node and cannot be
  regenerated by anything here. They are the evidence behind the
  language-modelling table and embed the training script and the optimizer
  definitions verbatim.

The scan opens the archives: `scan_tree` reads every text member of an archive
that ships, so a zip cannot smuggle past the gate what a loose file could not.
Archives that stay out of the bundle are not opened — what they hold cannot reach
a reviewer, and failing the gate on a stale bundle nobody sends is noise.

### Git revisions are redacted at bundle time

The exporters stamp the commit each run was made at into its `metrics.json`,
`configs.json`, `environment.json` and `MANIFEST.json` — right for a run tree, and
what the reproducibility checklist asks a *published* artifact to record. In the
anonymous bundle it is the one identifying field the scan cannot argue about: a
40-hex SHA is not a name, an email or a path, so no rule in `PATTERNS` fires on
it, and the federated bundle carried the repository's then-current `HEAD`. Paste
that into a commit search and you have the repository, its name and its authors.

So `build_bundle` and `build_dir` rewrite every such field to
`<redacted for review>` — in loose files and inside the archives, which are
rebuilt member by member for the purpose — and `MANIFEST.md` records that they
did. It costs the supplement nothing: a revision resolves only against a
repository the reviewer does not have. Your own `results/` tree keeps the real
hashes; only what ships is redacted.

The rule keys on the *field name*, not on "looks like hex", so a run identifier
such as `EF21-MuonSign_lr0.06_e6770317.txt` survives.
`test_the_bundle_records_no_git_revision` checks both the redactor and the built
bundle.

## Before submitting

1. `python3 -m anonymize --check` → `OK`.
2. `python3 -m tests.test_code` → all pass.
3. Build, then **grep the built artifact itself** rather than trusting the
   builder. Recurse into the nested export archives; they are most of the
   generated text in the bundle, and generated text is where the leaks were:

   ```bash
   python3 -m anonymize --build ../supplement.zip
   python3 - <<'PY'
   import re, tarfile, zipfile, io
   BIN = ('.png', '.pdf', '.jpg', '.pt', '.bin', '.npz')
   pat = re.compile(r'(/home/(?!<user>)|/Users/(?!<user>)|[A-Za-z]:.{0,2}Users'
                    r'|@[\w.-]+\.(com|org|ru))', re.I)

   def texts(name, data):                      # (label, text) for one entry
       if name.endswith('.zip'):
           with zipfile.ZipFile(io.BytesIO(data)) as z:
               for i in z.infolist():
                   if not i.is_dir():
                       yield from texts(f'{name}:{i.filename}', z.read(i))
       elif name.endswith(('.tar.gz', '.tgz')):
           with tarfile.open(fileobj=io.BytesIO(data)) as t:
               for m in t.getmembers():
                   if m.isfile():
                       yield from texts(f'{name}:{m.name}', t.extractfile(m).read())
       elif not name.endswith(BIN):
           yield name, data.decode('utf-8', 'replace')

   z = zipfile.ZipFile('../supplement.zip')
   hits = [(n, i, l) for e in z.namelist() if not e.endswith('/')
           for n, t in texts(e, z.read(e))
           for i, l in enumerate(t.splitlines(), 1) if pat.search(l)]
   print(f'{len(z.namelist())} entries, {len(hits)} suspicious lines')
   for h in hits[:20]:
       print(' ', h)
   PY
   ```

   Everything it prints should be a deliberate test fixture carrying an
   `anonymize: allow` pragma, or prose about *not* collecting a hostname.
4. If you ship anything from `results/` outside the bundle, scan it too. The
   bundle carries the three export archives and `nanogpt/`; the raw trees are
   excluded, and `--scrub-results --write` is what makes one safe to hand over.
5. Check the paper's own PDF metadata separately. This tool covers `code/` only,
   and LaTeX embeds the author in `/Author` and `/Title` unless told otherwise.

## The limits of this

The scan is a regex sweep. It will not catch a self-citation phrased as "our
earlier work", an acknowledgements paragraph, a distinctive variable name, or a
screenshot. It covers `code/`, not the paper, not `git` history, and not the
repository the code is hosted in. Treat it as the floor, not the ceiling.
