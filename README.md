# Exact account-level audits of signed Web graphs

Reproducibility code and saved experimental evidence for *Who Can Overturn a
Trust Prediction? Exact Account-Level Audits of Signed Web Graphs*.

## Run

Use Python 3.12:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_inputs.py
python verify.py
```

The download step retrieves five public datasets from Stanford SNAP and checks
pinned SHA-256 hashes. Verification then runs on a CPU without network access
or robust-model retraining. Allow approximately ten minutes, depending on
hardware. Success ends with `VERIFICATION PASS`; a failed check exits nonzero.

Verification checks file integrity, small-graph proof and solver tests,
600 solver witnesses, six ordinary model refits, every MILP-path witness and
every 25th remaining ungated audit. It also recomputes validation-selection
sensitivity, paired uncertainty estimates, committed-routing results and
numerical summaries from saved query records. It does not refit the robust or neural models.

## Code and results

| Location | Purpose |
|---|---|
| [experiments/hybrid-auditor/hybrid.py](experiments/hybrid-auditor/hybrid.py) | Exact structural branches and rational confirmation of MILP witnesses |
| [experiments/hybrid-auditor/ungated.py](experiments/hybrid-auditor/ungated.py) | Ordinary four-walk model fitting and complete test-query audits |
| [experiments/audit-core/cover_learn.py](experiments/audit-core/cover_learn.py) | Robust four-walk training |
| [experiments/balanced-evaluation/balanced.py](experiments/balanced-evaluation/balanced.py) | CBA metrics and validation-selected operating points |
| [experiments/hybrid-auditor/selection-uncertainty/analysis.py](experiments/hybrid-auditor/selection-uncertainty/analysis.py) | Selection sensitivity and paired dyadic intervals |
| [experiments/committed-routing/analysis.py](experiments/committed-routing/analysis.py) | Committed routing between certified walk predictions and the protected source; its tables are regenerated into `runs/v1/DIAGNOSTIC.md` |
| [experiments/](experiments/) | Baselines, controls, temporal studies, scalability, models and query records |
| [store/](store/) | Fixed splits, processed inputs, features and fitted objects |
| [src/](src/) | Data parsing and record-integrity utilities |

Each study's `runs/` directory holds its locked source/input hashes, models,
query results and completion records. `SUMMARY.json` and `BALANCED.json` contain
aggregate results. Training and audit settings, seeds and selection decisions
are recorded with the runs. The selection-sensitivity and committed-routing analyses are
adaptive follow-ups to earlier results; they fit no model, and their only new
choice (the history router) uses validation data only.

Exact CBA excludes unconfirmed numerical certificates. The paired intervals
condition on fitted models, selection and the evidence graph, assume
uncorrelated influence for disjoint query dyads, and exclude refitting uncertainty.

## External inputs and attribution

The datasets are Bitcoin-Alpha and Bitcoin-OTC (trust ratings), Wiki-Elec
(adminship votes), and Epinions and Slashdot (signed social relations).
Sources: [Bitcoin-Alpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html),
[Bitcoin-OTC](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html),
[Wiki-Elec](https://snap.stanford.edu/data/wiki-Elec.html),
[Epinions](https://snap.stanford.edu/data/soc-sign-epinions.html) and
[Slashdot](https://snap.stanford.edu/data/soc-sign-Slashdot090221.html).
Please follow the original providers' terms and citation requests.

The neural comparison uses [RIDGE](https://github.com/Junranus/RIDGE), by
Wu, Ooi and Xu, *Toward Robust Signed Graph Learning through Joint Input-Target
Denoising* (ACM Multimedia, 2025). Its source is not redistributed here. For
neural refitting, `python fetch_inputs.py --ridge` retrieves the pinned upstream
file and checks its hash. The upstream terms apply; this artifact grants no
license to third-party material. Saved neural results remain available for
comparison, and no robustness certificate is claimed for that baseline.

[MANIFEST.json](MANIFEST.json) records distributed file hashes.
[PROVENANCE.json](PROVENANCE.json) records public-path and metadata
normalization. Experimental numeric values and retained binary inputs are
preserved. [RELEASE.json](RELEASE.json) identifies this distribution.
