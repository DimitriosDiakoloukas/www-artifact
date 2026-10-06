# Execution and design disclosures

The primary and native protocols were locked before their runs. The layerwise follow-up was designed
from PRIMARY DEVELOPMENT failures of equal truncation, before any held-out approximation result.
It uses a new 80-query validation sample excluding both primary calibration subsets; epoch selection
uses the same validation split, so this is independent policy selection, not a new model-validation set.
Both complete policy files and their commits were locked before either held-out candidate file existed.
No held-out outcome triggers further tuning, checkpoint selection, candidate removal or design changes.

The larger learned controls retain all 12 outcomes, including five runs below the stated solved
threshold. They use the frozen 200-epoch budget and validation selection. Accuracy on resampled controls
is evaluated against the new labels, not the original labels. Thresholded test accuracy does not certify
correctness or a margin on every neighbour. Known exact functions and trained models are separate evidence.

The initial isolated-component numerical bridge failed its fixed 2e-6 absolute/relative criterion on
one representative perturbed SGCN query. The failure log and superseded partial audits remain in
large_controls; COMPONENT_VALIDATION_DEVIATION.json records the event. Version 2 preserves the tolerance,
models, features, draws and training selections. It checks original plus four representative perturbations
before measurement and uses the unchanged full-union encoder and all-query head for any query that fails.
Final analyses read audit_v2 only. Three of 240 queries use this full-function fallback. The other
components have exactly zero sign gradients; component probing exploits disconnected locality with frozen
BatchNorm statistics, and does not change the graph semantics or SIDNET's global initialisation rows.
Stopped wrapper processes allowed existing bounded children to finish, and replacement workers waited
for those children before resuming; original attempts and logs are retained.

A policy-test fixture used a Python integer rather than a NumPy scalar and failed before selection.
The fixture was corrected before any primary policy was fitted or locked. No experiment data changed.
Matched constrained sampling explicitly records infeasible exchanges and zero-change/capped cycle draws.
A collector correction separates infeasible requests from feasible no-change draws; no sampler changed.

After all held-out candidate measurements completed, runtime wrappers were redistributed over eight
V100 cards. Four parent wrappers stopped while their current benchmark children drained; replacement
workers wait for any old child on their GPU and skip existing complete files. This changes scheduling
only. No GPU runs concurrent timed jobs. Each checkpoint's full baseline and candidates use the same
GPU; cards have 16 or 32 GB, recorded in environment metadata. All calibration ledgers are unchanged.

Auxiliary logit effects, fixed-original-label AUC, correctly classified control strata and native
structural/path-agreement strata are descriptive analyses added after intervention outcomes.
They do not alter any intervention, query draw, accuracy threshold or policy. Natural edited-label
truth is unavailable, so intervened AUC is explicitly evaluated against unchanged original labels.

After runtime measurement, CPU policy-selection cost was measured by replaying the unchanged selectors
on disposable validation copies. All locked decisions/parameters reproduce exactly; held-out files are
not copied or read. The whole 12-model selection-process wall time is conservatively charged to EACH
checkpoint, added to end-to-end development costs in the collector. The raw benchmark records are
unchanged. This accounts for imports, hash checking, validation scoring and selector fitting without
claiming new policies or tuning after held-out outcomes.
