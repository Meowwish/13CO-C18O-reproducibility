# Reproduction levels

Use Python 3.11. `requirements.txt` records the versions in the tested Gadi
environment. On a Gadi login node first run `module load python3/3.11.0`.
Use PBS for model replay, chemical solves, projection and substantial cached
analysis. No RoSE source or raw simulation snapshot is required for levels 1–2.

## 1. Small published results

```
python reproduce.py --plots
```

This verifies the committed file manifest, reconstructs six full-frame ratios
from saved complete-frame line integrals, rebuilds the selected-precision table
from its interval records, exports direct outer profiles and the other structured
results to CSV, and makes an example profile figure. It is not an independent
re-solve of particles. Sample/covariance and frozen-model replay are level 2.

## 2. Cached reference statistics and frozen models

Request `cached-validation` and `models-training` for the release in
`DATA_CATALOGUE.json`. The author supplies archive and per-file SHA256 hashes.
Unpack into a fresh directory:

```
python tools/unpack_bundle.py cached-validation.tar external --sha256 HASH_FROM_AUTHOR
python tools/unpack_bundle.py models-training.tar external --sha256 HASH_FROM_AUTHOR
cd external/paper-e9e9d9e-20261109/workflow
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python global_ratios.py
python x8_claim_checks_20261105_v3/reproduce.py
python x8_outer_validation_20261105/reproduce.py
python query_models.py
```

The two independent audit programs reconstruct 1064 and 1220 interval sets
using saved direct labels, known inclusion probabilities and measurement
operators. Model replay evaluates six frozen predictions on 32,768 particles;
it does not fit them again. Original model paths embedded in the residual
bundle are replaced by verified relative paths in this adapter.

From the repository root, the full 606-position selection calculation can be
rerun with the same random seed and all 2352 interval records:

```
python tools/replay_selection.py external/paper-e9e9d9e-20261109/workflow
```

On Gadi this command requires PBS. `--allow-local` explicitly permits it on a
separate machine. `selected_replay/` must be fresh for each run. Frozen sources
are preserved; only root-location and output bookkeeping are adapted. The
provided hashes identify bytes, not physical accuracy.

## 3. Models, maps and chemistry from paper-derived inputs

Request `derived-particle-inputs` for the complete prepared x8 input frame and
`maps` for the cached projected maps. `models-training` also contains the
derived original-550/575 fitting labels and split data. These are paper-derived
files, not raw RoSE outputs. The exact solver, isotope chemistry extensions,
model implementation, sampling/measurement code and settings are retained in
`workflow/`; frozen manifests retain original absolute paths as provenance.
Relocate configurations and verify the requested file manifest before executing
a workflow. Historical submit/build scripts are not an automatic entry point.

For the manuscript figures, place a checkout of the manuscript repository at
the pinned paper commit into the requested `workflow/Paper/` directory. The
retained figure recipes use the supplied maps, result tables and frozen
references. Stylistic/PDF metadata differences need not reproduce identical
PDF bytes. Do not interpret original experimental drafts as the accepted result.

The currently running `science_closure_20261109` follow-up is separately marked
as pending and is excluded from the result claims of this release. Its exact
code and plan are provided for provenance. No reproduction mode reruns the
RoSE hydrodynamics, verifies physical chemical-timescale ordering, or creates
new-time accuracy evidence before its direct jobs have been assessed.
