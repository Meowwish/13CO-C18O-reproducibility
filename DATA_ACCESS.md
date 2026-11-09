# Data access and retention

Contact **Chuhan Zhang, chuhan.zhang@anu.edu.au**, stating the release ID and
the bundle IDs required. Large paper-derived datasets are maintained on NCI
Gadi under `/g/data/jh2/cz4203/13CO_C18O_reproducibility/`, separately from the
working scratch tree. They are available from the corresponding author on
request. No automated email is sent by these tools.

`DATA_CATALOGUE.json` lists sizes and packaging status. Each completed bundle
is accompanied by its SHA256 hash and a file-level manifest; `tools/unpack_bundle.py`
validates both. Data IDs refer to the pinned paper release, not mutable scratch
filenames. New scientific follow-ups receive a distinct release when accepted.

Available categories are cached probability samples/direct labels/operators,
frozen models and fitting labels, the complete prepared x8 chemistry-input
frame, and cached maps. Request only the categories needed for the intended
reproduction level. Cases can be identified by cohort, particle ID, source
part/row, branch and stored inclusion probability.

The repository and request bundles exclude RoSE hydrodynamic source and raw
snapshots. Access to those upstream materials is a separate matter and is not
promised here. No unrelated project data, credentials or scheduler environment
dumps are distributed.
