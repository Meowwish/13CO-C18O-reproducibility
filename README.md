# Reproducing the CO isotope line-ratio paper

This repository contains only the analysis and chemistry extensions, frozen
settings, result tables, numerical audits and figure recipes needed for
*From stellar enrichment to 13CO/C18O line ratios: controlled experiments in a
Milky Way-like disc*. The manuscript repository is
https://github.com/Meowwish/13CO-C18O.

RoSE simulation source, raw snapshots and unrelated project/development files
are excluded. Large paper-derived numerical inputs, reference labels, projection
operators and frozen models are managed separately on NCI Gadi. Their catalogue
provides checksums and request instructions; contact chuhan.zhang@anu.edu.au
for the required paper-derived bundles. The repository does not promise access
to RoSE upstream data.

The numerical release is pinned to manuscript commit
`e9e9d9efd3440be0dde9a739ec34bddeb35a54f6`. Run `python reproduce.py --plots`
for small-table reconstruction; see [REPRODUCING.md](REPRODUCING.md) for cached
reference/model replay and [DATA_ACCESS.md](DATA_ACCESS.md) for the requested
Gadi bundles. `FILE_SHA256.json` identifies the committed files.

The large-data packaging and clean replay are scheduled separately on PBS;
their current state is recorded in `RELEASE_STATUS.json`. Pending scientific
follow-ups are labelled under `ongoing/` and do not count as completed
validations or change the published numerical results of this release.
