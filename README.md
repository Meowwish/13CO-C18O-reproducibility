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

All four Gadi data archives passed payload and checksum verification. A clean
requested-data replay passed the global-integral reconstruction, 1064 and 1220
independent interval checks, six frozen models on 32,768 particles, and all
2352 selected-statistic records. See `results/clean_replay_report.json` and
`RELEASE_STATUS.json`. These are numerical reproduction checks, not physical
accuracy claims. Pending new scientific follow-ups are labelled under `ongoing/`
and do not change the accepted numerical results of this release.
