# Remaining scientific questions: frozen follow-up

1. Reconstruct x8 detection-selected curves from the completed independent
606-position direct-reference cohort, not a fitted correction. Propagate the
existing joint reference bands, uncertain detection membership and positive
finite-population stratified bootstrap through median, ratio-of-medians and
sum-ratio depth spreads. Compare median depth spread minus sum-ratio spread
with joint cross-depth resampling. Report broad or inconclusive intervals.
The frozen surrogate is unchanged; demonstrable prediction errors remain errors.

2. Retry every unresolved physical state in the original independent 32,768
particle global and 32,768 particle outer cohorts. Include 32 old successful
controls per branch/cohort. Retain AUTO tEqGuess=None, smallabd=1e-15,
tol=1e-3, maxTime=1e18 and existing damping factors. Increase only mxstep from
5000 to 50000 and whole-state wall budget from 600 to 1800 s. Require strict
final CO/13CO/C18O success. Report every control disagreement and unresolved
state. Preserve old files and produce separate overlays; controls never replace
old labels. Independently evaluate element-constrained luminosity envelopes
before/after recovery using the exact implemented LVG escape law, allowing
negative optical depth. Include CMB absorption in signed lower bounds. These
envelopes assume physical individual isotope budgets and normalised nonnegative
level populations; they are not universal maser or physical-model error bounds.
Propagate conservative approximate sampling endpoint bands into global source/
exchange contrasts, outer slopes and the fixed SFR slope catalogue. A wide box
or nonpositive total is an unresolved result, not evidence of negligible light.

3. Test frozen predictions directly at x8 outputs 551, 563 and 567, spanning
the early, middle and later parts of the existing short sequence. Draw 32,768
particles independently per output with known positive inclusion probabilities
within 16 radial layers × three density classes: 1024 per inner radial layer,
3072 per outer layer (including R>=15). Streaming random-priority reservoirs
are fixed before direct labels. Use that output's already frozen oxygen scale,
Full/SLUG reconstruction and global uniform isotope ratios. No model fitting.
All five physical states are solved with the standard AUTO/1e-15/1e-3,
mxstep5000, maxTime1e18, 600 s protocol. Exact time-independent adjoints are
applied to new coordinates/kernels, retaining native/lattice and normalized/
zero-padded operator distinctions. Evaluate global source and exchange signs,
outer slopes and direct/surrogate precision, conditional on qualified states.
The initial 491,520 new states may establish signs without meeting every local
precision goal. Additional allocation is decided from the results, never by
retuning acceptance limits. These times are correlated, not independent galaxies.

Local checks must cover exact finite priority-reservoir sampling, the actual
LAMDA transitions and analytic LVG bounds, survey covariance, contrast extrema,
and import/AST/shell checks without running chemistry on login. Freeze all code,
settings and inherited solver identities. PBS uses mk27/rsaa: sixteen direct
partitions per batch (104 CPUs,64GB,8h), temporal preparation (16 CPUs,64GB,4h),
geometry (32 CPUs,128GB,4h), and small review jobs. Hold, inspect effective
resources/dependencies, then release. Submit dependent reviews but do not wait
for computational completion. Pending results do not enter the paper.

Physical cloud chemistry-timescale work remains explicitly skipped; observation
absolute-intensity/statistical matching remains deferred. Neither is silently
restarted by this plan. Public reproduction excludes upstream RoSE code and
snapshots; large paper-derived inputs are indexed and preserved on Gadi.
