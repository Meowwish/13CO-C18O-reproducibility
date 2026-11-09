"""Rerun all selected-statistic intervals using the supplied portable workflow."""
from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', type=Path)
    parser.add_argument('--allow-local', action='store_true', help='Explicitly permit substantial cached analysis outside PBS')
    args = parser.parse_args(); work = args.workflow.resolve()
    if not os.environ.get('PBS_NCPUS'):
        assert args.allow_local, 'Use PBS on Gadi, or explicitly --allow-local on your own machine.'
        os.environ['PBS_NCPUS'] = '1'
    sys.path.insert(0, str(work/'x8_selection_reference_20261108'))
    # Import the immutable numerical implementation; override only root and
    # provenance-location bookkeeping. No physics/inference is added.
    sys.path.insert(1, str(work/'x8_aperture_validation_20261105'))
    import selection_common as c
    import selection_review as review
    source = work/'x8_selection_reference_20261108'; out = work/'selected_replay'
    out.mkdir(exist_ok=True)
    assert not (out/'review.json').exists(), 'Use a fresh replay workspace.'
    for name in ('sample.h5', 'direct'):
        target = out/name
        if not target.exists():target.symlink_to(source/name, target_is_directory=name=='direct')
    original = json.loads((source/'review.json').read_text())
    prepared = json.loads((source/'prepared.json').read_text())
    assert c.sha(source/'sample.h5') == prepared['sample_sha256'] == original['sample_sha256']
    c.ROOT = out
    c.ready = lambda:(original['manifest_sha256'], prepared)
    review.main()
    replay = json.loads((out/'review.json').read_text())
    assert len(replay['selected_statistics']) == len(original['selected_statistics']) == 2352
    for group in ('selected_statistics', 'selected_chains', 'selection_migration', 'pixel_checks'):
        assert len(replay[group]) == len(original[group]), group
        for a, b in zip(replay[group], original[group]):
            for key in a:
                if isinstance(a[key], (float, int)) and a[key] is not None:
                    np.testing.assert_allclose(a[key], b[key], rtol=2e-10, atol=2e-12)
                elif isinstance(a[key], list):
                    try:
                        av, bv = np.asarray(a[key], dtype=float), np.asarray(b[key], dtype=float)
                    except (ValueError,TypeError):assert a[key] == b[key], (group, key)
                    else:np.testing.assert_allclose(av, bv, rtol=2e-10, atol=2e-12, equal_nan=True)
                else:assert a[key] == b[key], (group, key)
    print('All 2352 selected-statistic records and joint-reference calculations replayed.')


if __name__ == '__main__':main()
