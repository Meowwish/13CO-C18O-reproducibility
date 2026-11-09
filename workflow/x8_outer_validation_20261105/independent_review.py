"""Reconstruct cached radial statistics without importing the submitted review.

Small cached probability sample only. No chemistry, production map projection,
training, or fitting runs on login. Scalar influence variances independently
check the stored covariance propagation, including paired and radial contrasts.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
ISO = ROOT.parent
MODELS = ('production_full', 'production_slug', 'actual_on', 'actual_off', 'uniform_on', 'uniform_off')
BRANCH = [0, 1, 0, 2, 3, 4]
PAIRS = dict(source=(1, 0), exchange=(2, 3), uniform_exchange=(4, 5),
             isotope_pattern_on=(2, 4), isotope_pattern_off=(3, 5))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    report = json.loads((ROOT/'review.json').read_text())
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, digest in manifest['hashes'].items():
        assert sha(path) == digest, path
    assert sha(ROOT/'manifest.json') == report['manifest_sha256']
    with h5py.File(ROOT/'sample.h5') as f:
        n = len(f['sample/mass'])
        ids = f['sample/particle_id'][:]
        strata = f['sample/stratum'][:]
        population, allocation = f['stratum_population'][:], f['stratum_sample'][:]
        pi = f['sample/inclusion_probability'][:]
        nuclei = f['sample/nuclei'][:]
        proxy = f['sample/lum_per_H'][:, :, 1:3]
    np.testing.assert_allclose(pi, allocation[strata]/population[strata], rtol=0, atol=0)
    np.testing.assert_array_equal(np.bincount(strata, minlength=len(population)), allocation)
    assert population.sum() == 43133718 and n == 32768
    assert sha(ROOT/'sample.h5') == report['sample_sha256']
    values = np.full((5*n, 2), np.nan)
    qualified = np.zeros(5*n, bool)
    seen = np.zeros(5*n, bool)
    failures, timeouts, reused = 0, 0, 0
    output_hashes = {}
    for part in range(8):
        path = ROOT/f'direct/part{part:02d}.npz'
        output_hashes[str(path)] = sha(path)
        with np.load(path) as f:
            task = f['task_ids']
            metadata = json.loads(str(f['metadata']))
            assert bool(f['complete']) and metadata['manifest_sha256']==report['manifest_sha256']
            assert metadata['sample_sha256']==report['sample_sha256']
            assert not seen[task].any() and np.all(task % 8 == part)
            np.testing.assert_array_equal(f['particle_id'], ids[task//5])
            status = f['status']
            q = (status == 1) & np.all(f['final_returns'] == 1, axis=1)
            np.testing.assert_array_equal(q, f['qualified'])
            assert np.all(status != 0) and np.all(np.isfinite(f['values'][q]))
            values[task], qualified[task], seen[task] = f['values'][:, 7:9], q, True
            failures += int((status == -1).sum())
            timeouts += int((status == -2).sum())
            reused += int((f['origin'] == 1).sum())
    assert seen.all()
    values, qualified = values.reshape(n, 5, 2), qualified.reshape(n, 5)
    np.testing.assert_array_equal(qualified.sum(0), report['qualified_states_by_branch'])
    regions = list(dict.fromkeys(r['region'] for r in report['model_checks']))
    kernel = np.full((n, 4, len(regions)), np.nan)
    seen = np.zeros(n, bool)
    for part in range(4):
        path = ROOT/f'geometry/part{part:02d}.npz'
        output_hashes[str(path)] = sha(path)
        with np.load(path) as f:
            assert bool(f['complete']) and str(f['manifest_sha256'])==report['manifest_sha256']
            assert str(f['sample_sha256'])==report['sample_sha256']
            selected = f['ids']
            assert not seen[selected].any()
            kernel[selected], seen[selected] = f['kernel'], True
    assert seen.all() and np.all(np.isfinite(kernel)) and np.all(kernel >= 0)
    groups = [(np.flatnonzero(strata == h), int(population[h])) for h in np.unique(strata)]
    def variance(scores):
        answer = np.zeros(scores.shape[1:])
        for indices, N in groups:
            nh = len(indices)
            if nh == N:
                continue
            centered = scores[indices]-scores[indices].mean(axis=0)
            answer += nh*(1-nh/N)*np.sum(centered*centered, axis=0)/(nh-1)
        return answer
    def measure(model, good, geom):
        k = kernel[:, 2*geom+int(model >= 2)]
        wk = k*(nuclei/pi)[:, None]
        d = np.where(good[:, None], values[:, BRANCH[model]], 0.)
        p = np.where(good[:, None], proxy[:, model], 0.)
        D = np.stack([np.sum(wk*d[:, line, None], axis=0) for line in range(2)], axis=1)
        P = np.stack([np.sum(wk*p[:, line, None], axis=0) for line in range(2)], axis=1)
        ld, lp = np.log10(D[:, 0]/D[:, 1]), np.log10(P[:, 0]/P[:, 1])
        sd = wk*(d[:, 0, None]/D[None, :, 0]-d[:, 1, None]/D[None, :, 1])/np.log(10)
        sp = wk*(p[:, 0, None]/P[None, :, 0]-p[:, 1, None]/P[None, :, 1])/np.log(10)
        positive = wk[:, :, None]*np.maximum(d[:, None, :], 0.)
        ess = positive.sum(0)**2/np.sum(positive**2, axis=0)
        light = wk[:, :, None]*np.maximum(proxy[:, model, None, :], 0.)
        missing = light[~good].sum(0)/light.sum(0)
        return ld, lp, sd, sp, ess, missing
    checks, maximum_point, maximum_se = 0, 0., 0.
    def compare(record, point, var):
        nonlocal checks, maximum_point, maximum_se
        se = float(np.sqrt(max(0., var)))
        maximum_point = max(maximum_point, abs(record['point']-point))
        maximum_se = max(maximum_se, abs(record['standard_error']-se))
        np.testing.assert_allclose(record['point'], point, rtol=0, atol=3e-12)
        np.testing.assert_allclose(record['standard_error'], se, rtol=0, atol=3e-12)
        np.testing.assert_allclose(record['interval_95'], [point-1.96*se, point+1.96*se], rtol=0, atol=3e-12)
        checks += 1
    model_rows = {(r['geometry'], r['model'], r['region']): r for r in report['model_checks']}
    pair_rows = {(r['geometry'], r['comparison'], r['region']): r for r in report['paired_checks']}
    for geom, label in enumerate(('native', 'lattice')):
        for model, name in enumerate(MODELS):
            m = measure(model, qualified[:, BRANCH[model]], geom)
            d, p, sd, sp, ess, missing = m
            vd, ve = variance(sd), variance(sd-sp)
            for j, region in enumerate(regions):
                row = model_rows[label, name, region]
                compare(row['direct_log_energy_ratio'], d[j], vd[j])
                compare(row['direct_minus_surrogate_dex'], d[j]-p[j], ve[j])
                np.testing.assert_allclose(row['effective_line_samples'], ess[j], rtol=1e-11)
                np.testing.assert_allclose(row['missing_predicted_positive_line_fractions'], missing[j], rtol=1e-11)
            for row in report['shape_checks']:
                if row['geometry'] != label or row['model'] != name:
                    continue
                a = np.zeros(len(regions))
                if row['contrast'] == 'outer_endpoint_14_15_minus_8_9':
                    a[8], a[14] = -1, 1
                    family = 1
                elif row['contrast'] == 'outer_OLS_slope_dex_per_kpc':
                    x = np.arange(8.5, 15.)
                    a[8:15] = (x-x.mean())/np.sum((x-x.mean())**2)
                    family = 1
                else:
                    found = re.fullmatch(r'annulus_(\d+)_(\d+)_minus_(\d+)_(\d+)', row['contrast'])
                    a[int(found[1])], a[int(found[3])] = 1, -1
                    family = 14
                v, e = float(variance(sd@a)), float(variance((sd-sp)@a))
                compare(row['direct'], float(d@a), v)
                compare(row['direct_minus_surrogate'], float((d-p)@a), e)
                z = float(norm.isf(.05/(2*family)))
                band = row['within_curve_familywise_95']
                np.testing.assert_allclose(band['interval_95'], [d@a-z*np.sqrt(v), d@a+z*np.sqrt(v)], atol=3e-12)
        for comparison, (a, b) in PAIRS.items():
            good = qualified[:, BRANCH[a]] & qualified[:, BRANCH[b]]
            da, pa, sa, spa, _, _ = measure(a, good, geom)
            db, pb, sb, spb, _, _ = measure(b, good, geom)
            d, p = da-db, pa-pb
            vd, ve = variance(sa-sb), variance(sa-sb-spa+spb)
            for j, region in enumerate(regions):
                row = pair_rows[label, comparison, region]
                compare(row['direct_response_dex'], d[j], vd[j])
                compare(row['response_error_dex'], d[j]-p[j], ve[j])
    submissions = json.loads((ROOT/'submissions.json').read_text())
    job_ids = [r['job_id'] for r in submissions['jobs']]
    process = subprocess.run(['qstat', '-xf', '-F', 'json', *job_ids], capture_output=True, text=True, check=True)
    jobs = json.loads(process.stdout)['Jobs']
    assert all(j['job_state']=='F' and j.get('Exit_status')==0 for j in jobs.values())
    total_su = sum(float(re.search(r'Service Units:\s*([\d.]+)', p.read_text())[1]) for p in (ROOT/'logs').glob('*.OU'))
    summary = dict(passed=True, reviewed_particle_count=n, physical_states=5*n,
        qualified_states=int(qualified.sum()), failed_states=failures, timed_out_states=timeouts, reused_states=reused,
        interval_sets_independently_reconstructed=checks, maximum_point_difference=maximum_point,
        maximum_standard_error_difference=maximum_se, reviewed_source_sha256=sha(ROOT/'review.json'),
        manifest_sha256=sha(ROOT/'manifest.json'), output_hashes=output_hashes,
        all_14_jobs_exit_zero=True, service_units=total_su, MSU=total_su/1e6,
        physical_solves_in_audit=0, interpretation='Sampling uncertainty conditional on successful states; no physical-model or unresolved-light bound.')
    (ROOT/'independent_audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    (ROOT/'completed_jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
    (ROOT/'STATUS.md').write_text('All 14 PBS jobs completed with exit status 0.\n'
        'Independent cached-statistics audit passed; see independent_audit.json and REVIEW.md.\n'
        'One Uniform/off outer annulus fails the pointwise 0.05 dex interval screen; no unconditional acceptance.\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'output_hashes'}, indent=2))


if __name__ == '__main__':
    main()
