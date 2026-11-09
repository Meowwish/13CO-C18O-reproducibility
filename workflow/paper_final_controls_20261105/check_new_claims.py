"""Check every newly quoted result against completed cached reference outputs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ISO=ROOT.parent


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    text=(ISO/'Paper/mnras_template.tex').read_text()
    study=ISO/'x8_claim_checks_20261105_v3'
    d=json.loads((study/'review.json').read_text());audit=json.loads((study/'independent_audit.json').read_text())
    assert audit['passed'] and audit['review_sha256']==sha(study/'review.json')
    for observable in ('line_ratio','molecule_number_ratio','emission_per_molecule_ratio'):
        for reference in ('default','extended'):
            rows=[r for r in d['radial_records'] if (r['observable'],r['reference'])==(observable,reference)]
            assert len(rows)==42 and all(r['interval_within_0p05_dex'] for r in rows)
            if reference=='extended':assert f'{max(abs(r["direct_minus_surrogate"]["point"]) for r in rows):.4f}' in text
    assert len(d['control_sfr_slopes'])==32 and all(r['direct']['interval_95'][1]<0 for r in d['control_sfr_slopes'])
    assert f'{max(abs(r["direct_minus_surrogate"]["point"]) for r in d["control_sfr_slopes"]):.5f}' in text
    cells=[]
    for model in ('actual_on','actual_off','uniform_on','uniform_off'):
        for window in (1,10):
            r=next(r['direct'] for r in d['control_sfr_slopes'] if (r['reference'],r['model'],r['window_Myr'],r['radial'])==('extended',model,window,'fixed_effect'))
            cell=f'${r["point"]:+.4f}\\;[{r["interval_95"][0]:+.4f},{r["interval_95"][1]:+.4f}]$'
            assert cell in text,cell;cells.append(cell)
    for contrast in ('annulus_2_3_minus_1_2','annulus_3_4_minus_2_3'):
        r=next(r for r in d['inner_adjacent_contrasts'] if (r['reference'],r['model'],r['observable'],r['contrast'])==('extended','uniform_off','line_ratio',contrast))
        assert f'{r["direct"]["point"]:+.4f}' in text or f'{r["direct"]["point"]:.4f}' in text
        for value in r['within_inner_curve_familywise_95']:assert f'{value:+.4f}' in text,f'{value:+.4f}'
    rows=[r for r in d['radial_records'] if r['reference']=='extended']
    assert f'{100*max(max(r["missing_predicted_line_fraction"]) for r in rows):.2f}' in text
    assert f'{100*max(max(r["missing_predicted_molecular_fraction"]) for r in rows):.2f}' in text
    exposure=json.loads((study/'shielding_exposure.json').read_text())
    for line in ('13CO','C18O'):
        row=next(r for r in exposure['records'] if (r['branch'],r['region'],r['line'])==('actual_on','full_field',line))
        assert f'{100*row["out_of_carbon_table_light_fraction"]:.2f}' in text
        if line=='13CO':assert f'{100*row["molecular_lookup_shift_exceeds_0p05_dex_fraction"]:.2f}' in text
    for a,b in json.loads((ROOT/'figure_checks.json').read_text())['direct600_depth_spread_dex'].values():
        assert f'{a:.3f}--{b:.3f}' in text
    # Within-radius slope differences support exchange steepening; pattern
    # intervals containing zero are not turned into equality claims.
    for r in d['paired_sfr_slope_differences']:
        if r['reference']=='extended' and r['radial']=='fixed_effect':
            ci=r['direct_slope_difference']['interval_95']
            if r['comparison'] in ('exchange','uniform_exchange'):assert ci[1]<0
            else:assert ci[0]<=0<=ci[1]
    result=dict(passed=True,paper_sha256=sha(ISO/'Paper/mnras_template.tex'),
        review_sha256=audit['review_sha256'],sfr_table_cells=len(cells),
        all_252_inner_ratio_intervals_checked=True,all_32_sfr_intervals_negative=True,
        covariance_corrected_dip_checked=True,physical_timescale_work=False,
        coupled_shielding_results_incorporated=False)
    (ROOT/'new_claim_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print('New numerical statements checked; no pending physics result incorporated.')


if __name__=='__main__':main()
