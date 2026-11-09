"""Summarize every predeclared epoch without treating time samples as independent."""
import json
import os
import numpy as np
import extension as c


def extrema(values):
    values=np.asarray([v for v in values if v is not None],float)
    values=values[np.isfinite(values)]
    return dict(min=float(values.min()),median=float(np.median(values)),max=float(values.max())) if len(values) else None


def main():
    assert int(os.environ.get('PBS_NCPUS','0'))>0
    sha=c.verify(); summaries=[]
    for snap in range(551,576):
        row=json.loads((c.ROOT/f'time/{snap}/summary.json').read_text())
        assert row['manifest_sha256']==sha and row['snapshot']==snap
        summaries.append(row)
    time=np.array([r['time_Myr'] for r in summaries]); np.testing.assert_allclose(np.diff(time),1,atol=1e-7)
    source=np.array([r['source_multiplier'] for r in summaries])
    exchange=np.array([r['exchange_multiplier'] for r in summaries])
    radial=np.array([r['R13_18_brightness'][2:] for r in summaries])
    slopes=[]
    for branch in ('Full','SLUG'):
        for window in (1,10):
            for method in ('whole','fixed_effect'):
                values=[]; incomplete=[]
                for row in summaries:
                    value=next(x['slope'] for x in row['sfr_slopes'] if x['branch']==branch and x['window_Myr']==window and x['method']==method)
                    if row['sfr_catalogue_audit'][(0 if window==1 else 1)][2]>0:
                        incomplete.append(row['snapshot'])
                    else: values.append(value)
                slopes.append(dict(branch=branch,window_Myr=window,method=method,complete_window_extrema=extrema(values),
                                   complete_window_negative=sum(v is not None and v<0 for v in values),
                                   excluded_unknown_formation_mass_epochs=incomplete))
    result=dict(manifest_sha256=sha,epochs=25,time_span_Myr=[time[0],time[-1]],
        source_full_field_extrema=extrema(source[:,0]),source_inner_extrema=extrema(source[:,1]),
        source_below_one_count=np.sum(source<1,axis=0),
        exchange_full_field_extrema=extrema(exchange[:,0]),exchange_inner_extrema=extrema(exchange[:,1]),
        exchange_above_one_count=np.sum(exchange>1,axis=0),
        radial_strict_increase_count=np.sum(np.all(np.diff(radial,axis=1)>0,axis=1),axis=0),
        radial_strict_decrease_count=np.sum(np.all(np.diff(radial,axis=1)<0,axis=1),axis=0),
        minimum_99pct_coverage_models_and_regions=np.sum(np.array([r['coverage_99pct'] for r in summaries]),axis=0),
        oxygen_scale_extrema=extrema([r['oxygen_scale'] for r in summaries]),sfr_slopes=slopes,
        replay575=summaries[-1].get('replay575_production_Full_particle_total_delta_dex'),
        scope='All consecutive outputs retained; temporally correlated, no independent-epoch standard error or evolutionary generalization. Frozen-surrogate predictions only.')
    c.save(c.ROOT/'temporal_review.json',result)
    c.save(c.ROOT/'temporal_series.json',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for r,label in ((0,'Full field'),(1,'1–8 kpc')):
        ax[0].plot(time-time[0],source[:,r],label=label)
        ax[1].plot(time-time[0],exchange[:,r],label=label)
    for a in ax[:2]:
        a.axhline(1,color='0.5',lw=.7);a.legend();a.set_xlabel('Time since snapshot 551 [Myr]')
    ax[0].set_ylabel('SLUG / Full R13/18');ax[1].set_ylabel('Exchange on / off R13/18')
    im=ax[2].imshow(radial[:,:,0].T,origin='lower',aspect='auto',extent=[-.5,24.5,1,8])
    ax[2].set(xlabel='Time since snapshot 551 [Myr]',ylabel='Radius [kpc]',title='Full radial R13/18')
    fig.colorbar(im,ax=ax[2]);fig.savefig(c.ROOT/'temporal_diagnostics.png',dpi=180);plt.close(fig)
    (c.ROOT/'TEMPORAL_REVIEW.md').write_text('# Frozen-surrogate temporal sequence\n\n'
        'All 25 outputs (1 Myr cadence, 24 Myr span) are retained in temporal_series.json. '
        'temporal_review.json reports effect ranges, signs, radial monotonicity, coverage and SFR slopes. '
        'Epochs share a simulation and are correlated. No independent-time significance is assigned. '
        '575 replay and coverage are diagnostics, not direct-solver accuracy at new epochs.\n')
    print('All 25 epoch summaries reviewed.')


if __name__=='__main__':main()
