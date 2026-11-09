"""Reader-facing figures from audited cached summaries; no model inference."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
ISO=ROOT.parent
FIG=ISO/'Paper/figs'
CLAIMS=ISO/'x8_claim_checks_20261105_v3'
LABELS={'actual_on':'Actual isotopes; exchange on','actual_off':'Actual isotopes; exchange off',
    'uniform_on':'Uniform isotopes; exchange on','uniform_off':'Uniform isotopes; exchange off'}
COLORS=('#1769aa','#bf6000','#37803b','#8e4d94')


def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'));fig.savefig(ROOT/(name+'.png'),dpi=130);plt.close(fig)


def main():
    audit=json.loads((CLAIMS/'independent_audit.json').read_text());assert audit['passed']
    r=json.loads((CLAIMS/'review.json').read_text())
    assert hashlib.sha256((CLAIMS/'review.json').read_bytes()).hexdigest()==audit['review_sha256']
    radial=json.loads((ISO/'isotope_fractionation_controls/x8_snapshot575/summary.json').read_text())['radial']
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),layout='constrained')
    for ax,mode,title in zip(axes[:2],('uncut','common_D1000'),('All samples','Common dual detections')):
        for (model,label),color in zip(LABELS.items(),COLORS):
            rows=sorted([x for x in radial if (x['branch'],x['mode'])==(model,mode)],key=lambda x:x['r_lo'])
            ax.plot([x['r_lo']+.5 for x in rows],[x['ratio'] for x in rows],'o-',ms=3,color=color,label=label)
        ax.axvspan(0,1,color='.9');ax.axvline(8,color='.4',ls=':',lw=.8)
        if mode!='uncut':ax.axvspan(8,15,color='.9')
        ax.set(xlabel='Radius [kpc]',ylabel=r'$\sum W_{13}/\sum W_{18}$',title=title,xlim=(0,15));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    subset=[next(x for x in r['inner_adjacent_contrasts'] if (x['reference'],x['model'],x['observable'],x['contrast'])==
        ('extended','uniform_off','line_ratio',f'annulus_{j+1}_{j+2}_minus_{j}_{j+1}')) for j in range(1,7)]
    x=np.arange(6);point=np.array([z['direct']['point'] for z in subset]);ci=np.array([z['within_inner_curve_familywise_95'] for z in subset])
    axes[2].errorbar(x-.05,point,yerr=[point-ci[:,0],ci[:,1]-point],fmt='o',capsize=3,label='Direct; simultaneous within curve')
    axes[2].plot(x+.05,[z['matched_surrogate'] for z in subset],'s',label='Surrogate; matched sample',color='#bf6000')
    axes[2].axhline(0,color='.5',lw=.8);axes[2].grid(axis='y',alpha=.2)
    axes[2].set(xticks=x,xticklabels=[f'{j+1}–{j+2}\n−{j}–{j+1}' for j in range(1,7)],
        xlabel='Annulus contrast [kpc]',ylabel=r'$\Delta\log_{10}R^{13}_{18}$ [dex]',title='Uniform isotopes; exchange off')
    axes[2].legend(fontsize=6.5)
    save(fig,'snapshot575_exchange_shape_checked')
    # Main SFR figure: source-mixture curve and direct four-control radius test.
    source=ISO/'highres_snapshot575_temporal/output/matched_sfr/SFR_statistics.csv'
    cached=list(csv.DictReader(source.open()))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for branch,color in [('Full',COLORS[0]),('SLUG',COLORS[1])]:
        rows=sorted([x for x in cached if (x['window'],x['mask_mode'],x['branch'])==('10Myr','common_unselected',branch)
            and -3<=float(x['log_sfr_lo'])<-.5 and int(x['selected'])>=10],key=lambda x:float(x['log_sfr_lo']))
        assert len(rows)==5
        axes[0].errorbar([(float(x['log_sfr_lo'])+float(x['log_sfr_hi']))/2 for x in rows],
            [float(x['ratio_of_sums']) for x in rows],yerr=[float(x['spatial_jackknife_error']) for x in rows],
            fmt='o-',capsize=3,color=color,label='Full enrichment' if branch=='Full' else 'SLUG-only')
    axes[0].set(xlabel=r'$\log_{10}[\Sigma_{\rm SFR}/({\rm M}_\odot\,{\rm yr}^{-1}\,{\rm kpc}^{-2})]$',
        ylabel=r'$\sum W_{13}/\sum W_{18}$',title='10 Myr: common uncut footprint');axes[0].legend(fontsize=8)
    for window,offset,color in [(1,-.06,COLORS[0]),(10,.06,COLORS[1])]:
        rows=[next(z for z in r['control_sfr_slopes'] if (z['reference'],z['model'],z['window_Myr'],z['radial'])==
            ('extended',model,window,'fixed_effect')) for model in LABELS]
        y=np.array([z['direct']['point'] for z in rows]);ci=np.array([z['direct']['interval_95'] for z in rows])
        axes[1].errorbar(np.arange(4)+offset,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt='o',capsize=3,color=color,label=f'{window} Myr')
    axes[1].axhline(0,color='.5',lw=.8);axes[1].set(xticks=np.arange(4),
        xticklabels=['Actual\non','Actual\noff','Uniform\non','Uniform\noff'],ylabel='Radius-controlled SFR slope [dex/dex]',
        title='Direct four-control reference',xlabel='Isotope pattern / carbon exchange');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'snapshot575_sfr_controls_checked')
    # Selected-statistic evidence from direct original-resolution maps only.
    source=ISO/'chemTable_temporal_550_575/final_test_600/selection/radial_statistics.csv'
    direct=[x for x in csv.DictReader(source.open()) if x['selection']=='dual' and x['mask_mode']=='own_selection'
        and 1<=float(x['radius_lo'])<8]
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,layout='constrained')
    keys=('median_ratio_direct','ratio_of_medians_direct','ratio_of_sums_direct','direct_fraction')
    titles=('Median of local ratios','Ratio of median intensities','Ratio of summed intensities','Selected fraction')
    for depth,color in zip((10,100,1000,10000,100000,1000000),plt.cm.viridis(np.linspace(.05,.95,6))):
        rows=sorted([z for z in direct if int(z['depth'])==depth],key=lambda z:float(z['radius_lo']))
        for ax,key in zip(axes.flat,keys):
            y=[float(z[key]) if int(z['direct_count'])>=10 or key=='direct_fraction' else np.nan for z in rows]
            ax.plot([float(z['radius_lo'])+.5 for z in rows],y,'o-',ms=3,color=color,label=f'$D=10^{{{int(np.log10(depth))}}}$')
    ranges={}
    for ax,key,title in zip(axes.flat,keys,titles):
        ax.set(title=title,xlim=(1,8),ylabel='Fraction' if key=='direct_fraction' else r'$R^{13}_{18}$');ax.grid(alpha=.2)
        if key!='ratio_of_sums_direct':ax.set_yscale('log')
        if key!='direct_fraction':
            spreads=[]
            for annulus in range(1,8):
                values=[float(z[key]) for z in direct if int(z['direct_count'])>=10 and float(z['radius_lo'])==annulus and float(z[key])>0]
                spreads.append(float(np.log10(max(values)/min(values))))
            ranges[key]=[min(spreads),max(spreads)]
    axes[0,0].legend(fontsize=7,ncol=2)
    for ax in axes[-1]:ax.set_xlabel('Radius [kpc]')
    fig.suptitle('Original-resolution snapshot 600: direct reference maps; noiseless dual-line cuts',fontsize=11)
    save(fig,'snapshot600_direct_selection_statistics')
    # Matched reference profiles make each layer's statistical evidence visible.
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,layout='constrained')
    cfg=json.loads((ISO/'highres_snapshot575_temporal/config.json').read_text())['projection']['line_frequencies_Hz']
    offset=3*np.log10(cfg['C18O']/cfg['13CO'])
    for col,(model,label) in enumerate([('production_full','Full enrichment'),('uniform_off','Uniform isotopes; exchange off')]):
        for row,observable,ylabel in [(0,'molecule_number_ratio','Molecule-number ratio'),(1,'emission_per_molecule_ratio','Relative emission per molecule')]:
            subset=[next(z for z in r['radial_records'] if (z['reference'],z['model'],z['observable'],z['region'])==
                ('extended',model,observable,f'annulus_{j}_{j+1}')) for j in range(1,8)]
            add=offset if row else 0
            point=np.array([z['direct_log_ratio']['point'] for z in subset])+add
            ci=np.array([z['direct_log_ratio']['interval_95'] for z in subset])+add
            y=10**point;lower,upper=10**ci[:,0],10**ci[:,1]
            axes[row,col].errorbar(np.arange(1.5,8),y,yerr=[y-lower,upper-y],fmt='o-',capsize=3,label='Direct reference')
            axes[row,col].plot(np.arange(1.5,8),[10**(z['matched_surrogate_log_ratio']+add) for z in subset],
                's--',color=COLORS[1],label='Surrogate; same weighted sample')
            axes[row,col].set(ylabel=ylabel,xlim=(1,8));axes[row,col].grid(alpha=.2)
            if row==0:axes[row,col].set_title(label)
            if row==1:axes[row,col].set_xlabel('Radius [kpc]')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('x8 snapshot 575: conditional 95% sampling intervals',fontsize=11)
    save(fig,'snapshot575_molecular_reference_checked')
    (ROOT/'figure_checks.json').write_text(json.dumps(dict(passed=True,claim_review_sha256=audit['review_sha256'],
        direct600_depth_spread_dex=ranges,physical_timescale_work=False,figures=['snapshot575_exchange_shape_checked',
        'snapshot575_sfr_controls_checked','snapshot600_direct_selection_statistics','snapshot575_molecular_reference_checked']),indent=2)+'\n')
    print('Saved four checked figures; no new map calculation.',flush=True)


if __name__=='__main__':main()
