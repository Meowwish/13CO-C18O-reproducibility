"""Redraw two tiny cached radial summaries; no map or particle calculations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
ISO=ROOT.parent
FIG=ISO/'Paper/figs'


def main():
    with np.load(ISO/'highres_snapshot575_temporal/output/source_comparison/radial_comparison.npz') as f:
        x=f['radius_kpc'];full=f['full'];slug=f['slug_only'];difference=f['difference_dex']
    errors=[]
    for folder in ('output','slug_only/output'):
        with np.load(ISO/f'highres_snapshot575_temporal/{folder}/snapshot575_radial_profiles.npz') as f:
            errors.append(f['R13_18_jackknife_error'])
    fig,axes=plt.subplots(2,1,figsize=(5.8,6),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    for y,e,label,color in zip((full,slug),errors,('Full enrichment','SLUG-only'),('#1769aa','#bf6000')):
        axes[0].errorbar(x,y,yerr=e,fmt='o-',capsize=2,label=label,color=color,ms=3)
    axes[0].set(ylabel=r'$\sum W_{13}/\sum W_{18}$');axes[0].legend(fontsize=9)
    axes[1].plot(x,difference,'o-',color='#59378b',ms=3)
    axes[1].axhline(0,color='.5',lw=.7)
    axes[1].set(xlabel='Radius [kpc]',ylabel=r'$\log_{10}(R_{\rm SLUG}/R_{\rm Full})$')
    for ax in axes:
        ax.axvspan(0,1,color='.9');ax.axvline(8,color='.4',ls='--',lw=.8);ax.set_xlim(0,15);ax.grid(alpha=.2)
    fig.savefig(FIG/'snapshot575_full_vs_slug_R13_18_outer_checked.pdf');plt.close(fig)
    d=json.loads((ISO/'isotope_fractionation_controls/x8_snapshot575/summary.json').read_text())['radial']
    labels={'actual_on':'Actual isotopes; exchange on','actual_off':'Actual isotopes; exchange off',
        'uniform_on':'Uniform isotopes; exchange on','uniform_off':'Uniform isotopes; exchange off'}
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,layout='constrained')
    for ax,mode,title in zip(axes,('uncut','own_D1000','common_D1000'),
            ('All samples','Each control: dual detections','Common dual detections')):
        for (name,label),color in zip(labels.items(),('#1769aa','#bf6000','#37803b','#8e4d94')):
            rows=sorted((r for r in d if r['branch']==name and r['mode']==mode),key=lambda r:r['r_lo'])
            ax.plot([r['r_lo']+.5 for r in rows],[r['ratio'] for r in rows],marker='o',ms=3,color=color,label=label,
                ls='--' if name.endswith('off') else '-')
        ax.axvspan(0,1,color='.9');ax.axvline(8,color='.4',ls=':',lw=.8)
        if mode!='uncut':ax.axvspan(8,15,color='.9')
        ax.set(xlim=(0,15),title=title,xlabel='Radius [kpc]');ax.grid(alpha=.2)
    axes[0].set_ylabel(r'$\sum W_{13}/\sum W_{18}$');axes[0].legend(fontsize=7.5)
    fig.savefig(FIG/'snapshot575_x8_exchange_radial_outer_checked.pdf');plt.close(fig)
    print('Saved source and four-control figures from cached radial summaries.')


if __name__=='__main__':main()
