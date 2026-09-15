"""Render every panel as a standalone image and as part of its composite figure.

Outputs, under figures/:
  individual/  one file per panel (PDF vector + PNG at 600 dpi + SVG)
  composite/   the assembled multi-panel figures used in the manuscript
  extra/       additional analyses drawn from the same data, not in the paper
  clustered-overview.*  a single page collecting the summary panels
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figstyle import use_style
use_style()
import panels as P

F=str(paths.FIGURES)
for d in ('individual','composite','extra','data'):
    os.makedirs(f'{F}/{d}', exist_ok=True)

DPI=600
def save(fig, path):
    for ext in ('pdf','png','svg'):
        fig.savefig(f'{path}.{ext}', dpi=DPI)
    plt.close(fig)
    print('  ->', os.path.basename(path))

def label(ax, letter):
    t=ax.get_title(); ax.set_title(''); ax.set_title('(%s) %s'%(letter,t), loc='left')

def one(fn, name, folder='individual', size=(3.4,2.6), **kw):
    fig, ax = plt.subplots(figsize=size)
    fn(ax, fig=fig, **kw)
    save(fig, f'{F}/{folder}/{name}')

# --------------------------------------------------------------- individual
B = P.benzene_field()
print('Figure 1 panels')
for fn, nm, sz in [(P.p1a,'fig1a_total-density',(3.2,2.6)),
                   (P.p1b,'fig1b_pi-density',(3.2,2.6)),
                   (P.p1c,'fig1c_accessibility-mask',(3.2,2.6)),
                   (P.p1d,'fig1d_accessible-field',(3.2,2.6)),
                   (P.p1e,'fig1e_persistence-diagram',(3.4,2.8)),
                   (P.p1f,'fig1f_barcode',(3.4,2.8))]:
    one(fn, nm, size=sz, B=B)

print('Figure 2 panels')
one(P.p2a,'fig2a_ring-count',            size=(4.0,3.2))
one(P.p2b,'fig2b_shielding-integral',    size=(3.6,2.8))
one(P.p2c,'fig2c_shielding-occlusion',   size=(3.6,2.8))

print('Figure 3 panels')
one(P.p3a,'fig3a_grid-convergence',      size=(3.4,2.6))
one(P.p3b,'fig3b_rotation-spread',       size=(3.4,2.6))
one(P.p3c,'fig3c_functional-sensitivity',size=(3.4,2.6))

print('Figure 4 panels')
for t,nm in [('eint_Na+','fig4a_regression-Na'),('eint_Cl-','fig4b_regression-Cl'),
             ('eint_benzene','fig4c_regression-benzene'),('eint_CH4','fig4d_regression-CH4')]:
    fig,ax=plt.subplots(figsize=(3.2,2.8)); P.p4(ax,t,fig=fig,legend=True)
    save(fig, f'{F}/individual/{nm}')

# --------------------------------------------------------------- composites
print('Composite figures')
fig=plt.figure(figsize=(7.4,5.2)); gs=fig.add_gridspec(2,3,hspace=.48,wspace=.62)
for k,(fn,lab) in enumerate([(P.p1a,'a'),(P.p1b,'b'),(P.p1c,'c'),
                             (P.p1d,'d'),(P.p1e,'e'),(P.p1f,'f')]):
    ax=fig.add_subplot(gs[k])
    kw=dict(B=B) if fn in (P.p1e,P.p1f) else dict(B=B, cbar_label=False)
    fn(ax,fig=fig,**kw); label(ax,lab)
save(fig, f'{F}/composite/figure1_construction')

fig=plt.figure(figsize=(7.2,3.5)); gs=fig.add_gridspec(1,3,wspace=.42,width_ratios=[1.3,1,1],bottom=.34,top=.90)
H2=None
for k,(fn,lab) in enumerate([(P.p2a,'a'),(P.p2b,'b'),(P.p2c,'c')]):
    ax=fig.add_subplot(gs[k])
    fn(ax,fig=fig,legend=False) if fn is P.p2a else fn(ax,fig=fig)
    if fn is P.p2a: H2=ax.get_legend_handles_labels()
    label(ax,lab)
fig.legend(*H2, ncol=5, loc='lower center', bbox_to_anchor=(0.5,-0.015),
           frameon=False, fontsize=6.5, handletextpad=.4, columnspacing=1.2)
save(fig, f'{F}/composite/figure2_topology-and-shielding')

fig=plt.figure(figsize=(7.2,3.1)); gs=fig.add_gridspec(1,3,wspace=.52,bottom=.34)
for k,(fn,lab) in enumerate([(P.p3a,'a'),(P.p3b,'b'),(P.p3c,'c')]):
    ax=fig.add_subplot(gs[k]); fn(ax,fig=fig); label(ax,lab)
save(fig, f'{F}/composite/figure3_stability')

fig=plt.figure(figsize=(7.2,2.6)); gs=fig.add_gridspec(1,4,wspace=.46,top=.80)
H=None
for k,(t,lab) in enumerate([('eint_Na+','a'),('eint_Cl-','b'),
                            ('eint_benzene','c'),('eint_CH4','d')]):
    ax=fig.add_subplot(gs[k]); h=P.p4(ax,t,fig=fig,legend=False); label(ax,lab)
    if k: ax.set_ylabel('')
    H=h
fig.legend(H, ['baseline', r'baseline $+\,\pi$-IDT'], loc='upper center',
           ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5,1.02))
save(fig, f'{F}/composite/figure4_regression')

# --------------------------------------------------------------- extras
print('Extra panels')
EX=[(P.px_threshold,'extra_threshold-absolute',(3.6,2.7)),
    (P.px_threshold_rel,'extra_threshold-relative',(3.6,2.7)),
    (P.px_mask,'extra_mask-sweep',(4.4,3.0)),
    (P.px_univariate,'extra_univariate-heatmap',(4.0,4.2)),
    (P.px_pca,'extra_pca-descriptor-space',(4.6,3.6)),
    (P.px_dendrogram,'extra_dendrogram',(4.6,5.6)),
    (P.px_cost,'extra_cost-per-molecule',(3.6,2.7)),
    (P.px_poav,'extra_poav-vs-exact',(4.2,5.0)),
    (P.px_acene,'extra_acene-series',(3.4,2.6)),
    (P.px_barcodes,'extra_barcode-gallery',(4.2,3.4)),
    (P.px_diagrams,'extra_persistence-gallery',(3.8,3.2)),
    (P.px_occl_vs_size,'extra_occlusion-vs-size',(3.8,3.0))]
for fn,nm,sz in EX:
    one(fn, nm, folder='extra', size=sz)

# --------------------------------------------------- clustered overview page
print('Clustered overview')
fig=plt.figure(figsize=(13.5,10.0))
gs=fig.add_gridspec(3,4, hspace=.72, wspace=.40, height_ratios=[1,1,1])
spec=[(P.p1b,'a',dict(B=B,cbar=False)),(P.p1d,'b',dict(B=B,cbar=False)),
      (P.p1e,'c',dict(B=B)),(P.p1f,'d',dict(B=B)),
      (P.p2a,'e',{}),(P.p2b,'f',{}),(P.p2c,'g',{}),(P.px_threshold,'h',{}),
      (P.p3a,'i',{}),(P.p3b,'j',{}),
      (P.px_pca,'k',dict(annotate=('hexafluorobenzene','corannulene')))]
for k,(fn,lab,kw) in enumerate(spec):
    ax=fig.add_subplot(gs[k]); fn(ax,fig=fig,**kw); label(ax,lab)
ax=fig.add_subplot(gs[11]); P.p4(ax,'eint_benzene',fig=fig,legend=True); label(ax,'l')
fig.suptitle(r'$\pi$-Interaction Density Topology: summary of results',
             fontsize=12, y=0.985)
save(fig, f'{F}/clustered-overview')
print('done')
