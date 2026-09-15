"""One function per figure panel. Every function takes an Axes and draws into it,
so the same code produces the standalone image and the composite figure."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, json, os
import numpy as np, pandas as pd
from figstyle import (PALETTE, GREY, INK, MUTED, DIMCOL, GROUPS,
                      GROUP_COLOR, GROUP_MARKER, regroup, despine)

R=str(paths.RESULTS); F=str(paths.FIGURES)
A   = json.load(open(f'{R}/setA.json'))
NR  = json.load(open(f'{R}/nring.json'))
SP  = json.load(open(f'{R}/shield_poav.json'))
RB  = json.load(open(f'{R}/robust.json'))
GR  = json.load(open(f'{R}/regression.json'))
MS  = json.load(open(f'{R}/mask_series.json'))
TS  = json.load(open(f'{R}/threshold_sens.json'))
TR  = json.load(open(f'{R}/threshold_rel.json'))
PV  = json.load(open(f'{R}/poav_v2.json'))
SER = ['benzene','toluene','p-xylene','mesitylene','hexamethylbenzene','hexaethylbenzene']
SERLAB = ['benzene','toluene','$p$-xylene','mesitylene','HMB','HEB']

# ----------------------------------------------------------------- fig 1 data
CACHE=f'{F}/data/benzene_field.npz'
def benzene_field(h=0.10):
    if os.path.exists(CACHE):
        z=np.load(CACHE, allow_pickle=True)
        return {k:z[k] for k in z.files}
    import pidt, opt
    s,x,rd = pidt.geometry_from_smiles('c1ccccc1')
    g = opt.prepare_geometry(s,x); x=g['xyz']
    mol,mf = pidt.run_dft(s,x)
    dm,_   = pidt.valence_dm(mf); dm_all = mf.make_rdm1()
    dmp,_,_= pidt.pi_density_matrix_symmetry(mol, dm)
    pia,_  = pidt.pi_atom_set(mol, dmp)
    G,shape,ax = pidt.make_grid(x, pad=3.5, h=h)
    rho_t = pidt.eval_density(mol, dm_all, G).reshape(shape)
    rho_p = pidt.eval_density(mol, dmp,    G).reshape(shape)
    S     = pidt.accessibility_mask(G,s,x,pia).reshape(shape)
    bars  = pidt.persistence((rho_p*S).ravel(), shape)
    d=dict(rho_t=rho_t, rho_p=rho_p, S=S, A=rho_p*S,
           ax0=ax[0], ax1=ax[1], ax2=ax[2],
           b0=bars[0], b1=bars[1], b2=bars[2], xyz=x, syms=np.array(s))
    np.savez_compressed(CACHE, **d)
    return d

def _slice(B, F_, plane, iz=None, iy=None):
    if plane=='xy': return F_[:,:,iz].T, [B['ax0'][0],B['ax0'][-1],B['ax1'][0],B['ax1'][-1]], '$x$ / Å','$y$ / Å'
    return F_[:,iy,:].T, [B['ax0'][0],B['ax0'][-1],B['ax2'][0],B['ax2'][-1]], '$x$ / Å','$z$ / Å'

def _field_panel(ax, B, F_, plane, iz=None, iy=None, cmap='viridis', vmax=None,
                 label=None, cbar=True, fig=None):
    Z,ext,xl,yl = _slice(B,F_,plane,iz,iy)
    im=ax.imshow(Z, origin='lower', extent=ext, cmap=cmap, aspect='auto',
                 vmax=vmax if vmax is not None else np.percentile(Z,99.7))
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(False)
    if cbar and fig is not None:
        cb=fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
        cb.ax.tick_params(labelsize=5.5, length=2); cb.outline.set_linewidth(0.4)
        cb.locator = __import__('matplotlib').ticker.MaxNLocator(4); cb.update_ticks()
        if label: cb.set_label(label, fontsize=5.5, labelpad=1)
    return im

def p1a(ax, fig=None, B=None, cbar=True, cbar_label=True):
    B=B or benzene_field(); iz=int(np.argmin(np.abs(B['ax2'])))
    _field_panel(ax,B,B['rho_t'],'xy',iz=iz,label='e bohr$^{-3}$' if cbar_label else None,fig=fig,cbar=cbar)
    ax.set_title(r'total density $\rho$ ($z=0$)')
def p1b(ax, fig=None, B=None, cbar=True, cbar_label=True):
    B=B or benzene_field(); iy=int(np.argmin(np.abs(B['ax1'])))
    _field_panel(ax,B,B['rho_p'],'xz',iy=iy,label='e bohr$^{-3}$' if cbar_label else None,fig=fig,cbar=cbar)
    ax.set_title(r'$\pi$ density, $xz$')
def p1c(ax, fig=None, B=None, cbar=True, cbar_label=True):
    B=B or benzene_field(); iz=int(np.argmin(np.abs(B['ax2'])))
    _field_panel(ax,B,B['S'],'xy',iz=iz,cmap='Greys_r',vmax=1,label='$S$' if cbar_label else None,fig=fig,cbar=cbar)
    ax.set_title(r'accessibility $S$ ($z=0$)')
def p1d(ax, fig=None, B=None, cbar=True, cbar_label=True):
    B=B or benzene_field(); iz=int(np.argmin(np.abs(B['ax2']-1.0)))
    _field_panel(ax,B,B['A'],'xy',iz=iz,label='e bohr$^{-3}$' if cbar_label else None,fig=fig,cbar=cbar)
    ax.set_title(r'$A_\pi=\rho_\pi S$ ($z=1$ Å)')
def p1e(ax, fig=None, B=None):
    B=B or benzene_field()
    for dim in (0,1,2):
        Bd=B[f'b{dim}'].reshape(-1,2)
        if not len(Bd): continue
        L=Bd[:,0]-Bd[:,1]; m=L>5e-3
        ax.scatter(Bd[m,0],Bd[m,1],s=26,facecolors='none',edgecolors=DIMCOL[dim],
                   linewidths=1.1,label=r'$H_%d$ (persistent)'%dim,zorder=3)
        ax.scatter(Bd[~m,0],Bd[~m,1],s=6,color=DIMCOL[dim],alpha=.3,zorder=2,
                   label='below threshold' if dim==0 else None)
        # symmetry-degenerate points superpose; mark their multiplicity
        P_=np.round(Bd[m],6)
        if len(P_):
            uq,ct=np.unique(P_,axis=0,return_counts=True)
            for (b,d_),c in zip(uq,ct):
                if c>1:
                    ax.annotate(r'$\times%d$'%c,(b,d_),textcoords='offset points',
                                xytext=(-16,4),fontsize=6,color=DIMCOL[dim])
    mx=float(B['b0'].reshape(-1,2)[:,0].max())
    ax.plot([0,mx],[0,mx],'-',color=MUTED,lw=.6,zorder=1)
    ax.set_xlabel(r'birth $\varepsilon_b$ / a.u.'); ax.set_ylabel(r'death $\varepsilon_d$ / a.u.')
    ax.set_title('persistence diagram')
    ax.legend(loc='upper left', borderaxespad=.3, fontsize=6,
              handletextpad=.4, labelspacing=.35)
    despine(ax)
def p1f(ax, fig=None, B=None):
    B=B or benzene_field(); y=0; ticks=[]
    for dim in (0,1):
        Bd=B[f'b{dim}'].reshape(-1,2); L=Bd[:,0]-Bd[:,1]
        for i in np.argsort(-L):
            if L[i]<1e-3: continue
            ax.plot([Bd[i,1],Bd[i,0]],[y,y],color=DIMCOL[dim],lw=2.0,
                    solid_capstyle='round'); y+=1
    ax.plot([],[],color=DIMCOL[0],lw=2,label='$H_0$')
    ax.plot([],[],color=DIMCOL[1],lw=2,label='$H_1$')
    ax.set_xlabel(r'$\varepsilon$ / a.u.'); ax.set_yticks([])
    ax.set_title(r'$\pi$-barcode')
    ax.set_ylim(-1.5, y+0.30*max(y,1))
    ax.legend(loc='upper right', ncol=2)
    despine(ax, keep=('bottom',))

# ----------------------------------------------------------------- fig 2
def _df():
    d=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))}
                    for r in A.values()]).set_index('name')
    d['n_pi_rings']=pd.Series(NR); d['grp']=d['group'].map(regroup)
    return d

def p2a(ax, fig=None, legend=True):
    d=_df()
    for g in GROUPS:
        sel=d[d.grp==g]
        if not len(sel): continue
        ax.scatter(sel['n_pi_rings'], sel['piIDT1_n'], marker=GROUP_MARKER[g],
                   s=30, facecolors='none' if g!='reference' else GROUP_COLOR[g],
                   edgecolors=GROUP_COLOR[g], linewidths=1.1, label=g, zorder=3)
    lim=[-0.4, d['n_pi_rings'].max()+0.6]
    ax.plot(lim,[2*l for l in lim],'--',color=MUTED,lw=.8,zorder=1,
            label=r'$\beta_1=2N_{\rm ring}$')
    for m,off in [('thiophene',(-22,9)),('biphenyl',(7,7)),('corannulene',(-50,-3))]:
        ax.annotate(m,(d.loc[m,'n_pi_rings'],d.loc[m,'piIDT1_n']),
                    textcoords='offset points',xytext=off,fontsize=6,color=INK)
    ax.set_xlim(*lim); ax.set_xlabel(r'number of $\pi$ rings $N_{\rm ring}$')
    ax.set_ylabel(r'$\pi$-IDT$_1$ loop count $\beta_1$')
    ax.set_title(r'$\beta_1$ counts rings')
    ax.set_ylim(-2, d['piIDT1_n'].max()+4)
    if legend:
        ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5,-0.32),
                  handletextpad=.4, columnspacing=.9, borderaxespad=0.,
                  fontsize=6, frameon=False)
    despine(ax)

def p2b(ax, fig=None):
    y=[SP[m]['I'] for m in SER]
    ax.bar(range(6), y, color=PALETTE[0], width=0.62, zorder=3)
    for i,v in enumerate(y):
        ax.text(i, v+0.02, '%.2f'%v, ha='center', fontsize=6, color=INK)
    ax.set_xticks(range(6)); ax.set_xticklabels(SERLAB, rotation=38, ha='right',
                       rotation_mode='anchor', fontsize=6.5)
    ax.set_ylabel(r'$I_\pi$ / e'); ax.set_ylim(0, max(y)*1.18)
    ax.set_title(r'accessible $\pi$ density')
    despine(ax)

def p2c(ax, fig=None):
    y=[SP[m]['Om'] for m in SER]
    ax.plot(range(6), y, 'o-', color=PALETTE[1], lw=1.4, ms=5, zorder=3)
    for i,v in enumerate(y):
        ha = 'left' if i==0 else ('right' if i==len(y)-1 else 'center')
        dx = 0.12 if i==0 else (-0.12 if i==len(y)-1 else 0.0)
        ax.text(i+dx, v+0.005, '%.3f'%v, ha=ha, fontsize=6, color=INK)
    ax.set_xticks(range(6)); ax.set_xticklabels(SERLAB, rotation=38, ha='right',
                       rotation_mode='anchor', fontsize=6.5)
    ax.set_ylabel(r'occlusion ratio $\Omega$')
    ax.set_xlim(-0.55, 5.45)
    ax.set_ylim(min(y)-0.02, max(y)+0.03)
    ax.set_title(r'buried fraction $\Omega$')
    despine(ax)

# ----------------------------------------------------------------- fig 3
def p3a(ax, fig=None):
    MK=['o','s','^','D','v','P','X']
    for i,(nm,rec) in enumerate(RB.items()):
        hs=sorted(rec['grid'], key=float)
        ax.plot([float(h) for h in hs],[rec['grid'][h]['piIDT1_totpers'] for h in hs],
                marker=MK[i%len(MK)], ms=4.5, lw=1.2, ls='-',
                markerfacecolor='none', markeredgewidth=1.1,
                color=PALETTE[i%len(PALETTE)], label=nm, zorder=3)
    ax.set_xlabel('grid spacing $h$ / Å'); ax.set_ylabel(r'$T_1$ / a.u.')
    ax.set_title('grid convergence')
    ax.set_ylim(0, None)
    ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5,-0.32),
              fontsize=6, frameon=False, columnspacing=1.0, handletextpad=.4)
    despine(ax)

def p3b(ax, fig=None):
    labs=[];vals=[]
    for nm,rec in RB.items():
        v=[rec['ref_poav']['piIDT1_totpers']]+[d['piIDT1_totpers'] for d in rec['rot']]
        vals.append(np.array(v)/v[0]); labs.append(nm)
    bp=ax.boxplot(vals, tick_labels=labs, widths=.55,
                  patch_artist=True, medianprops=dict(color=INK, lw=1.0))
    for b in bp['boxes']:
        b.set_facecolor(PALETTE[0]); b.set_alpha(.30); b.set_edgecolor(PALETTE[0])
    for w in bp['whiskers']+bp['caps']: w.set_color(MUTED)
    ax.axhline(1, color=MUTED, lw=.7, ls='--')
    ax.set_ylabel(r'$T_1/T_1^{\rm ref}$')
    ax.set_title('rotation sensitivity')
    ax.tick_params(axis='x', labelsize=6, rotation=25); despine(ax)

def p3c(ax, fig=None):
    RBx={k:v for k,v in RB.items() if 'xc' in v}
    fx=list(RBx[list(RBx)[0]]['xc'])
    w=0.8/max(len(RBx),1)
    for i,(nm,rec) in enumerate(RBx.items()):
        ref=rec['xc']['wb97x-d3bj']['piIDT1_totpers']
        ax.bar(np.arange(len(fx))+i*w-0.4+w/2,
               [rec['xc'][f]['piIDT1_totpers']/ref for f in fx],
               width=w*0.9, color=PALETTE[i%len(PALETTE)], label=nm, zorder=3)
    XCLAB={'wb97x-d3bj':r'$\omega$B97X-D3(BJ)','b3lyp':'B3LYP','pbe0':'PBE0','pbe':'PBE'}
    ax.set_xticks(range(len(fx)))
    ax.set_xticklabels([XCLAB.get(f,f) for f in fx], rotation=25, ha='right', fontsize=6)
    ax.axhline(1, color=MUTED, lw=.7, ls='--'); ax.set_ylim(0.90,1.10)
    ax.set_ylabel(r'$T_1$ (relative)'); ax.set_title('functional sensitivity')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5,-0.34),
              fontsize=6, frameon=False, columnspacing=1.0, handletextpad=.4)
    despine(ax)

# ----------------------------------------------------------------- fig 4
EN = '\u2013'   # matplotlib mathtext has no LaTeX '--' ligature
PROBE={'eint_Na+':   'Na$^+$ (cation%s$\\pi$)'%EN,
       'eint_Cl-':   'Cl$^-$ (anion%s$\\pi$)'%EN,
       'eint_benzene':'benzene ($\\pi$%s$\\pi$)'%EN,
       'eint_CH4':   'CH$_4$ (CH%s$\\pi$)'%EN}
def p4(ax, target, fig=None, legend=True):
    g=GR[target]; y=np.array(g['y'])
    handles=[]
    for s_,c,mk,lab in [('baseline',GREY,'o','baseline'),
                        ('baseline+piIDT',PALETTE[1],'s',r'+ $\pi$-IDT')]:
        p=np.array(g[s_]['pred'])
        h=ax.scatter(y,p,s=22,facecolors='none',edgecolors=c,marker=mk,
                     linewidths=1.0,label=lab,zorder=3)
        handles.append(h)
    lo=min(y.min(), np.array(g['baseline']['pred']).min())
    hi=max(y.max(), np.array(g['baseline']['pred']).max())
    pad=0.06*(hi-lo)
    ax.plot([lo,hi],[lo,hi],'-',color=MUTED,lw=.7,zorder=1)
    ax.set_xlim(y.min()-pad, y.max()+pad)
    ax.set_ylim(lo-pad, hi+pad+0.34*(hi-lo))
    fmtq=lambda v: ('%.2f'%v).replace('-', '\u2212')
    ax.text(0.04,0.97,'$Q^2$  %s  baseline\n$Q^2$  %s  $+\\,\\pi$-IDT'
            %(fmtq(g['baseline']['q2_mean']), fmtq(g['baseline+piIDT']['q2_mean'])),
            transform=ax.transAxes, va='top', ha='left', fontsize=6, color=INK,
            linespacing=1.5)
    ax.set_title(PROBE[target]); ax.set_xlabel(r'$E_{\rm int}$ / kcal mol$^{-1}$')
    ax.set_ylabel('cross-validated prediction')
    if legend:
        ax.legend(loc='lower right', fontsize=6)
    despine(ax)
    return handles

# ----------------------------------------------------------------- extras
def px_threshold(ax, fig=None):
    ths=sorted(float(k) for k in TS)
    ax.plot([t*1e3 for t in ths],[TS[str(t)] for t in ths],'o-',color=PALETTE[0],
            lw=1.4,ms=5,label='absolute threshold',zorder=3)
    ax.axvspan(1,6,color=PALETTE[0],alpha=.10,zorder=1)
    ax.axhline(39,color=MUTED,lw=.7,ls=':')
    ax.text(6.2,37.4,'39 molecules',fontsize=6,color=MUTED)
    ax.set_xlabel(r'$\varepsilon_{\rm noise}$ / $10^{-3}$ a.u.')
    ax.set_ylabel(r'molecules with $\beta_1=2N_{\rm ring}$')
    ax.set_title('threshold robustness'); ax.legend(); despine(ax)

def px_threshold_rel(ax, fig=None):
    fr=sorted(float(k) for k in TR)
    ax.plot([f*100 for f in fr],[TR['%.2f'%f] for f in fr],'s-',color=PALETTE[1],
            lw=1.4,ms=5,label=r'relative to $\max A_\pi$',zorder=3)
    ax.axhline(36,color=PALETTE[0],lw=1.0,ls='--',label='best absolute (36)')
    ax.set_xlabel(r'threshold / \% of $\max A_\pi$'.replace('\\%','%'))
    ax.set_ylabel(r'molecules with $\beta_1=2N_{\rm ring}$')
    ax.set_title('relative thresholds do worse')
    ax.legend(); despine(ax)

def px_mask(ax, fig=None):
    keys=sorted(MS[SER[0]])
    dcol={'0.5':PALETTE[0],'1.0':PALETTE[1],'1.5':PALETTE[2]}
    lsty={'0.9':'-','1.0':'--','1.1':':'}
    mrk ={'0.9':'o','1.0':'s','1.1':'^'}
    for k in keys:
        sc,dl=k.split('_')
        ax.plot(range(6),[MS[m][k]['I'] for m in SER],marker=mrk[sc],ms=3.5,lw=1.1,
                ls=lsty[sc],color=dcol[dl],alpha=.95,
                label=r'$\lambda$=%s, $\delta$=%s Å'%(sc,dl),zorder=3)
    import matplotlib.ticker as mt
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(mt.LogLocator(base=10, subs=(0.2,0.3,0.5,1.0,2.0,3.0)))
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v,_: ('%g'%v)))
    ax.yaxis.set_minor_formatter(mt.NullFormatter())
    ax.set_xticks(range(6)); ax.set_xticklabels(SERLAB, rotation=38, ha='right',
                       rotation_mode='anchor', fontsize=6.5)
    ax.set_ylabel(r'$I_\pi$ / e  (log scale)')
    ax.set_title(r'effect of the mask parameters')
    yl=ax.get_ylim(); ax.set_ylim(yl[0]/2.6, yl[1])
    ax.legend(ncol=3, fontsize=5.5, loc='lower left', framealpha=1.0,
              frameon=True, edgecolor='none', facecolor='white')
    despine(ax)

def px_univariate(ax, fig=None):
    UV=pd.read_csv(f'{R}/univariate.csv').set_index('feature')
    cols=['eint_Na+','eint_Cl-','eint_benzene','eint_CH4']
    M=UV[cols].astype(float)
    im=ax.imshow(M.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels([r'Na$^+$',r'Cl$^-$','benzene',r'CH$_4$'])
    NICE={'nheavy':r'$N_{\rm heavy}$','vdw_area':'vdW area','mw':r'$M_{\rm w}$',
          'esp_min':r'$V_{\min}$','esp_max':r'$V_{\max}$','esp_var':r'$\sigma^2_V$',
          'esp_axial_2.6':r'$V_{\rm ax}(2.6)$','Npi_grid':r'$N_\pi$',
          'Api_integral':r'$I_\pi$','occlusion_ratio':r'$\Omega$',
          'A_pi_max':r'$\max A_\pi$','piIDT0_n':r'$\beta_0$',
          'piIDT0_totpers':r'$T_0$','piIDT0_entropy':r'$E_0$','piIDT1_n':r'$\beta_1$',
          'piIDT1_totpers':r'$T_1$','piIDT1_maxpers':r'$\ell_1^{\max}$',
          'piIDT1_entropy':r'$E_1$'}
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([NICE.get(i,i) for i in M.index], fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v=M.values[i,j]
            ax.text(j,i,'%+.2f'%v,ha='center',va='center',fontsize=5.5,
                    color='white' if abs(v)>0.6 else INK)
    ax.grid(False); ax.set_title('Pearson $r$ with probe interaction energy')
    if fig is not None:
        cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03); cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.4)

IDTF=['piIDT0_n','piIDT0_totpers','piIDT0_entropy','piIDT1_n','piIDT1_totpers',
      'piIDT1_maxpers','piIDT1_entropy','piIDT2_n','Api_integral','occlusion_ratio',
      'A_pi_max','Npi_grid']
def _X():
    d=_df(); X=d[IDTF].astype(float).values
    X=(X-X.mean(0))/np.where(X.std(0)<1e-12,1,X.std(0))
    return d, X

def px_pca(ax, fig=None, annotate=None):
    from sklearn.decomposition import PCA
    d,X=_X(); P=PCA(n_components=2).fit(X); Y=P.transform(X)
    for g in GROUPS:
        m=(d.grp==g).values
        if not m.any(): continue
        ax.scatter(Y[m,0],Y[m,1],marker=GROUP_MARKER[g],s=32,
                   facecolors='none' if g!='reference' else GROUP_COLOR[g],
                   edgecolors=GROUP_COLOR[g],linewidths=1.1,label=g,zorder=3)
    LAB=[('coronene',(-44,-9)),('porphine',(-40,5)),('corannulene',(-52,6)),
         ('hexafluorobenzene',(-56,5)),('pentacene',(-42,-10))]
    if annotate is not None:
        LAB=[(n,o) for n,o in LAB if n in annotate]
    for nm,off in LAB:
        i=list(d.index).index(nm)
        ax.annotate(nm,(Y[i,0],Y[i,1]),textcoords='offset points',xytext=off,
                    fontsize=5.5,color=INK)
    xl=ax.get_xlim(); ax.set_xlim(xl[0]-0.10*(xl[1]-xl[0]), xl[1]+0.10*(xl[1]-xl[0]))
    ax.set_xlabel('PC1 (%.0f%% of variance)'%(100*P.explained_variance_ratio_[0]))
    ax.set_ylabel('PC2 (%.0f%%)'%(100*P.explained_variance_ratio_[1]))
    ax.set_title(r'$\pi$-IDT descriptor space')
    yl=ax.get_ylim(); ax.set_ylim(yl[0]-0.10*(yl[1]-yl[0]), yl[1]+0.46*(yl[1]-yl[0]))
    xl=ax.get_xlim(); ax.set_xlim(xl[0], xl[1]+0.14*(xl[1]-xl[0]))
    ax.legend(ncol=3, fontsize=5.5, loc='upper left', handletextpad=.35,
              columnspacing=.7); despine(ax)

def px_dendrogram(ax, fig=None):
    from scipy.cluster.hierarchy import linkage, dendrogram
    d,X=_X()
    Z=linkage(X, method='ward')
    dn=dendrogram(Z, labels=list(d.index), ax=ax, orientation='right',
                  color_threshold=0.55*Z[:,2].max(),
                  above_threshold_color=MUTED, leaf_font_size=5.5)
    ax.set_xlabel('Ward linkage distance'); ax.grid(False)
    ax.set_title(r'hierarchical clustering on $\pi$-IDT descriptors')
    despine(ax, keep=('bottom',))

def px_cost(ax, fig=None):
    d=_df()
    ax.scatter(d['nheavy'], d['t_total'], s=28, facecolors='none',
               edgecolors=PALETTE[0], linewidths=1.1, zorder=3, label='total')
    ax.scatter(d['nheavy'], d['t_grid'], s=20, marker='^', facecolors='none',
               edgecolors=PALETTE[2], linewidths=1.0, zorder=3,
               label='grid + persistence')
    ax.set_yscale('log'); ax.set_xlabel('heavy atoms')
    ax.set_ylabel('wall-clock / s (2 cores)')
    ax.set_title('cost per molecule'); ax.legend(); despine(ax)

def px_poav(ax, fig=None):
    nm=sorted(PV, key=lambda n: PV[n]['reldiff'])
    v=[100*PV[n]['reldiff'] for n in nm]
    cols=[PALETTE[1] if x>20 else PALETTE[0] for x in v]
    ax.barh(range(len(nm)), v, color=cols, height=.72, zorder=3)
    ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm, fontsize=5.5)
    ax.set_xlabel(r'$\|D_\pi^{\rm POAV}-D_\pi^{\rm exact}\|_F/\|D_\pi^{\rm exact}\|_F$  /  %')
    ax.set_title('local-axis vs exact partition')
    ax.axvline(np.median(v), color=MUTED, lw=.8, ls='--')
    ax.text(np.median(v)+1, 1, 'median %.1f%%'%np.median(v), fontsize=6, color=MUTED)
    despine(ax)

def px_acene(ax, fig=None):
    d=_df(); ac=['benzene','naphthalene','anthracene','tetracene','pentacene']
    n=[1,2,3,4,5]
    ax.plot(n, d.loc[ac,'piIDT1_n'], 'o-', color=PALETTE[0], lw=1.4, ms=5,
            label=r'$\beta_1$', zorder=3)
    ax.plot(n, d.loc[ac,'piIDT0_n'], 's--', color=PALETTE[1], lw=1.4, ms=5,
            label=r'$\beta_0$', zorder=3)
    ax2=None
    ax.set_xticks(n); ax.set_xticklabels(ac, rotation=30, ha='right')
    ax.set_ylabel('feature count')
    ax.set_title('the acene series: one to five fused rings')
    ax.legend(); despine(ax)

def px_barcodes(ax, fig=None):
    mols=['benzene','naphthalene','anthracene','pyrene','coronene','porphine','thiophene']
    y=0; yt=[]; yl=[]
    for m in mols:
        y0=y
        for dim in (0,1):
            B=np.array(A[m]['bars'][str(dim)]).reshape(-1,2)
            L=B[:,0]-B[:,1]
            for i in np.argsort(-L):
                if L[i]<5e-3: continue
                ax.plot([B[i,1],B[i,0]],[y,y],color=DIMCOL[dim],lw=1.5,
                        solid_capstyle='round'); y+=1
        yt.append((y0+y-1)/2); yl.append(m); y+=2
    ax.plot([],[],color=DIMCOL[0],lw=2,label='$H_0$')
    ax.plot([],[],color=DIMCOL[1],lw=2,label='$H_1$')
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6)
    ax.set_xlabel(r'$\varepsilon$ / a.u.'); ax.set_title(r'$\pi$-barcodes')
    ax.legend(loc='lower right'); despine(ax, keep=('bottom',))

def px_diagrams(ax, fig=None):
    mols=['benzene','naphthalene','coronene','porphine','thiophene','corannulene']
    for i,m in enumerate(mols):
        B=np.array(A[m]['bars']['1']).reshape(-1,2)
        L=B[:,0]-B[:,1]; keep=L>5e-3
        g=regroup(A[m]['group'])
        ax.scatter(B[keep,0],B[keep,1],s=26,marker=GROUP_MARKER[g],
                   facecolors='none',edgecolors=GROUP_COLOR[g],
                   linewidths=1.0,label='%s (%s)'%(m,g),zorder=3)
    mx=0.05
    ax.plot([0,mx],[0,mx],'-',color=MUTED,lw=.6,zorder=1)
    ax.set_xlabel(r'birth $\varepsilon_b$ / a.u.'); ax.set_ylabel(r'death $\varepsilon_d$ / a.u.')
    ax.set_title(r'$H_1$ persistence diagrams'); ax.legend(ncol=2); despine(ax)

def px_occl_vs_size(ax, fig=None):
    d=_df()
    for g in GROUPS:
        sel=d[d.grp==g]
        if not len(sel): continue
        ax.scatter(sel['nheavy'], sel['occlusion_ratio'], marker=GROUP_MARKER[g],
                   s=28, facecolors='none' if g!='reference' else GROUP_COLOR[g],
                   edgecolors=GROUP_COLOR[g], linewidths=1.0, label=g, zorder=3)
    ax.set_xlabel('heavy atoms'); ax.set_ylabel(r'occlusion ratio $\Omega$')
    ax.set_title(r'occlusion vs molecular size')
    ax.legend(ncol=2, fontsize=5.5); despine(ax)
