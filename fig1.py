"""Figure 1: the pi-IDT construction, illustrated on benzene and coronene."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, json, pidt, opt
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7.5,'font.family':'serif',
                     'mathtext.fontset':'dejavuserif','savefig.bbox':'tight',
                     'savefig.pad_inches':0.02,'axes.linewidth':0.6})
P=str(paths.PAPER)
H=0.10; PAD=3.5

def build(smi):
    s,x,rd = pidt.geometry_from_smiles(smi)
    g = opt.prepare_geometry(s,x); x=g['xyz']
    mol,mf = pidt.run_dft(s,x)
    dm,_ = pidt.valence_dm(mf)
    dm_all = mf.make_rdm1()
    dmp,_,_ = pidt.pi_density_matrix_symmetry(mol, dm)
    pia,_ = pidt.pi_atom_set(mol, dmp)
    G,shape,ax = pidt.make_grid(x, pad=PAD, h=H)
    rho_t = pidt.eval_density(mol, dm_all, G).reshape(shape)
    rho_p = pidt.eval_density(mol, dmp, G).reshape(shape)
    S = pidt.accessibility_mask(G,s,x,pia).reshape(shape)
    return dict(syms=s,xyz=x,ax=ax,shape=shape,rho_t=rho_t,rho_p=rho_p,S=S,
                A=rho_p*S, bars=pidt.persistence((rho_p*S).ravel(), shape))

b = build('c1ccccc1')
ax_ = b['ax']; iz0 = int(np.argmin(np.abs(ax_[2])))
iz1 = int(np.argmin(np.abs(ax_[2]-1.0)))            # 1.0 A above the plane
iy0 = int(np.argmin(np.abs(ax_[1])))

fig = plt.figure(figsize=(7.0,4.4))
gs  = fig.add_gridspec(2,3, hspace=.38, wspace=.30)

def panel(k, F, title, cmap='viridis', vmax=None, plane='xy', iz=None):
    a=fig.add_subplot(gs[k])
    if plane=='xy':
        Z=F[:,:,iz].T; ext=[ax_[0][0],ax_[0][-1],ax_[1][0],ax_[1][-1]]
        xl,yl='$x$ / \u00c5','$y$ / \u00c5'
    else:
        Z=F[:,iy0,:].T; ext=[ax_[0][0],ax_[0][-1],ax_[2][0],ax_[2][-1]]
        xl,yl='$x$ / \u00c5','$z$ / \u00c5'
    im=a.imshow(Z,origin='lower',extent=ext,cmap=cmap,
                vmax=vmax if vmax else np.percentile(Z,99.7))
    a.set_title(title,fontsize=7.5); a.set_xlabel(xl); a.set_ylabel(yl)
    cb=fig.colorbar(im,ax=a,fraction=.046,pad=.03); cb.ax.tick_params(labelsize=6)
    return a

panel(0, b['rho_t'], r'(a) total density $\rho(\mathbf{r})$, $z=0$', iz=iz0)
panel(1, b['rho_p'], r'(b) $\pi$ density $\rho_\pi(\mathbf{r})$, $xz$ plane', plane='xz')
panel(2, b['S'], r'(c) accessibility $S(\mathbf{r})$, $z=0$', cmap='Greys_r', vmax=1, iz=iz0)
panel(3, b['A'], '(d) $A_\\pi(\\mathbf{r})=\\rho_\\pi S$, $z=1.0$ \u00c5', iz=iz1)

a=fig.add_subplot(gs[4])
cols={0:'#1b4965',1:'#c1121f',2:'#2a9d8f'}
for dim in (0,1,2):
    B=b['bars'][dim]
    if len(B)==0: continue
    L=B[:,0]-B[:,1]
    m=L>5e-3
    a.scatter(B[m,0],B[m,1],s=18,facecolors='none',edgecolors=cols[dim],
              label=r'$H_%d$'%dim,linewidths=.9)
    a.scatter(B[~m,0],B[~m,1],s=5,color=cols[dim],alpha=.25)
mx=max(b['bars'][0][:,0].max(),1e-9)
a.plot([0,mx],[0,mx],'k-',lw=.6)
a.set_xlabel('birth  $\\varepsilon_b$ / a.u.'); a.set_ylabel('death  $\\varepsilon_d$ / a.u.')
a.set_title('(e) persistence diagram (benzene)',fontsize=7.5); a.legend(fontsize=6)

a=fig.add_subplot(gs[5])
y=0
for dim in (0,1):
    B=b['bars'][dim]; L=B[:,0]-B[:,1]; idx=np.argsort(-L)
    for i in idx:
        if L[i]<1e-3: continue
        a.plot([B[i,1],B[i,0]],[y,y],color=cols[dim],lw=1.6); y+=1
a.set_xlabel('$\\varepsilon$ / a.u.'); a.set_yticks([])
a.set_title(r'(f) $\pi$-barcode ($H_0$ blue, $H_1$ red)',fontsize=7.5)
fig.savefig(f'{P}/figs/fig1.pdf')
print('fig1 written; benzene H1 lifetimes',
      np.round(np.sort((b['bars'][1][:,0]-b['bars'][1][:,1]))[::-1][:4],4))
