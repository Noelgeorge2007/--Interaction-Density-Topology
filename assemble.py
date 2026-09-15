"""Assemble the single-file journal manuscript from the modular sources."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import re
P=str(paths.PAPER) + '/'
BIBPASS = '--bibpass' in sys.argv
OUT = (str(paths.SUBMISSION / 'pi-IDT-bibpass.tex') if BIBPASS
       else str(paths.SUBMISSION / 'pi-IDT.tex'))

def rd(f):
    if not f.endswith(('.tex','.bbl')): f+='.tex'
    return open(P+f).read().rstrip('\n')

# The body files still reference paper/figs/*.pdf, the draft figures that
# analysis.py and analysis2.py emit. Remap them to the published figures that
# export_figs.py draws and build_paper.sh stages into submission/figures/.
FIGMAP={'figs/fig1.pdf':'figures/Figure1_construction.pdf',
        'figs/fig2.pdf':'figures/Figure2_topology-and-shielding.pdf',
        'figs/fig3.pdf':'figures/Figure3_stability.pdf',
        'figs/fig4.pdf':'figures/Figure4_regression.pdf'}

def inline_tables(s):
    return re.sub(r'\\input\{(tables/[a-z0-9_]+)\}', lambda m: rd(m.group(1)), s)

def remap(s):
    for a,b in FIGMAP.items(): s=s.replace('{'+a+'}','{'+b+'}')
    return s

def floats_for(labels):
    out=[]
    for lab in labels:
        out.append(remap(inline_tables(rd('float_'+lab.replace(':','_')))))
    return '\n\n'.join(out)

def expand_markers(s):
    def rep(m):
        return floats_for(m.group(1).split(','))
    return re.sub(r'%%FLOATS:([^%]+)%%', rep, s)

HEAD = r"""%% ---------------------------------------------------------------------------
%%  pi-Interaction Density Topology: persistent homology of the accessible
%%  pi-electron density as a molecular descriptor
%%
%%  Single-file LaTeX source. Compile with two passes of pdflatex:
%%      pdflatex pi-IDT && pdflatex pi-IDT
%%  No bibtex run is required; the bibliography is embedded at the end.
%%
%%  Uses only standard TeX Live packages. Figures are read from ./figures/ as
%%  Figure1_construction.pdf ... Figure4_regression.pdf
%% ---------------------------------------------------------------------------
"""
parts=[HEAD]
for f in ['preamble','numbers','front']:
    parts.append('%%%%%%%% %s %%%%%%%%\n'%f + rd(f))
parts.append('\n\\begin{document}\n\\maketitle\n')
#  Introduction, Theory, Methods, Results, Discussion, Limitations,
#  Relation to prior art, Conclusions, then the back matter.
for f in ['abstract','body_1','body_2','body_5','body_3','body_4',
          'body_7','body_6','body_8']:
    parts.append('\n%%%%%%%% %s %%%%%%%%\n'%f + expand_markers(rd(f)) + '\n')
parts.append('\n\\clearpage\n')
for f in ['back']:
    parts.append('\n%%%%%%%% %s %%%%%%%%\n'%f + rd(f) + '\n')
parts.append('\n\\clearpage\n\n%%%%%%%% bibliography (embedded) %%%%%%%%\n')
parts.append('\\bibliography{refs}' if BIBPASS else rd('pidt.bbl'))
parts.append('\n\\end{document}\n')
s='\n'.join(parts)
if not BIBPASS:
    s=s.replace('\\bibliographystyle{unsrtnat}\n','')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT,'w').write(s)
left=re.findall(r'\\input\{[^}]*\}', s)+re.findall(r'%%FLOATS', s)
print('wrote %s: %d lines'%(OUT, s.count('\n')))
print('unresolved:', left)
print('figures:', re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]*)\}', s))
