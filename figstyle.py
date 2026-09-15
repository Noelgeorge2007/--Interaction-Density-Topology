"""Shared plotting style for every pi-IDT figure.

Categorical palette validated with the dataviz palette checker
(light surface #fcfcfb): all seven slots pass the lightness band, chroma floor,
CVD separation (worst adjacent dE 8.3 protan), the normal-vision floor
(worst 15.1) and the 3:1 contrast check. Group identity is additionally carried
by marker shape, so colour is never the only encoding.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE = ['#2f6fb2', '#c9333f', '#128f80', '#9c7000',
           '#7d4fb3', '#c94f96', '#4a7a2b']
GREY   = '#6b6b6b'
INK    = '#1a1a1a'
MUTED  = '#767676'
GRID   = '#d9d9d9'

# fixed order, never cycled
GROUPS = ['acene', 'PAH', 'heteroarene', 'macrocycle',
          'substituted', 'shielding', 'non-planar', 'reference']
GROUP_COLOR  = dict(zip(GROUPS, PALETTE + [GREY]))
GROUP_MARKER = {'acene':'o','PAH':'s','heteroarene':'^','macrocycle':'D',
                'substituted':'v','shielding':'<','non-planar':'P','reference':'X'}
# homology dimension colours (sequential in role, distinct in hue)
DIMCOL = {0: PALETTE[0], 1: PALETTE[1], 2: PALETTE[2]}

def regroup(g):
    """Collapse the raw group labels to the eight plotted categories."""
    return {'biaryl':'non-planar','curved':'non-planar'}.get(g, g)

def use_style(base=8.0):
    plt.rcParams.update({
        'font.size': base, 'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.6, 'axes.edgecolor': MUTED,
        'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': base-1, 'ytick.labelsize': base-1,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4,
        'grid.alpha': 0.8, 'axes.axisbelow': True,
        'legend.frameon': False, 'legend.fontsize': base-2,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'savefig.facecolor': 'white',
    })

def despine(ax, keep=('left','bottom')):
    for side in ('top','right','left','bottom'):
        ax.spines[side].set_visible(side in keep)
