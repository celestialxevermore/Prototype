"""Split v10 (6 panels) into two 1x3 figures.
Group A: Medicaldataset, Cardiovascular, Erbil
Group B: cardio_SAheart, heart_failure, heart
Renders both PNG + PDF, both with/without n variants.
"""
import matplotlib.pyplot as plt
from ideal_radar_mock_4panels_10 import sources, plot_panel

OUT_DIR = '/home/eungyeop/LLM/tabular/ProtoLLM_entropic20251217/figures/radar'

GROUPS = {
    'a': ['Medicaldataset', 'Cardiovascular', 'Erbil'],
    'b': ['cardio_SAheart', 'heart_failure', 'heart'],
}


def render_group(tag, names, show_n):
    selected = [s for s in sources if s[0] in names]
    selected = sorted(selected, key=lambda s: names.index(s[0]))
    M = len(selected)
    fig, axes = plt.subplots(1, M, figsize=(6 * M, 6), subplot_kw=dict(polar=True))
    if M == 1:
        axes = [axes]
    for i, (short, _, num_cols, polys) in enumerate(selected):
        plot_panel(axes[i], short, num_cols, polys, show_n=show_n)
    plt.tight_layout()
    suffix = 'with' if show_n else 'no'
    base = f'{OUT_DIR}/ideal_radar_mock_4panels_10_split_{tag}_{suffix}'
    for ext in ('png', 'pdf'):
        path = f'{base}.{ext}'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


for tag, names in GROUPS.items():
    for show_n in (True, False):
        render_group(tag, names, show_n)
