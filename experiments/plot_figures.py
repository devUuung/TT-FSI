"""Regenerate the paper's two data figures from the saved California case."""
import argparse
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from figstyle import use_style, figure, save, COLUMN, PALETTE, DIVERGING

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Inc', 'Age', 'Rm', 'Bed', 'Pop', 'Occ', 'Lat', 'Lon']


def run(data=None, destination=None):
    """Plot only scores whose manifest matches the current execution fingerprint."""
    from paper_tables import execution_fingerprint,file_hash
    config=json.loads((ROOT/'configs/paper_reproduction.json').read_text())
    base=ROOT/config['output_directory']
    data = Path(data) if data is not None else base/'california'
    manifest=json.loads((data/'cache.json').read_text())
    if manifest.get('execution_fingerprint')!=execution_fingerprint():
        raise ValueError('California figures require a current validated measurement')
    for name,sha in manifest['files'].items():
        if file_hash(data/name)!=sha:raise ValueError('California score checksum mismatch')
    destination = Path(destination) if destination is not None else base/'figures'
    destination.mkdir(parents=True, exist_ok=True)
    scores_file = np.load(data/'scores-ell2.npz')
    masks = scores_file['shapiq_masks']; good = masks != 0
    reference = scores_file['shapiq_values'][good]; tt = scores_file['cpu'][masks[good]]
    use_style()

    fig, ax = figure(width=COLUMN, ratio=0.9)
    lo = min(reference.min(), tt.min()); hi = max(reference.max(), tt.max()); pad = (hi-lo)*0.07
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color='0.6', lw=0.8, zorder=1)
    ax.scatter(reference, tt, s=15, color=PALETTE['blue'], edgecolors='white', linewidths=0.3, zorder=2)
    ax.set_xlabel('shapiq exact FSI score'); ax.set_ylabel('TT-FSI score')
    ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)
    save(fig, str(destination/'shapiq_california_agreement'))

    scores = scores_file['cpu']
    plt.rcParams.update({'font.size': 5.5, 'axes.labelsize': 5.5, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5})
    matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8): matrix[i, j] = scores[(1 << i) | (1 << j)]
    pairs = sorted(((abs(scores[(1 << i) | (1 << j)]), i, j) for i in range(8) for j in range(i+1, 8)), reverse=True)[:10]
    # Wide two-panel layout at the printed column width (about 2.3:1, as in the
    # submitted manuscript); the colorbar is sized to the heatmap's height.
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.42), gridspec_kw={'width_ratios': [1.0, 1.25]}, layout='constrained')
    cap = np.max(np.abs(matrix)); im = axes[0].imshow(matrix, cmap=DIVERGING, vmin=-cap, vmax=cap, interpolation='nearest')
    axes[0].set_xticks(range(8), LABELS, rotation=90, ha='center'); axes[0].set_yticks(range(8), LABELS)
    axes[0].set_xlabel('Feature'); axes[0].set_ylabel('Feature')
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='FSI')
    names = [LABELS[i]+' × '+LABELS[j] for _, i, j in pairs]
    signed = np.array([scores[(1 << i) | (1 << j)] for _, i, j in pairs])
    axes[1].barh(np.arange(10), signed, color=[PALETTE['blue'] if v >= 0 else PALETTE['orange'] for v in signed])
    axes[1].set_yticks(np.arange(10), names); axes[1].invert_yaxis(); axes[1].axvline(0, color='0.4', lw=0.7)
    axes[1].set_xlabel('Pairwise FSI'); axes[1].set_ylabel('Interaction')
    for ax, label in zip(axes, ['(a)', '(b)']): ax.text(0, 1.04, label, transform=ax.transAxes, fontweight='bold')
    save(fig, str(destination/'california_fsi_instance'))
    print('PAPER_FIGURES_WRITTEN '+str(destination), flush=True)
    return destination


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=None)
    ap.add_argument('--destination', type=Path, default=None)
    args = ap.parse_args(); run(args.data, args.destination)


if __name__ == '__main__': main()
