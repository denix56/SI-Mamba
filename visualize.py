#!/usr/bin/env python
"""
Point-cloud neighbourhood viewer with Plotly native K-slider.

Loads a .npz containing:
  - center       : B × K × N × 3      (for inferring N; not plotted here)
  - neighborhood : B × K × N × M × 3  (American or British spelling)

At startup, prompts for a batch index, then opens an interactive Plotly window
with a slider to scrub through the K-dimension. No HTML file is written.

Usage:
    python pc_viewer_k_slider.py clouds.npz

Dependencies:
    pip install plotly numpy
"""
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def load_data(npz_path):
    with np.load(npz_path) as f:
        C  = f['center']
        if 'neighborhood' in f:
            NB = f['neighborhood']
        else:
            NB = f['neighbourhood']

        NB = NB + C[..., None, :]
    return C, NB


def build_k_slider_figure(nbhd, max_points=None):
    """
    Build a Plotly Figure with a native slider for the K-dimension,
    given nbhd of shape (K, N, M, 3).
    """
    K, N, M, _ = nbhd.shape
    frames = []
    for k in range(K):
        slice_k = nbhd[k]  # (N, M, 3)
        pts = slice_k.reshape(-1, 3)
        color_idx = np.repeat(np.arange(N), M)
        scatter = go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(
                size=4,
                color=color_idx,
                colorscale='Turbo',
                cmin=0, cmax=N-1,
                opacity=0.8,
                showscale=True,
                colorbar=dict(title='N index')
            )
        )
        frames.append(go.Frame(name=str(k), data=[scatter]))

    # Initialize with K=0
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    # Configure slider
    steps = []
    for k in range(K):
        steps.append(dict(
            method='animate',
            label=str(k),
            args=[[str(k)],
                  dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))]
        ))

    slider = dict(
        active=0,
        currentvalue=dict(prefix='K = '),
        pad=dict(t=40),
        steps=steps
    )

    fig.update_layout(
        title='Batch (fixed)',
        scene=dict(aspectmode='data'),
        sliders=[slider],
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )
    return fig


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    npz_path = Path(sys.argv[1])
    if not npz_path.exists():
        print(f"File not found: {npz_path}")
        sys.exit(1)

    C, NB = load_data(npz_path)
    B, K, N, M, _ = NB.shape

    # Prompt for batch index
    while True:
        try:
            resp = input(f"Enter batch index (0 to {B-1}): ")
            b = int(resp)
            if 0 <= b < B:
                break
            else:
                print("Out of range.")
        except ValueError:
            print("Please enter an integer.")

    # Extract neighborhood for chosen batch: shape (K, N, M, 3)
    nbhd_batch = NB[b]

    # Build and show figure
    fig = build_k_slider_figure(nbhd_batch)
    fig.show()


if __name__ == '__main__':
    main()
