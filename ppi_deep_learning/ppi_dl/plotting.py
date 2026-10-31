"""Plotting helpers shared by the notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"SARS-CoV": "#4C72B0", "SARS-CoV-2": "#DD8452"}


def plot_residue_importance(interface, pair_scores, title="", top_k=10, variant="cov2",
                            ax_pair=None):
    """Two-panel bar plot of per-residue importance for hACE2 and the RBD.

    ``pair_scores`` is one score per pair feature (length 4,886). Scores are
    summed onto residues and normalized to a maximum of 1, the NIGMS convention.
    The top ``top_k`` residues on each side are labelled.
    """
    ace, rbd = interface.residue_importance(np.abs(pair_scores))
    ace, rbd = ace / (ace.max() + 1e-12), rbd / (rbd.max() + 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for ax, s, res, name in [(axes[0], ace, interface.ace2_resids, "hACE2"),
                             (axes[1], rbd, interface.rbd_resids[variant],
                              "RBD (SARS-CoV-2 numbering)" if variant == "cov2" else "RBD (SARS-CoV numbering)")]:
        ax.bar(np.arange(len(s)), s, width=1.0, color="0.35")
        top = np.argsort(s)[::-1][:top_k]
        for t in top:
            ax.annotate(str(res[t]), (t, s[t]), textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7, rotation=90)
        ax.set_xlabel(f"{name} interface residue (token index)")
        ax.set_ylabel("normalized importance")
        ax.set_ylim(0, 1.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, (ace, rbd)


def top_residues(interface, pair_scores, k=10, variant="cov2"):
    """Return the top-k (residue number, score) for hACE2 and the RBD."""
    ace, rbd = interface.residue_importance(np.abs(pair_scores))
    ace, rbd = ace / ace.max(), rbd / rbd.max()
    ia, ir = np.argsort(ace)[::-1][:k], np.argsort(rbd)[::-1][:k]
    return ([(int(interface.ace2_resids[i]), round(float(ace[i]), 3)) for i in ia],
            [(int(interface.rbd_resids[variant][i]), round(float(rbd[i]), 3)) for i in ir])


def plot_feature_distributions(real, generated, feature_idx, labels=None, title="",
                               names=("real", "generated")):
    """Overlay histograms of a few features for real vs generated frames."""
    k = len(feature_idx)
    fig, axes = plt.subplots(1, k, figsize=(3.0 * k, 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, feature_idx):
        both = np.concatenate([real[:, j], generated[:, j]])
        lo, hi = np.percentile(both, [0.5, 99.5])      # ignore rare extreme contacts
        bins = np.linspace(lo, hi, 40)
        ax.hist(real[:, j], bins=bins, alpha=0.55, density=True, label=names[0], color="0.4")
        ax.hist(generated[:, j], bins=bins, alpha=0.55, density=True, label=names[1], color="#C44E52")
        ax.set_title(labels[j] if labels is not None else f"feature {j}", fontsize=8)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_pca_overlay(scorecard, X_gen, y_gen=None, title="", ax=None):
    """Project real held-out frames and generated frames onto the first 2 PCs."""
    Zr = scorecard.Zte
    Zg = scorecard.pca.transform(np.asarray(X_gen, dtype=np.float32))
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.4, 4))
    else:
        fig = ax.figure
    yr = scorecard.data.y_test
    for c, name in enumerate(["SARS-CoV", "SARS-CoV-2"]):
        ax.scatter(*Zr[yr == c, :2].T, s=5, alpha=0.35, color=COLORS[name], label=f"real {name}")
    ax.scatter(*Zg[:, :2].T, s=5, alpha=0.35, color="k", marker="x", label="generated")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax
