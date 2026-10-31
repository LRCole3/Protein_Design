"""Data loading for the SARS-CoV / SARS-CoV-2 : hACE2 interface dataset.

The data come from the NIGMS Sandbox module
"Protein-Protein Interactions using ML"
(https://github.com/NIGMS/Protein-Protein-Interactions-using-ML).

Each MD frame is described by 4,886 distances (nm) between C-alpha atoms of an
hACE2 residue and a viral receptor-binding-domain (RBD) residue that come within
1.5 nm of each other at the interface. There are two independent 1,000-frame
runs per variant.

Note: the upstream submodule 3 notebook loads ``cov2_run2`` from
``cropped_cov1_run2.npy`` (a copy-paste slip). Here every run is loaded from
its own file.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE_URL = ("https://raw.githubusercontent.com/NIGMS/"
            "Protein-Protein-Interactions-using-ML/main/tutorial_files/ML_input_files/")
RUN_FILES = {
    ("cov1", 1): "cropped_cov1_run1.npy",
    ("cov1", 2): "cropped_cov1_run2.npy",
    ("cov2", 1): "cropped_cov2_run1.npy",
    ("cov2", 2): "cropped_cov2_run2.npy",
}
PAIR_FILES = {"cov1": "cov1pairs_resids.15.txt", "cov2": "cov2pairs_resids.15.txt"}
LABELS = {"cov1": 0, "cov2": 1}
CLASS_NAMES = ["SARS-CoV", "SARS-CoV-2"]

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------- utils
def set_seed(seed: int = 0) -> None:
    """Seed python, numpy and (if installed) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device():
    """Return a CUDA/MPS device if available, otherwise CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def data_dir(path: str | os.PathLike | None = None) -> Path:
    """Resolve the data directory (argument > $PPI_DATA_DIR > ./data)."""
    p = Path(path or os.environ.get("PPI_DATA_DIR", _DEFAULT_DIR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_data(path: str | os.PathLike | None = None, verbose: bool = True) -> Path:
    """Download the NIGMS ML input files once (~78 MB) and cache them locally.

    If you already cloned the NIGMS repository, point ``path`` (or the
    ``PPI_DATA_DIR`` environment variable) at its
    ``tutorial_files/ML_input_files`` folder and nothing is downloaded.
    """
    import urllib.request

    d = data_dir(path)
    for name in list(RUN_FILES.values()) + list(PAIR_FILES.values()):
        target = d / name
        if target.exists() and target.stat().st_size > 0:
            continue
        if verbose:
            print(f"downloading {name} ...")
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(BASE_URL + name, tmp)
        tmp.replace(target)
    return d


# --------------------------------------------------------------------------- residues
@dataclass
class Interface:
    """Bookkeeping that maps the 4,886 pair features onto residues.

    Residue tokens are aligned between variants: RBD token ``j`` is the same
    structural position in SARS-CoV and SARS-CoV-2 (their numbering differs by
    13-14 because of an insertion), so a model can share weights across both.
    """
    pairs: dict                      # variant -> (n_pairs, 2) array of resids
    ace2_resids: np.ndarray          # (n_ace2,) hACE2 residue numbers
    rbd_resids: dict                 # variant -> (n_rbd,) RBD residue numbers
    pair_ace2_idx: np.ndarray        # (n_pairs,) token index of the ACE2 residue
    pair_rbd_idx: np.ndarray         # (n_pairs,) token index of the RBD residue
    n_ace2: int = field(init=False)
    n_rbd: int = field(init=False)

    def __post_init__(self):
        self.n_ace2 = len(self.ace2_resids)
        self.n_rbd = len(self.rbd_resids["cov2"])

    def pair_labels(self, variant: str = "cov2") -> list[str]:
        p = self.pairs[variant]
        return [f"ACE2-{a}:RBD-{b}" for a, b in p]

    def residue_importance(self, pair_scores: np.ndarray, reduce: str = "sum"):
        """Collapse per-pair scores (n_pairs,) to per-residue scores.

        Returns ``(ace2_scores, rbd_scores)`` indexed by token. This mirrors the
        NIGMS approach of summing pair importances onto each residue.
        """
        pair_scores = np.asarray(pair_scores, dtype=float)
        ace = np.zeros(self.n_ace2)
        rbd = np.zeros(self.n_rbd)
        np.add.at(ace, self.pair_ace2_idx, pair_scores)
        np.add.at(rbd, self.pair_rbd_idx, pair_scores)
        if reduce == "mean":
            ace /= np.maximum(np.bincount(self.pair_ace2_idx, minlength=self.n_ace2), 1)
            rbd /= np.maximum(np.bincount(self.pair_rbd_idx, minlength=self.n_rbd), 1)
        return ace, rbd

    def to_pair_matrix(self, X: np.ndarray, fill: float = 0.0):
        """Scatter flat features (N, n_pairs) into (N, n_ace2, n_rbd) + mask."""
        X = np.asarray(X)
        M = np.full((X.shape[0], self.n_ace2, self.n_rbd), fill, dtype=np.float32)
        M[:, self.pair_ace2_idx, self.pair_rbd_idx] = X
        mask = np.zeros((self.n_ace2, self.n_rbd), dtype=bool)
        mask[self.pair_ace2_idx, self.pair_rbd_idx] = True
        return M, mask


def load_interface(path=None) -> Interface:
    d = download_data(path, verbose=False)
    pairs = {v: np.loadtxt(d / f, dtype=int) for v, f in PAIR_FILES.items()}
    ace2 = np.unique(pairs["cov2"][:, 0])
    assert np.array_equal(pairs["cov1"][:, 0], pairs["cov2"][:, 0])
    rbd = {v: np.unique(p[:, 1]) for v, p in pairs.items()}
    a_idx = np.searchsorted(ace2, pairs["cov2"][:, 0])
    r_idx = np.searchsorted(rbd["cov2"], pairs["cov2"][:, 1])
    # the same token indices must come out of the CoV numbering
    assert np.array_equal(r_idx, np.searchsorted(rbd["cov1"], pairs["cov1"][:, 1]))
    return Interface(pairs, ace2, rbd, a_idx, r_idx)


# --------------------------------------------------------------------------- dataset
@dataclass
class PPIData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    interface: Interface
    split: str
    class_names: list = field(default_factory=lambda: list(CLASS_NAMES))

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def to_distance(self, X_std: np.ndarray) -> np.ndarray:
        """Undo standardization and the 1/d transform -> distances in nm."""
        inv = X_std * self.std + self.mean
        with np.errstate(divide="ignore"):
            return 1.0 / inv

    def summary(self) -> str:
        return (f"split={self.split}: train {self.X_train.shape}, test {self.X_test.shape}, "
                f"train class counts {np.bincount(self.y_train).tolist()}, "
                f"test class counts {np.bincount(self.y_test).tolist()}")


def load_runs(path=None) -> dict:
    d = download_data(path)
    return {k: np.load(d / f).astype(np.float32) for k, f in RUN_FILES.items()}


def load_ppi(split: str = "block", test_size: float = 0.2, seed: int = 0,
             block: int = 50, path=None) -> PPIData:
    """Load, inverse-transform (1/d) and standardize the interface features.

    split="run"    : train on run 1 of each variant, test on run 2 (honest:
                     frames from an independent simulation).
    split="block"  : cut every trajectory into ``block``-frame chunks and hold
                     out every 5th chunk (``test_size`` = 0.2). Train and test
                     come from the same simulations (same distribution) but
                     neighbouring frames rarely straddle the split. This is the
                     default and the split the generative notebooks use.
    split="random" : shuffle all 4,000 frames and hold out ``test_size``
                     (what the NIGMS notebook does; neighbouring frames from the
                     same trajectory end up on both sides of the split).

    Standardization statistics are computed on the training set only.
    Labels: 0 = SARS-CoV, 1 = SARS-CoV-2.
    """
    runs = load_runs(path)
    iface = load_interface(path)
    inv = {k: 1.0 / v for k, v in runs.items()}

    if split == "run":
        Xtr = np.concatenate([inv[("cov1", 1)], inv[("cov2", 1)]])
        ytr = np.repeat([0, 1], [len(inv[("cov1", 1)]), len(inv[("cov2", 1)])])
        Xte = np.concatenate([inv[("cov1", 2)], inv[("cov2", 2)]])
        yte = np.repeat([0, 1], [len(inv[("cov1", 2)]), len(inv[("cov2", 2)])])
    elif split == "random":
        keys = [("cov1", 1), ("cov1", 2), ("cov2", 1), ("cov2", 2)]
        X = np.concatenate([inv[k] for k in keys])
        y = np.concatenate([np.full(len(inv[k]), LABELS[k[0]]) for k in keys])
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_test = int(round(test_size * len(X)))
        te, tr = idx[:n_test], idx[n_test:]
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    elif split == "block":
        keys = [("cov1", 1), ("cov1", 2), ("cov2", 1), ("cov2", 2)]
        every = int(round(1 / test_size))
        Xtr, ytr, Xte, yte = [], [], [], []
        for k in keys:
            A = inv[k]
            is_test = (np.arange(len(A)) // block) % every == every - 1
            Xtr.append(A[~is_test]); Xte.append(A[is_test])
            ytr.append(np.full((~is_test).sum(), LABELS[k[0]]))
            yte.append(np.full(is_test.sum(), LABELS[k[0]]))
        Xtr, ytr, Xte, yte = map(np.concatenate, (Xtr, ytr, Xte, yte))
    else:
        raise ValueError("split must be 'block', 'run' or 'random'")

    mean = Xtr.mean(0)
    std = Xtr.std(0) + 1e-6
    f = lambda A: ((A - mean) / std).astype(np.float32)
    return PPIData(f(Xtr), ytr.astype(np.int64), f(Xte), yte.astype(np.int64),
                   mean.astype(np.float32), std.astype(np.float32), iface, split)
