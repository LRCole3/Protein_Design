"""A shared scorecard for judging generated interface frames.

Every generative notebook (VAE, GAN, diffusion, diffusion transformer) is scored
with the same yardsticks so the progression can be compared numerically.
All comparisons are made against held-out frames (``data.X_test``; use the
default ``split="block"`` so that train and test come from the same
simulations), and a reference row "real (train frames)" shows what an ideal
generator - one that produces genuinely new frames from the same simulations -
would score.

Metrics
-------
std_ratio    median sigma_gen / sigma_real  (<1 = over-smoothed, >1 = noisy)
mmd          RBF maximum mean discrepancy in a 20-D PCA space (0 = identical)
precision    fraction of generated frames that fall inside the real data
             manifold (k-NN balls of real frames)  -> sample *quality*
coverage     fraction of real frames that have a generated frame inside
             their k-NN ball                       -> sample *diversity*
             (Naeem et al., 2020, "Reliable Fidelity and Diversity Metrics")
c2st_auc     classifier two-sample test: AUC of a random forest trying
             to tell real from generated (0.5 = indistinguishable). Note that
             held-out MD blocks are themselves partly distinguishable from the
             training blocks (the reference row scores ~0.85), so compare a
             model against the reference, not against 0.5.
judge_acc    for conditional models: accuracy of a frozen SARS-CoV vs
             SARS-CoV-2 classifier on samples generated *for* each label
nn_ratio     median distance gen->nearest training frame divided by the same
             for real test frames. Much less than 1 = memorizing training data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _rbf_mmd2(A, B, gamma):
    def k(X, Y):
        d = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None] - 2 * X @ Y.T
        return np.exp(-gamma * np.maximum(d, 0))
    return float(k(A, A).mean() + k(B, B).mean() - 2 * k(A, B).mean())


class Scorecard:
    def __init__(self, data, n_pca: int = 20, k: int = 5, seed: int = 0):
        self.data, self.k, self.seed = data, k, seed
        self.pca = PCA(n_pca, random_state=seed).fit(data.X_train)
        self.Ztr = self.pca.transform(data.X_train)
        self.Zte = self.pca.transform(data.X_test)
        self.judge = LogisticRegression(C=0.01, max_iter=3000).fit(data.X_train, data.y_train)
        d = ((self.Zte[:, None] - self.Zte[None]) ** 2).sum(-1)
        self.gamma = 1.0 / np.median(d[d > 0])
        self.nn_real = NearestNeighbors(n_neighbors=k + 1).fit(self.Zte)
        self.radii = self.nn_real.kneighbors(self.Zte)[0][:, -1]
        self.nn_train = NearestNeighbors(n_neighbors=1).fit(self.Ztr)
        self.ref_nn = np.median(self.nn_train.kneighbors(self.Zte)[0])
        self.rows = []

    def evaluate(self, X_gen, y_gen=None, name="model", extra=None, store=True):
        X_gen = np.asarray(X_gen, dtype=np.float32)
        Xr = self.data.X_test
        Zg = self.pca.transform(X_gen)
        rng = np.random.default_rng(self.seed)
        row = {"model": name}
        row["std_ratio"] = float(np.median(X_gen.std(0) / (Xr.std(0) + 1e-8)))
        n = min(len(Zg), len(self.Zte), 1000)
        ig, ir = rng.choice(len(Zg), n, replace=False), rng.choice(len(self.Zte), n, replace=False)
        row["mmd"] = _rbf_mmd2(Zg[ig], self.Zte[ir], self.gamma)
        # precision: generated point lies within some real point's k-NN ball
        dist_all = ((Zg[:, None] - self.Zte[None]) ** 2).sum(-1) ** 0.5
        row["precision"] = float((dist_all <= self.radii[None]).any(1).mean())
        # coverage: real point has a generated point within its k-NN ball
        nn_g = NearestNeighbors(n_neighbors=1).fit(Zg).kneighbors(self.Zte)[0][:, 0]
        row["coverage"] = float((nn_g <= self.radii).mean())
        # classifier two-sample test
        Zc = np.concatenate([Zg[ig], self.Zte[ir]])
        yc = np.r_[np.ones(n), np.zeros(n)]
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                    random_state=self.seed)
        row["c2st_auc"] = float(cross_val_score(rf, Zc, yc, cv=5, scoring="roc_auc").mean())
        row["judge_acc"] = (float((self.judge.predict(X_gen) == np.asarray(y_gen)).mean())
                            if y_gen is not None else np.nan)
        row["nn_ratio"] = float(np.median(self.nn_train.kneighbors(Zg)[0]) / self.ref_nn)
        if extra:
            row.update(extra)
        if store:
            self.rows.append(row)
        return row

    def quick(self, X_gen):
        """Cheap progress check during training: precision, coverage and the
        fraction of samples the frozen judge calls SARS-CoV-2."""
        Zg = self.pca.transform(np.asarray(X_gen, dtype=np.float32))
        d = ((Zg[:, None] - self.Zte[None]) ** 2).sum(-1) ** 0.5
        return {"precision": float((d <= self.radii[None]).any(1).mean()),
                "coverage": float((d.min(0) <= self.radii).mean()),
                "frac_cov2": float(self.judge.predict(np.asarray(X_gen)).mean())}

    def reference(self):
        """Score the training run as if it were a generator (the target)."""
        row = self.evaluate(self.data.X_train, self.data.y_train, name="real (train frames)",
                            store=False)
        row["nn_ratio"] = np.nan  # a training frame is trivially its own nearest neighbour
        self.rows.append(row)
        return row

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).set_index("model").round(3)

    def save(self, notebook: str):
        """Append this notebook's rows to results/scorecard.csv (for later comparison)."""
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / "scorecard.csv"
        new = pd.DataFrame(self.rows).assign(notebook=notebook)
        if path.exists():
            old = pd.read_csv(path)
            old = old[old["notebook"] != notebook]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)
        return path.relative_to(RESULTS.parent)


def load_scorecards() -> pd.DataFrame:
    path = RESULTS / "scorecard.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
