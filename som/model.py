"""
Self-organizing map trained online (one sample per iteration), Euclidean distance,
Gaussian neighbourhood on a rectangular grid.

sigma(t) = sigma0 * exp(-t / (T / ln sigma0)), from max(rows, cols) / 2 down to 1
lr(t)    = lr0 * exp(-t / T)

Samples are drawn with replacement in blocks of BLOCK hours from a seeded
generator. Checkpoints are written at block boundaries together with the
generator state, so an interrupted run continues exactly as an uninterrupted one.
"""
import math
import pickle
from pathlib import Path

import numpy as np

BLOCK = 50_000


class SOM:
    def __init__(self, rows, cols, dim, seed=42, lr0=0.5, sigma0=None):
        self.rows, self.cols, self.dim, self.seed = rows, cols, dim, seed
        self.n = rows * cols
        self.lr0 = lr0
        self.sigma0 = sigma0 or max(rows, cols) / 2.0
        rng = np.random.default_rng(seed)
        self.weights = (rng.standard_normal((self.n, dim)) * 0.1).astype("float64")
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        self.grid = np.stack([rr.ravel(), cc.ravel()], axis=1).astype("float64")
        self.iteration = 0
        self.rng_state = rng.bit_generator.state

    def _sigma(self, t, T):
        return self.sigma0 * math.exp(-t / (T / math.log(self.sigma0)))

    def _lr(self, t, T):
        return self.lr0 * math.exp(-t / T)

    def fit(self, X, n_iter, ckpt=None, log=None):
        rng = np.random.default_rng()
        rng.bit_generator.state = self.rng_state
        N = len(X)
        t = self.iteration
        W, G = self.weights, self.grid
        while t < n_iter:
            idx = rng.integers(0, N, BLOCK)
            order = np.argsort(idx, kind="stable")
            block = np.empty((BLOCK, self.dim), dtype="float64")
            block[order] = np.asarray(X[idx[order]], dtype="float64")
            for b in range(BLOCK):
                if t >= n_iter:
                    break
                x = block[b]
                diff = x - W
                bmu = int(np.einsum("ij,ij->i", diff, diff).argmin())
                s = self._sigma(t, n_iter)
                g = G - G[bmu]
                h = np.exp(-(g[:, 0] ** 2 + g[:, 1] ** 2) / (2 * s * s))
                W += (self._lr(t, n_iter) * h)[:, None] * diff
                t += 1
            self.iteration = t
            self.rng_state = rng.bit_generator.state
            if ckpt:
                self.save(ckpt)
            if log:
                log.info(f"  iteration {t:,}/{n_iter:,} sigma {self._sigma(t, n_iter):.3f}")

    def assign(self, X, batch=8192):
        """Best and second best unit per row, and the quantization error of the best."""
        W = self.weights
        w2 = (W * W).sum(axis=1)
        bmu = np.empty(len(X), dtype=np.int16)
        bmu2 = np.empty(len(X), dtype=np.int16)
        qe = np.empty(len(X), dtype=np.float32)
        for s in range(0, len(X), batch):
            x = np.asarray(X[s:s + batch], dtype="float64")
            d2 = (x * x).sum(axis=1)[:, None] - 2.0 * x @ W.T + w2
            two = np.argpartition(d2, 1, axis=1)[:, :2]
            dd = np.take_along_axis(d2, two, axis=1)
            sw = dd[:, 1] < dd[:, 0]
            two[sw] = two[sw][:, ::-1]
            bmu[s:s + batch] = two[:, 0]
            bmu2[s:s + batch] = two[:, 1]
            qe[s:s + batch] = np.sqrt(np.maximum(np.take_along_axis(d2, two[:, :1], axis=1)[:, 0], 0))
        return bmu, bmu2, qe

    def save(self, path):
        d = dict(weights=self.weights, rows=self.rows, cols=self.cols, dim=self.dim, seed=self.seed,
                 lr0=self.lr0, sigma0=self.sigma0, iteration=self.iteration, rng_state=self.rng_state)
        tmp = Path(str(path) + ".tmp")
        tmp.write_bytes(pickle.dumps(d))
        tmp.replace(path)

    def export(self, path):
        """Weights and settings in plain formats (.npz and .json), for use without pickle."""
        import json
        path = Path(path)
        np.savez_compressed(path.with_suffix(".npz"), weights=self.weights.astype("float64"))
        path.with_suffix(".json").write_text(json.dumps(dict(
            rows=self.rows, cols=self.cols, dim=self.dim, seed=self.seed, lr0=self.lr0,
            sigma0=self.sigma0, iterations=self.iteration), indent=2))

    @classmethod
    def from_export(cls, path):
        import json
        path = Path(path)
        d = json.loads(path.with_suffix(".json").read_text())
        som = cls(d["rows"], d["cols"], d["dim"], d["seed"], d["lr0"], d["sigma0"])
        som.weights = np.load(path.with_suffix(".npz"))["weights"]
        som.iteration = d["iterations"]
        return som

    @classmethod
    def load(cls, path):
        d = pickle.loads(Path(path).read_bytes())
        som = cls(d["rows"], d["cols"], d["dim"], d["seed"], d["lr0"], d["sigma0"])
        som.weights, som.iteration, som.rng_state = d["weights"], d["iteration"], d["rng_state"]
        return som
