# data.py
import os
import numpy as np


class BRCADataConfig:
    """
    Central place for dataset paths and subset sizes.
    """

    def __init__(
        self,
        root_dir: str = "BRCA",
        view: int = 1,
        tr_keep: int = 50,
        te_keep: int = 20,
        feat_keep: int = 100,
    ):
        self.root_dir = root_dir
        self.view = view
        self.tr_keep = tr_keep
        self.te_keep = te_keep
        self.feat_keep = feat_keep

    @property
    def labels_tr(self):
        return os.path.join(self.root_dir, "labels_tr.csv")

    @property
    def labels_te(self):
        return os.path.join(self.root_dir, "labels_te.csv")

    @property
    def X_tr(self):
        return os.path.join(self.root_dir, f"{self.view}_tr.csv")

    @property
    def X_te(self):
        return os.path.join(self.root_dir, f"{self.view}_te.csv")


def load_brca_subset(cfg: BRCADataConfig):
    """
    Load BRCA subset with fixed sizes.
    Returns:
        X_all [N,Fin]
        y_all [N]
        idx_te
    """

    y_tr = np.loadtxt(cfg.labels_tr, delimiter=",").astype(int)
    y_te = np.loadtxt(cfg.labels_te, delimiter=",").astype(int)
    X_tr = np.loadtxt(cfg.X_tr, delimiter=",").astype(np.float32)
    X_te = np.loadtxt(cfg.X_te, delimiter=",").astype(np.float32)

    tr_keep = min(cfg.tr_keep, X_tr.shape[0])
    te_keep = min(cfg.te_keep, X_te.shape[0])
    feat_keep = min(cfg.feat_keep, X_tr.shape[1])

    X_tr = X_tr[:tr_keep, :feat_keep]
    y_tr = y_tr[:tr_keep]
    X_te = X_te[:te_keep, :feat_keep]
    y_te = y_te[:te_keep]

    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)

    uniq = np.unique(y_all)
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y_all = np.vectorize(mapping.get)(y_all).astype(int)

    idx_te = np.arange(tr_keep, tr_keep + te_keep, dtype=np.int64)

    return X_all, y_all, idx_te