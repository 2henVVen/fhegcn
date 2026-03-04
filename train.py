# train.py
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score

from model import PolyGCN


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fixed_subset(data_folder: str, view: int, tr_keep: int, te_keep: int, feat_keep: int):
    """
    MOGONET-style:
      labels_tr.csv, labels_te.csv
      {view}_tr.csv, {view}_te.csv
    Keep:
      first tr_keep train rows + first te_keep test rows
      first feat_keep columns
    """
    y_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=",").astype(int)
    y_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=",").astype(int)

    X_tr = np.loadtxt(os.path.join(data_folder, f"{view}_tr.csv"), delimiter=",").astype(np.float32)
    X_te = np.loadtxt(os.path.join(data_folder, f"{view}_te.csv"), delimiter=",").astype(np.float32)

    tr_keep = min(tr_keep, X_tr.shape[0])
    te_keep = min(te_keep, X_te.shape[0])
    feat_keep = min(feat_keep, X_tr.shape[1])

    X_tr = X_tr[:tr_keep, :feat_keep]
    y_tr = y_tr[:tr_keep]
    X_te = X_te[:te_keep, :feat_keep]
    y_te = y_te[:te_keep]

    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)

    # remap labels to 0..K-1
    uniq = np.unique(y_all)
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y_all = np.vectorize(mapping.get)(y_all).astype(int)

    idx_tr = np.arange(0, tr_keep, dtype=np.int64)
    idx_te = np.arange(tr_keep, tr_keep + te_keep, dtype=np.int64)

    return X_all, y_all, idx_tr, idx_te, mapping


def cosine_topk_adj(X: torch.Tensor, topk: int, add_self_loop: bool = True) -> torch.Tensor:
    """
    Dense normalized adjacency A_hat via cosine topk neighbors.
    """
    N = X.size(0)
    Xn = F.normalize(X, p=2, dim=1)
    S = Xn @ Xn.t()

    diag = torch.arange(N, device=X.device)
    S = S.clone()
    S[diag, diag] = -1e9

    k = min(topk, N - 1)
    _, nn_idx = torch.topk(S, k=k, dim=1)

    A = torch.zeros((N, N), dtype=torch.float32, device=X.device)
    row = torch.arange(N, device=X.device).unsqueeze(1).expand(N, k)
    A[row, nn_idx] = 1.0
    A = torch.maximum(A, A.t())

    if add_self_loop:
        A[diag, diag] = 1.0

    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    A_hat = (deg_inv_sqrt.unsqueeze(1) * A) * deg_inv_sqrt.unsqueeze(0)
    return A_hat


@torch.no_grad()
def eval_te(model: nn.Module, X: torch.Tensor, A_hat: torch.Tensor,
            y_cpu: np.ndarray, idx_te_cpu: np.ndarray, num_class: int):
    model.eval()
    logits = model(X, A_hat).detach().cpu()
    y_true = y_cpu[idx_te_cpu]
    y_pred = logits[idx_te_cpu].argmax(dim=1).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=("binary" if num_class == 2 else "macro"))
    return float(acc), float(f1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", type=str, required=True)
    p.add_argument("--view", type=int, default=1)
    p.add_argument("--num_class", type=int, required=True)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--topk", type=int, default=10)

    # fixed subset defaults you asked for
    p.add_argument("--tr_keep", type=int, default=50)
    p.add_argument("--te_keep", type=int, default=20)
    p.add_argument("--feat_keep", type=int, default=100)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--a_init", type=float, default=0.005)
    p.add_argument("--b_init", type=float, default=1.0)
    p.add_argument("--c_init", type=float, default=0.0)
    p.add_argument("--freeze_poly", action="store_true")

    p.add_argument("--out_dir", type=str, default="./work_dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- load subset
    X_all_np, y_all_np, idx_tr, idx_te, label_map = load_fixed_subset(
        args.data_folder, args.view, args.tr_keep, args.te_keep, args.feat_keep
    )
    N, Fin = X_all_np.shape
    print(f"Using fixed subset: N={N} (tr={args.tr_keep}, te={args.te_keep}), Fin={Fin} (feat_keep={args.feat_keep})")
    print(f"Label uniq(remapped): {sorted(np.unique(y_all_np).tolist())}")

    # --- tensors
    X = torch.tensor(X_all_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_all_np, dtype=torch.long, device=device)

    # --- build A_hat once
    t0 = time.time()
    A_hat = cosine_topk_adj(X, topk=args.topk, add_self_loop=True)
    print(f"Built A_hat in {time.time()-t0:.3f}s, nnz≈{int((A_hat>0).sum().item())}")

    # --- model
    model = PolyGCN(
        in_dim=Fin,
        hidden_dim=args.hidden_dim,
        out_dim=args.num_class,
        a_init=args.a_init,
        b_init=args.b_init,
        c_init=args.c_init,
        poly_trainable=not args.freeze_poly,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.out_dir,
        f"best_poly_view{args.view}_sub{args.tr_keep}+{args.te_keep}_feat{args.feat_keep}.bundle.pt"
    )

    best_acc = -1.0
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X, A_hat)
        loss = crit(logits[idx_tr], y[idx_tr])
        loss.backward()
        opt.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            acc, f1 = eval_te(model, X, A_hat, y_all_np, idx_te, args.num_class)
            print(f"[Epoch {epoch:04d}] loss={loss.item():.4f} | te_acc={acc:.4f} te_f1={f1:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_metrics = {"acc": acc, "f1": f1, "epoch": epoch}

                bundle = {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "label_mapping": label_map,
                    "best_metrics": best_metrics,
                    # Save plaintext A_hat too (for reproducibility / debugging)
                    "A_hat": A_hat.detach().cpu(),
                }
                torch.save(bundle, ckpt_path)

    print(f"\nDone. Best te_acc={best_acc:.4f}")
    print(f"Saved bundle ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()