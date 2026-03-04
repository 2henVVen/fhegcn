# amainfer.py
import os
import time
import argparse
import numpy as np

import torch
from sklearn.metrics import accuracy_score, f1_score

from model import PolyGCN
from amamodel import AMAModel, setup_ckks, reset_op_counts, print_op_counts


# -------------------------
# Fixed config (as you requested)
# -------------------------
DATA_ROOT = "BRCA"
VIEW = 1
TR_KEEP = 50
TE_KEEP = 20
FEAT_KEEP = 100
NUM_CLASS = 5  # always 5 classes


def load_fixed_subset():
    """
    Load MOGONET-style files from ./BRCA:
      labels_tr.csv, labels_te.csv
      {view}_tr.csv, {view}_te.csv

    Keep:
      first TR_KEEP train rows
      first TE_KEEP test rows
      first FEAT_KEEP columns
    """
    y_tr_path = os.path.join(DATA_ROOT, "labels_tr.csv")
    y_te_path = os.path.join(DATA_ROOT, "labels_te.csv")
    X_tr_path = os.path.join(DATA_ROOT, f"{VIEW}_tr.csv")
    X_te_path = os.path.join(DATA_ROOT, f"{VIEW}_te.csv")

    if not os.path.exists(y_tr_path):
        raise FileNotFoundError(f"{y_tr_path} not found.")
    if not os.path.exists(y_te_path):
        raise FileNotFoundError(f"{y_te_path} not found.")
    if not os.path.exists(X_tr_path):
        raise FileNotFoundError(f"{X_tr_path} not found.")
    if not os.path.exists(X_te_path):
        raise FileNotFoundError(f"{X_te_path} not found.")

    y_tr = np.loadtxt(y_tr_path, delimiter=",").astype(int)
    y_te = np.loadtxt(y_te_path, delimiter=",").astype(int)
    X_tr = np.loadtxt(X_tr_path, delimiter=",").astype(np.float32)
    X_te = np.loadtxt(X_te_path, delimiter=",").astype(np.float32)

    tr_keep = min(TR_KEEP, X_tr.shape[0])
    te_keep = min(TE_KEEP, X_te.shape[0])
    feat_keep = min(FEAT_KEEP, X_tr.shape[1])

    X_tr = X_tr[:tr_keep, :feat_keep]
    y_tr = y_tr[:tr_keep]
    X_te = X_te[:te_keep, :feat_keep]
    y_te = y_te[:te_keep]

    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)

    # remap labels to 0..K-1 (training did the same)
    uniq = np.unique(y_all)
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y_all = np.vectorize(mapping.get)(y_all).astype(int)

    idx_te = np.arange(tr_keep, tr_keep + te_keep, dtype=np.int64)
    return X_all, y_all, idx_te


@torch.no_grad()
def plain_check(model, X, A_hat, y_all_np, idx_te):
    model.eval()
    logits = model(X, A_hat).detach().cpu()
    y_true = y_all_np[idx_te]
    y_pred = logits[idx_te].argmax(dim=1).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(acc), float(f1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="bundle checkpoint path (*.bundle.pt)")

    # CKKS params (you can tune)
    p.add_argument("--ring_dim", type=int, default=8192)
    p.add_argument("--mult_depth", type=int, default=10)
    p.add_argument("--first_mod_size", type=int, default=34)
    p.add_argument("--scaling_mod_size", type=int, default=30)

    # Optional: list samples whose prediction changed (plain vs HE)
    p.add_argument("--show_changed", action="store_true", help="print test indices whose argmax changed")
    args = p.parse_args()

    # 1) load data (fixed)
    X_all_np, y_all_np, idx_te = load_fixed_subset()
    N, Fin = X_all_np.shape
    print(f"[Data] root=./{DATA_ROOT} view={VIEW} subset=(tr{TR_KEEP}, te{TE_KEEP}, feat{FEAT_KEEP})")
    print(f"[Data] N={N}, Fin={Fin}, NUM_CLASS fixed={NUM_CLASS}")

    X_plain = torch.tensor(X_all_np, dtype=torch.float32)  # CPU tensor

    # 2) load bundle (trained model + saved A_hat)
    bundle = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = bundle["model_state_dict"]
    train_args = bundle.get("args", {})

    A_hat = bundle.get("A_hat", None)
    if A_hat is None:
        raise ValueError("Bundle does not contain A_hat. Please use bundle saved by your train.py (it saves A_hat).")
    if not torch.is_tensor(A_hat):
        A_hat = torch.tensor(A_hat, dtype=torch.float32)
    A_hat = A_hat.to(torch.float32)  # CPU

    hidden_dim = int(train_args.get("hidden_dim", 64))
    topk = int(train_args.get("topk", 10))

    # 3) rebuild plaintext model and load weights
    model = PolyGCN(in_dim=Fin, hidden_dim=hidden_dim, out_dim=NUM_CLASS).cpu()
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[Model] loaded from bundle. hidden_dim={hidden_dim}, out_dim={NUM_CLASS}")
    print(f"[Graph] neighbor topk={topk} (from bundle args)")

    # 4) plaintext sanity (on same A_hat)
    plain_acc, plain_f1 = plain_check(model, X_plain, A_hat, y_all_np, idx_te)
    print(f"[Plain] te only: acc={plain_acc:.4f} f1(macro)={plain_f1:.4f}")

    # 5) extract weights for HE model
    Wgcn = model.gcn.weight.detach().cpu()
    bgcn = model.gcn.bias.detach().cpu()
    a = float(model.act.a.detach().cpu().item())
    b = float(model.act.b.detach().cpu().item())
    c = float(model.act.c.detach().cpu().item())
    Wclf = model.clf.weight.detach().cpu()
    bclf = model.clf.bias.detach().cpu()

    # 6) CKKS setup: need rotation keys up to L/2 for slot-sum in dot-product
    L = 1
    while L < Fin:
        L <<= 1
    max_rot_step = L // 2

    cc, pk, sk = setup_ckks(
        max_rot_step=max_rot_step,
        ring_dim=args.ring_dim,
        mult_depth=args.mult_depth,
        first_mod_size=args.first_mod_size,
        scaling_mod_size=args.scaling_mod_size,
    )

    # 7) Build AMA wrapper and neighbors from dense A_hat
    ama = AMAModel(cc, pk, N, Fin, Wgcn, bgcn, a, b, c, Wclf, bclf, num_class=NUM_CLASS)
    neigh = ama.build_neighbors_from_dense(A_hat, topk=topk, include_self=True)

    # 8) encrypt nodes
    t1 = time.time()
    enc_nodes = ama.encrypt_nodes(X_all_np.tolist())
    tenc = time.time() - t1
    print(f"[HE] encrypt nodes: {tenc:.3f}s (Fin={Fin} -> L={ama.L})")

    # 9) encrypted inference
    reset_op_counts()
    t2 = time.time()
    enc_logits_nodes = ama.forward(enc_nodes, neigh)
    tinf = time.time() - t2

    # 10) decrypt
    t3 = time.time()
    he_logits = ama.decrypt_logits_nodes(enc_logits_nodes, sk)  # torch.Tensor [N,5]
    tdec = time.time() - t3

    he_pred = he_logits.argmax(dim=1)

    # metrics on te
    y_true_te = torch.tensor(y_all_np, dtype=torch.long)[idx_te]
    y_pred_te = he_pred[idx_te]
    acc = accuracy_score(y_true_te.numpy(), y_pred_te.numpy())
    f1 = f1_score(y_true_te.numpy(), y_pred_te.numpy(), average="macro")

    print("\n[HE] Encrypted inference time: {:.3f}s".format(tinf))
    print("[HE] Decrypt time:            {:.3f}s".format(tdec))
    print("[HE] Test acc (te only):      {:.4f}".format(acc))
    print("[HE] Test f1  (macro):        {:.4f}".format(f1))

    print_op_counts()

    # ------------------------------------------------------------
    # 11) Compare HE logits vs Plain logits (THIS IS WHAT YOU WANTED)
    # ------------------------------------------------------------
    with torch.no_grad():
        plain_logits = model(X_plain, A_hat).detach().cpu()  # [N,5]

        diff = he_logits - plain_logits
        abs_diff = diff.abs()

        max_abs = float(abs_diff.max().item())
        mean_abs = float(abs_diff.mean().item())
        rmse = float(torch.sqrt((diff * diff).mean()).item())

        print("\n[Compare] HE logits vs Plain logits")
        print(f"[Compare] max |diff| : {max_abs:.6e}")
        print(f"[Compare] mean|diff| : {mean_abs:.6e}")
        print(f"[Compare] RMSE       : {rmse:.6e}")

        plain_pred = plain_logits.argmax(dim=1)
        he_pred2 = he_logits.argmax(dim=1)

        agree_all = float((plain_pred == he_pred2).float().mean().item())
        print(f"[Compare] argmax agreement (all nodes): {agree_all:.4f}")

        idx_te_t = torch.tensor(idx_te, dtype=torch.long)
        agree_te = float((plain_pred[idx_te_t] == he_pred2[idx_te_t]).float().mean().item())
        print(f"[Compare] argmax agreement (te only) : {agree_te:.4f}")

        per_class_mean_abs = abs_diff.mean(dim=0)
        print("[Compare] per-class mean|diff|:", [f"{float(v):.3e}" for v in per_class_mean_abs])

        # Optional: list changed predictions on test set
        if args.show_changed:
            changed = (plain_pred[idx_te_t] != he_pred2[idx_te_t]).nonzero(as_tuple=False).view(-1)
            print(f"\n[Compare] changed test samples: {int(changed.numel())}/{len(idx_te)}")
            for k in changed.tolist():
                idx = int(idx_te[k])
                pp = int(plain_pred[idx].item())
                hp = int(he_pred2[idx].item())
                # show logits for this sample
                pl = plain_logits[idx].tolist()
                hl = he_logits[idx].tolist()
                print(f"  idx={idx:3d} plain={pp} he={hp}")
                print(f"    plain_logits={['{:+.4f}'.format(v) for v in pl]}")
                print(f"    he_logits   ={['{:+.4f}'.format(v) for v in hl]}")

    # Done


if __name__ == "__main__":
    main()