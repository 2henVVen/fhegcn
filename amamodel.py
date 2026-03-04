# amamodel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import torch
from openfhe import SecurityLevel, CCParamsCKKSRNS, PKESchemeFeature, GenCryptoContext


@dataclass
class OpCounts:
    Rot: int = 0
    PMult: int = 0
    CMult: int = 0
    Add: int = 0


op_counts = OpCounts()


def reset_op_counts():
    op_counts.Rot = 0
    op_counts.PMult = 0
    op_counts.CMult = 0
    op_counts.Add = 0


def print_op_counts():
    print("\n=== OpenFHE Homomorphic Op Counts ===")
    print(f"  Rot   : {op_counts.Rot}")
    print(f"  PMult : {op_counts.PMult}")
    print(f"  CMult : {op_counts.CMult}")
    print(f"  Add   : {op_counts.Add}")
    print("======================================\n")


def setup_ckks(
    max_rot_step: int,
    ring_dim: int = 8192,
    mult_depth: int = 15,
    first_mod_size: int = 34,
    scaling_mod_size: int = 40,
):
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    params.SetRingDim(ring_dim)
    params.SetFirstModSize(first_mod_size)
    params.SetScalingModSize(scaling_mod_size)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    kp = cc.KeyGen()
    cc.EvalMultKeyGen(kp.secretKey)

    # rotation keys for power-of-two steps up to max_rot_step (and negatives)
    steps = []
    s = 1
    while s <= max_rot_step:
        steps.append(s)
        steps.append(-s)
        s <<= 1
    steps = sorted(set(steps))
    cc.EvalRotateKeyGen(kp.secretKey, steps, kp.publicKey)

    return cc, kp.publicKey, kp.secretKey


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class AMAModel:
    """
    Patient-node AMA:
      - node = patient
      - one ciphertext per patient packing 100 genes (pad to 128)
      - A aggregation via neighbor weighted-sum (no rotations)
    """

    def __init__(
        self,
        cc,
        pk,
        N: int,
        Fin: int,
        Wgcn: torch.Tensor,
        bgcn: torch.Tensor,
        a: float,
        b: float,
        c: float,
        Wclf: torch.Tensor,
        bclf: torch.Tensor,
        num_class: int = 5,
    ):
        self.cc = cc
        self.pk = pk
        self.N = int(N)
        self.Fin = int(Fin)

        self.Wgcn = Wgcn.detach().cpu()   # [Fin, H]
        self.bgcn = bgcn.detach().cpu()   # [H]
        self.a = float(a); self.b = float(b); self.c = float(c)

        self.Wclf = Wclf.detach().cpu()   # [C, H]
        self.bclf = bclf.detach().cpu()   # [C]

        self.H = int(self.Wgcn.shape[1])
        self.C = int(num_class)
        if self.Wclf.shape[0] != self.C:
            raise ValueError(f"Wclf out_dim={self.Wclf.shape[0]} but num_class fixed to {self.C}")

        self.L = next_pow2(self.Fin)  # 100 -> 128

    def _pt(self, vec: List[float]):
        return self.cc.MakeCKKSPackedPlaintext([float(x) for x in vec])

    def encrypt_nodes(self, X_2d: List[List[float]]):
        if len(X_2d) != self.N:
            raise ValueError(f"X rows {len(X_2d)} != N={self.N}")
        enc_nodes = []
        for i in range(self.N):
            row = X_2d[i]
            if len(row) != self.Fin:
                raise ValueError(f"X row {i} len {len(row)} != Fin={self.Fin}")
            vec = [float(v) for v in row] + [0.0] * (self.L - self.Fin)
            pt = self._pt(vec)
            ct = self.cc.Encrypt(self.pk, pt)
            enc_nodes.append(ct)
        return enc_nodes

    def build_neighbors_from_dense(self, A_hat: torch.Tensor, topk: int = 10, include_self: bool = True):
        A = A_hat.detach().cpu()
        if A.dim() != 2 or A.shape[0] != A.shape[1] or A.shape[0] != self.N:
            raise ValueError("A_hat must be [N,N] with N matching")
        neigh = []
        for i in range(self.N):
            row = A[i].tolist()
            pairs = [(j, float(row[j])) for j in range(self.N)]
            if not include_self:
                pairs = [(j, v) for (j, v) in pairs if j != i]
            pairs.sort(key=lambda t: abs(t[1]), reverse=True)
            pairs = pairs[: min(topk, self.N)]
            if include_self and all(j != i for j, _ in pairs):
                pairs.append((i, float(A[i, i].item())))
            neigh.append(pairs)
        return neigh

    def _sum_slots_pow2(self, ct):
        acc = ct
        step = 1
        while step < self.L:
            rot = self.cc.EvalRotate(acc, step)
            op_counts.Rot += 1
            acc = self.cc.EvalAdd(acc, rot)
            op_counts.Add += 1
            step <<= 1
        return acc

    def _dot_ct_plainvec(self, ct_x, w: List[float]):
        pt_w = self._pt(w)
        prod = self.cc.EvalMult(ct_x, pt_w)
        op_counts.PMult += 1
        self.cc.RescaleInPlace(prod)
        return self._sum_slots_pow2(prod)

    def _poly_scalar(self, ct_scalar):
        cc = self.cc

        x2 = cc.EvalSquare(ct_scalar)
        op_counts.CMult += 1
        cc.RelinearizeInPlace(x2)
        cc.RescaleInPlace(x2)

        ax2 = cc.EvalMult(x2, float(self.a))
        op_counts.PMult += 1
        cc.RescaleInPlace(ax2)

        bx = cc.EvalMult(ct_scalar, float(self.b))
        op_counts.PMult += 1
        cc.RescaleInPlace(bx)

        y = cc.EvalAdd(ax2, bx)
        op_counts.Add += 1

        if self.c != 0.0:
            pt_c = self._pt([self.c] * self.L)
            y = cc.EvalAdd(y, pt_c)
            op_counts.Add += 1
        return y

    def _aggregate_neighbors(self, ct_list, neigh):
        cc = self.cc
        outs = []
        for i in range(self.N):
            acc = None
            for j, aij in neigh[i]:
                tmp = cc.EvalMult(ct_list[j], float(aij))
                op_counts.PMult += 1
                cc.RescaleInPlace(tmp)
                if acc is None:
                    acc = tmp
                else:
                    cc.EvalAddInPlace(acc, tmp)
                    op_counts.Add += 1
            if acc is None:
                raise RuntimeError(f"Empty neighbor list at node {i}")
            outs.append(acc)
        return outs

    def forward(self, enc_nodes, neigh):
        # support_by_h[h][i]
        support_by_h = [[] for _ in range(self.H)]
        for i in range(self.N):
            ct_x = enc_nodes[i]
            for h in range(self.H):
                w = [float(v) for v in self.Wgcn[:, h].tolist()]
                w = w + [0.0] * (self.L - self.Fin)
                ct_s = self._dot_ct_plainvec(ct_x, w)
                support_by_h[h].append(ct_s)

        agg_by_h = []
        for h in range(self.H):
            agg_by_h.append(self._aggregate_neighbors(support_by_h[h], neigh))

        cc = self.cc
        for h in range(self.H):
            b = float(self.bgcn[h].item())
            pt_b = self._pt([b] * self.L) if b != 0.0 else None
            for i in range(self.N):
                ct = agg_by_h[h][i]
                if pt_b is not None:
                    ct = cc.EvalAdd(ct, pt_b)
                    op_counts.Add += 1
                agg_by_h[h][i] = self._poly_scalar(ct)

        enc_logits_nodes = [[None for _ in range(self.C)] for _ in range(self.N)]
        for i in range(self.N):
            for c in range(self.C):
                acc = None
                for h in range(self.H):
                    w = float(self.Wclf[c, h].item())
                    tmp = cc.EvalMult(agg_by_h[h][i], w)
                    op_counts.PMult += 1
                    cc.RescaleInPlace(tmp)
                    if acc is None:
                        acc = tmp
                    else:
                        cc.EvalAddInPlace(acc, tmp)
                        op_counts.Add += 1
                b = float(self.bclf[c].item())
                if b != 0.0:
                    ptb = self._pt([b] * self.L)
                    acc = cc.EvalAdd(acc, ptb)
                    op_counts.Add += 1
                enc_logits_nodes[i][c] = acc
        return enc_logits_nodes

    def decrypt_logits_nodes(self, enc_logits_nodes, sk) -> torch.Tensor:
        out = torch.zeros((self.N, self.C), dtype=torch.float32)
        for i in range(self.N):
            for c in range(self.C):
                pt = self.cc.Decrypt(enc_logits_nodes[i][c], sk)
                out[i, c] = float(pt.GetRealPackedValue()[0])
        return out