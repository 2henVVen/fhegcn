# model.py
import torch
import torch.nn as nn


def adj_matmul(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Dense adjacency multiply (we use dense A_hat)."""
    return adj @ x


class GraphConvolution(nn.Module):
    """
    Single GCN layer:
      Z = A_hat @ (X @ W) + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        out = adj_matmul(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class PolyActivation(nn.Module):
    """
    2nd-order polynomial activation:
      phi(x) = a*x^2 + b*x + c
    HE-friendly.
    """
    def __init__(self, a_init=0.005, b_init=1.0, c_init=0.0, trainable=True):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)), requires_grad=trainable)
        self.b = nn.Parameter(torch.tensor(float(b_init)), requires_grad=trainable)
        self.c = nn.Parameter(torch.tensor(float(c_init)), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * (x * x) + self.b * x + self.c


class PolyGCN(nn.Module):
    """
    Your model:
      h = Poly( GCN(x, A_hat) )
      logits = Linear(h)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 a_init=0.005, b_init=1.0, c_init=0.0, poly_trainable=True):
        super().__init__()
        self.gcn = GraphConvolution(in_dim, hidden_dim, bias=True)
        self.act = PolyActivation(a_init, b_init, c_init, trainable=poly_trainable)
        self.clf = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gcn(x, adj))
        return self.clf(h)