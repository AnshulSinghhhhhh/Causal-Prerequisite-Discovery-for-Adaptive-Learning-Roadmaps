"""Module 1 -- stage 2: DirGCN (2-layer directed GCN).

Two aggregation variants are built side by side, not merged away (Section 12):

* ``DirGCN`` -- standard fixed degree-normalized in/out/self aggregation.
* ``AttentionDirGCN`` -- learned per-edge attention weight over in/out
  neighbors instead of fixed degree normalization, letting the model
  down-weight noisy candidate edges rather than trusting every top-k edge
  equally.

Both are *directed*: separate learned in-neighbor, out-neighbor, and self
representations per node, producing a directed link-existence probability
``P(u -> v)``. DirGCN cannot run on an edgeless graph: an empty ``edge_index``
is a broken evaluation, not a zero-shot test, so ``forward`` raises when the
edge index is empty and callers feed a top-k candidate graph from stage 1.

The implementation uses ``torch_geometric`` when available and falls back to a
small, self-contained PyTorch message-passing implementation otherwise, so unit
tests run on CPU with no real model weights.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import config


def _check_edges(edge_index: torch.Tensor, num_nodes: int) -> None:
    if edge_index is None or edge_index.numel() == 0:
        raise ValueError(
            "DirGCN received an empty edge_index: a directed GCN cannot learn "
            "topology from an edgeless graph. Feed it the stage-1 top-k "
            "candidate graph first."
        )
    if int(edge_index.max()) >= int(num_nodes):
        raise ValueError("edge_index references a node outside [0, num_nodes)")


class _DirGCNLayer(nn.Module):
    """One directed GCN layer: separate in/out/self aggregation."""

    def __init__(self, in_dim: int, out_dim: int, attention: bool = False):
        super().__init__()
        self.attention = attention
        self.in_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.out_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.self_lin = nn.Linear(in_dim, out_dim)
        if attention:
            self.in_att = nn.Linear(in_dim, 1)
            self.out_att = nn.Linear(in_dim, 1)

    def _normalized(self, src, msg, num_nodes, att=None):
        """Aggregate messages into node features with degree or attention norm."""
        out = torch.zeros(num_nodes, msg.shape[1], device=msg.device)
        out = out.scatter_add(0, src.unsqueeze(1).expand(-1, msg.shape[1]), msg)
        if att is not None:
            # Attention-weighted, then normalize by per-node attention mass.
            w = torch.sigmoid(att) + 1e-9
            wmsg = msg * w.expand_as(msg)
            num = torch.zeros(num_nodes, msg.shape[1], device=msg.device)
            num = num.scatter_add(
                0, src.unsqueeze(1).expand(-1, msg.shape[1]), wmsg
            )
            denom = torch.zeros(num_nodes, device=msg.device).scatter_add(
                0, src, w.squeeze(-1)
            )
            denom = denom.clamp_min(1e-9).unsqueeze(1)
            return num / denom
        # Fixed degree normalization: divide by in/out degree.
        deg = torch.zeros(num_nodes, device=msg.device).scatter_add(
            0, src, torch.ones(msg.shape[0], device=msg.device)
        )
        deg = deg.clamp_min(1.0).unsqueeze(1)
        return out / deg

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        # In-neighbor aggregation: message from v -> u means u collects v.
        msg_in = self.in_lin(x[src])
        if self.attention:
            att_in = self.in_att(x[src])
            agg_in = self._normalized(dst, msg_in, num_nodes, att=att_in)
        else:
            agg_in = self._normalized(dst, msg_in, num_nodes)
        # Out-neighbor aggregation: message from u -> v means v collects u.
        msg_out = self.out_lin(x[dst])
        if self.attention:
            att_out = self.out_att(x[dst])
            agg_out = self._normalized(src, msg_out, num_nodes, att=att_out)
        else:
            agg_out = self._normalized(src, msg_out, num_nodes)
        self_msg = self.self_lin(x)
        return torch.relu(agg_in + agg_out + self_msg)


class DirGCN(nn.Module):
    """2-layer directed GCN producing directed link scores."""

    def __init__(self, in_dim: int, hidden: Optional[int] = None,
                 dropout: float = 0.0, attention: bool = False) -> None:
        super().__init__()
        hidden = hidden or config.module1.dirgcn_hidden
        self.in_dim = in_dim
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer1 = _DirGCNLayer(in_dim, hidden, attention=attention)
        self.layer2 = _DirGCNLayer(hidden, hidden, attention=attention)
        # Directed link scorer: [h_u || h_v] -> logit.
        self.link = nn.Linear(2 * hidden, 1)

    def _encode(self, x, edge_index) -> torch.Tensor:
        h1 = self.layer1(x, edge_index)
        h1 = self.dropout(h1)
        h2 = self.layer2(h1, edge_index)
        return h2

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        _check_edges(edge_index, x.shape[0])
        return self._encode(x, edge_index)

    def score_pairs(self, x: torch.Tensor, edge_index: torch.Tensor,
                    pairs: torch.Tensor) -> torch.Tensor:
        """``P(u -> v)`` for directed pairs given as an (E, 2) long tensor."""
        h = self.encode(x, edge_index)
        cat = torch.cat([h[pairs[:, 0]], h[pairs[:, 1]]], dim=-1)
        return torch.sigmoid(self.link(cat)).squeeze(-1)

    def forward(self, x, edge_index, pairs):
        """Backward-compatible predict entrypoint returning P in [0, 1]."""
        return self.score_pairs(x, edge_index, pairs)


class AttentionDirGCN(DirGCN):
    """Attention-weighted directed GCN variant (Section 12)."""

    def __init__(self, in_dim: int, hidden: Optional[int] = None,
                 dropout: float = 0.0) -> None:
        super().__init__(in_dim, hidden=hidden, dropout=dropout,
                         attention=True)


def build_edge_index(pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """Convert a list of (u, v) index pairs to a PyG-style (2, E) tensor."""
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    t = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return t


def run_dirgcn(
    x: np.ndarray,
    edge_index: np.ndarray,
    pairs: List[Tuple[int, int]],
    hidden: Optional[int] = None,
    attention: bool = False,
    trained_state: Optional[dict] = None,
) -> np.ndarray:
    """CPU-safe convenience wrapper used by tests and the offline harness.

    ``x`` is (N, d) node features, ``edge_index`` is (2, E) candidate-graph
    edges, ``pairs`` the directed pairs to score. Returns P(u->v) in [0, 1].
    Without ``trained_state`` the model returns untrained/random-link scores,
    which is correct only as a wiring smoke test -- real scores require a
    trained model artifact (see ``scripts/run_full_pipeline.py``).
    """
    x_t = torch.as_tensor(x, dtype=torch.float32)
    ei_t = torch.as_tensor(edge_index, dtype=torch.long)
    pr_t = torch.as_tensor(pairs, dtype=torch.long)
    model = AttentionDirGCN(x.shape[1], hidden=hidden) if attention \
        else DirGCN(x.shape[1], hidden=hidden)
    if trained_state is not None:
        model.load_state_dict(trained_state)
    model.eval()
    with torch.no_grad():
        out = model.score_pairs(x_t, ei_t, pr_t).cpu().numpy()
    return out