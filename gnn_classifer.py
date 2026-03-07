"""
Event-level GNN classifier to evaluate sampling quality.

Architecture: EdgeConv (ParticleNet-style) GNN
- Each event = one graph, particles = nodes
- Node features: [E, px, py, pz, betax, betay, betaz]
- Edges: kNN in (px, py, pz) space
- Global mean pooling -> MLP -> binary label (0=real, 1=generated)
- AUC evaluated over 10 random seeds; lower AUC = better generation quality

Usage:
    python gnn_classifier.py --data    # prepare per-event graphs
    python gnn_classifier.py --run     # train & evaluate GNN
    python gnn_classifier.py --data --run   # both steps
"""

from __future__ import annotations

import numpy as np
import random
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.utils import to_undirected

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store_true', help='Build per-event graph dataset')
parser.add_argument('--run',  action='store_true', help='Train & evaluate GNN classifier')
parser.add_argument('--data-path', type=str,
                    default='/work/submit/anton100/msci-project/FCC-BB-GenAI')
parser.add_argument('--gen-subdir', type=str, default='new_10',
                    help='Subdirectory under data-path that holds generated_events.npy')
parser.add_argument('--k-neighbors', type=int, default=8,
                    help='k for kNN edge construction in momentum space')
parser.add_argument('--max-events', type=int, default=None,
                    help='Cap on events per class (None = use all)')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--hidden', type=int, default=64,
                    help='Hidden dimension for EdgeConv MLPs')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n-runs', type=int, default=10,
                    help='Number of independent train/test splits for AUC estimation')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

BASE_PATH   = args.data_path
GEN_SUBDIR  = args.gen_subdir
GRAPH_CACHE = os.path.join(BASE_PATH, GEN_SUBDIR, 'event_graphs.pt')

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading & sanitisation  (unchanged logic from original script)
# ---------------------------------------------------------------------------

def load_events(path: str):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr)
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] >= 4:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Unrecognised format in {path}")


def sanitize_event(ev, me: float = 0.000511):
    ev = np.asarray(ev)

    # Format A: [pdg, E, betax, betay, betaz, x, y, z]
    if ev.ndim == 2 and ev.shape[1] >= 8:
        pdg   = ev[:, 0].astype(np.int64)
        Eabs  = np.abs(ev[:, 1].astype(np.float64))
        betax = ev[:, 2].astype(np.float64)
        betay = ev[:, 3].astype(np.float64)
        betaz = ev[:, 4].astype(np.float64)
        pvec  = Eabs[:, None] * np.stack([betax, betay, betaz], axis=1)
        px, py, pz = pvec[:, 0], pvec[:, 1], pvec[:, 2]
        return pdg, px, py, pz, Eabs, betax, betay, betaz

    # Format B: [E_signed, betax, betay, betaz, x, y, z]
    if ev.ndim == 2 and ev.shape[1] >= 7:
        E_signed = ev[:, 0].astype(np.float64)
        betax    = ev[:, 1].astype(np.float64)
        betay    = ev[:, 2].astype(np.float64)
        betaz    = ev[:, 3].astype(np.float64)
        Eabs     = np.abs(E_signed)
        pdg      = np.where(E_signed >= 0.0, 11, -11).astype(np.int64)
        pvec     = Eabs[:, None] * np.stack([betax, betay, betaz], axis=1)
        px, py, pz = pvec[:, 0], pvec[:, 1], pvec[:, 2]
        return pdg, px, py, pz, Eabs, betax, betay, betaz

    empty = np.array([], dtype=np.float64)
    return (empty.astype(np.int64), empty, empty, empty, empty, empty, empty, empty)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def knn_edges(positions: np.ndarray, k: int):
    """
    Build kNN edges in the given feature space (e.g. momentum 3-vector).
    Returns edge_index of shape [2, n_edges] as a LongTensor.
    Handles events with fewer than k+1 particles gracefully.
    """
    n = len(positions)
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    k_actual = min(k, n - 1)

    # Pairwise squared distances
    pos = torch.tensor(positions, dtype=torch.float)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)          # [n, n, d]
    dist2 = (diff ** 2).sum(dim=-1)                      # [n, n]
    dist2.fill_diagonal_(float('inf'))

    # k nearest neighbours for each node
    _, knn_idx = dist2.topk(k_actual, largest=False, dim=-1)  # [n, k]

    src = torch.arange(n).unsqueeze(1).expand(-1, k_actual).reshape(-1)
    dst = knn_idx.reshape(-1)

    edge_index = to_undirected(torch.stack([src, dst], dim=0))
    return edge_index


def event_to_graph(ev_raw, label: int, k: int = 8) -> Data | None:
    """
    Convert one raw event array to a PyG Data object.

    Node features (7): [E, px, py, pz, betax, betay, betaz]
    Edges: kNN in (px, py, pz) space.
    """
    pdg, px, py, pz, Eabs, betax, betay, betaz = sanitize_event(ev_raw)

    if len(Eabs) == 0:
        return None  # empty event – skip

    node_feats = np.column_stack([Eabs, px, py, pz, betax, betay, betaz])

    # Drop non-finite rows
    mask = np.all(np.isfinite(node_feats), axis=1)
    node_feats = node_feats[mask]

    if len(node_feats) == 0:
        return None

    x          = torch.tensor(node_feats, dtype=torch.float)
    edge_index = knn_edges(node_feats[:, 1:4], k=k)   # kNN in px,py,pz space
    y          = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


# ---------------------------------------------------------------------------
# GNN model  (EdgeConv / ParticleNet-style)
# ---------------------------------------------------------------------------

class EdgeConvBlock(nn.Module):
    """
    One EdgeConv layer: aggregates (x_i || x_j - x_i) for each edge via an MLP.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(2 * in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.conv = EdgeConv(mlp, aggr='mean')

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class EventGNN(nn.Module):
    """
    Three EdgeConv blocks + global mean pooling + MLP head.

    Input : variable-size graph (one per event)
    Output: logits for [real, generated]
    """
    def __init__(self, in_channels: int = 7, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.block1 = EdgeConvBlock(in_channels, hidden)
        self.block2 = EdgeConvBlock(hidden, hidden * 2)
        self.block3 = EdgeConvBlock(hidden * 2, hidden * 4)

        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        x = global_mean_pool(x, batch)   # [batch_size, hidden*4]
        return self.head(x)              # [batch_size, 2]


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss   = F.cross_entropy(logits, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch)
        probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        labels = batch.y.squeeze().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    return auc, all_probs, all_labels


# ---------------------------------------------------------------------------
# DATA PREPARATION  (--data flag)
# ---------------------------------------------------------------------------

if args.data:
    REAL_PATH = os.path.join(BASE_PATH, 'guineapig_raw_trimmed.npy')
    GEN_PATH  = '/work/submit/anton100/msci-project/FCC-BB-GenAI/new_25/generated_events.npy'

    print(f"Real data : {REAL_PATH}")
    print(f"Gen  data : {GEN_PATH}")

    print("\nLoading real events...")
    real_events = load_events(REAL_PATH)
    print(f"  {len(real_events)} real events loaded")

    print("Loading generated events...")
    gen_events = load_events(GEN_PATH)
    print(f"  {len(gen_events)} generated events loaded")

    # Optionally cap the number of events per class
    if args.max_events is not None:
        if len(real_events) > args.max_events:
            idx = np.random.choice(len(real_events), args.max_events, replace=False)
            real_events = [real_events[i] for i in idx]
            print(f"  Real events capped to {args.max_events}")
        if len(gen_events) > args.max_events:
            idx = np.random.choice(len(gen_events), args.max_events, replace=False)
            gen_events = [gen_events[i] for i in idx]
            print(f"  Gen  events capped to {args.max_events}")

    print(f"\nBuilding graphs (k={args.k_neighbors}) ...")
    graphs = []
    skipped = 0

    for ev in real_events:
        g = event_to_graph(ev, label=0, k=args.k_neighbors)
        if g is not None:
            graphs.append(g)
        else:
            skipped += 1

    for ev in gen_events:
        g = event_to_graph(ev, label=1, k=args.k_neighbors)
        if g is not None:
            graphs.append(g)
        else:
            skipped += 1

    n_real_graphs = sum(1 for g in graphs if g.y.item() == 0)
    n_gen_graphs  = sum(1 for g in graphs if g.y.item() == 1)

    print(f"  Real graphs  : {n_real_graphs}")
    print(f"  Gen  graphs  : {n_gen_graphs}")
    print(f"  Skipped (empty/non-finite) : {skipped}")

    print(f"\nSaving graph dataset to: {GRAPH_CACHE}")
    torch.save(graphs, GRAPH_CACHE)
    print(f"✓ Saved {len(graphs)} graphs")

    # Quick multiplicity summary
    mults = [g.num_nodes for g in graphs]
    print(f"\nMultiplicity per event  ->  "
          f"min={np.min(mults)}, median={int(np.median(mults))}, "
          f"mean={np.mean(mults):.1f}, max={np.max(mults)}")


# ---------------------------------------------------------------------------
# TRAINING & EVALUATION  (--run flag)
# ---------------------------------------------------------------------------

if args.run:
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    print(f"Loading graph dataset from: {GRAPH_CACHE}")
    graphs = torch.load(GRAPH_CACHE, weights_only=False)
    print(f"  {len(graphs)} graphs loaded")

    labels = np.array([g.y.item() for g in graphs])
    indices = np.arange(len(graphs))

    print(f"  Real (0): {np.sum(labels==0)}  |  Generated (1): {np.sum(labels==1)}")

    # -----------------------------------------------------------------------
    # Train over n_runs independent seeds
    # -----------------------------------------------------------------------
    auc_scores = []

    for run in range(args.n_runs):
        seed = random.randint(0, 100_000)
        set_seed(seed)

        print(f"\n{'='*60}")
        print(f"Run {run+1}/{args.n_runs}  |  seed={seed}")
        print(f"{'='*60}")

        # Stratified 80/20 split at event level
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=seed
        )

        train_graphs = [graphs[i] for i in train_idx]
        test_graphs  = [graphs[i] for i in test_idx]

        train_loader = DataLoader(train_graphs, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_graphs,  batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

        # Build fresh model for each run
        model = EventGNN(in_channels=7, hidden=args.hidden).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_auc   = 0.0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            scheduler.step()

            if epoch % 5 == 0 or epoch == args.epochs:
                val_auc, _, _ = evaluate(model, test_loader, device)
                marker = " ◀ best" if val_auc > best_auc else ""
                print(f"  Epoch {epoch:3d}/{args.epochs}  "
                      f"loss={train_loss:.4f}  val_AUC={val_auc:.4f}{marker}")
                if val_auc > best_auc:
                    best_auc   = val_auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Final evaluation with best weights
        model.load_state_dict(best_state)
        final_auc, _, _ = evaluate(model, test_loader, device)
        auc_scores.append(final_auc)
        print(f"\n  Run {run+1} best AUC: {final_auc:.4f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    mean_auc = np.mean(auc_scores)
    std_auc  = np.std(auc_scores)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS OVER {args.n_runs} RUNS")
    print(f"{'='*60}")
    print(f"  Mean AUC : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Min  AUC : {np.min(auc_scores):.4f}")
    print(f"  Max  AUC : {np.max(auc_scores):.4f}")
    print(f"\n  All scores: {[f'{a:.4f}' for a in auc_scores]}")
    print(f"\n  Note: AUC ≈ 0.5 → generated events indistinguishable from real")
    print(f"        AUC ≈ 1.0 → classifier easily separates real from generated")