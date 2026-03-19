#!/usr/bin/env python3
"""
Event-level GNN classifier to evaluate sampling quality.

Fast mode: 1 full run + 7 feature-ablation runs.
Lower AUC = better generation. AUC near 0.5 = indistinguishable from real.

Usage:
    python gnn_classifier.py --data
    python gnn_classifier.py --run
    python gnn_classifier.py --data --run
"""
from __future__ import annotations
import numpy as np, random, argparse, os
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.utils import to_undirected
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data',       action='store_true')
parser.add_argument('--run',        action='store_true')
parser.add_argument('--data-path',  type=str, default='/work/submit/anton100/msci-project/FCC-BB-GenAI')
parser.add_argument('--gen-subdir', type=str, default='new_37')
parser.add_argument('--k',          type=int, default=8,   help='kNN neighbours')
parser.add_argument('--max-events', type=int, default=5000, help='Events per class')
parser.add_argument('--epochs',     type=int, default=15)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--hidden',     type=int, default=32)
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--device',     type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

BASE        = args.data_path
CACHE       = os.path.join(BASE, args.gen_subdir, 'event_graphs.pt')
OUTDIR      = os.path.join(BASE, args.gen_subdir)
FEATURE_NAMES = ['E', 'betax', 'betay', 'betaz', 'x', 'y', 'z']

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_events(path):
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object: return list(arr)
    if arr.ndim == 3 and arr.shape[-1] >= 4: return [arr[i] for i in range(len(arr))]
    raise ValueError(f"Unrecognised format: {path}")

def sanitize_event(ev):
    ev = np.asarray(ev)
    if ev.ndim == 2 and ev.shape[1] >= 8:
        Eabs = np.abs(ev[:,1]); bx,by,bz = ev[:,2],ev[:,3],ev[:,4]
        x,y,z = ev[:,5],ev[:,6],ev[:,7]
        return Eabs, bx,by,bz, x,y,z
    if ev.ndim == 2 and ev.shape[1] >= 7:
        Eabs = np.abs(ev[:,0]); bx,by,bz = ev[:,1],ev[:,2],ev[:,3]
        x,y,z = ev[:,4],ev[:,5],ev[:,6]
        return Eabs, bx,by,bz, x,y,z
    e = np.array([])
    return e,e,e,e,e,e,e

def knn_edges(pos, k):
    n = len(pos)
    if n <= 1: return torch.zeros((2,0), dtype=torch.long)
    k = min(k, n-1)
    p = torch.tensor(pos, dtype=torch.float)
    d = ((p.unsqueeze(0)-p.unsqueeze(1))**2).sum(-1)
    d.fill_diagonal_(float('inf'))
    _, idx = d.topk(k, largest=False, dim=-1)
    src = torch.arange(n).unsqueeze(1).expand(-1,k).reshape(-1)
    return to_undirected(torch.stack([src, idx.reshape(-1)]))

def event_to_graph(ev_raw, label, k=8):
    E,bx,by,bz,x,y,z = sanitize_event(ev_raw)
    if len(E)==0: return None
    X = np.column_stack([E,bx,by,bz,x,y,z])
    X = X[np.all(np.isfinite(X),axis=1)]
    if len(X)==0: return None
    return Data(x=torch.tensor(X,dtype=torch.float),
                edge_index=knn_edges(X[:,1:4],k),  # kNN in beta space
                y=torch.tensor([label],dtype=torch.long))

# ── Model ─────────────────────────────────────────────────────────────────────
class GNN(nn.Module):
    def __init__(self, in_ch=7, h=32):
        super().__init__()
        def block(a,b):
            mlp = nn.Sequential(nn.Linear(2*a,b), nn.BatchNorm1d(b), nn.ReLU(),
                                 nn.Linear(b,b),   nn.BatchNorm1d(b), nn.ReLU())
            return EdgeConv(mlp, aggr='mean')
        self.c1 = block(in_ch, h)
        self.c2 = block(h,     h*2)
        self.head = nn.Sequential(nn.Linear(h*2,h), nn.ReLU(),
                                   nn.Dropout(0.3),  nn.Linear(h,2))
    def forward(self, data):
        x,ei,b = data.x, data.edge_index, data.batch
        x = self.c1(x,ei); x = self.c2(x,ei)
        return self.head(global_mean_pool(x,b))

# ── Train / eval ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, device):
    model.train(); total=0.0
    for batch in loader:
        batch=batch.to(device); opt.zero_grad()
        loss=F.cross_entropy(model(batch), batch.y.squeeze())
        loss.backward(); opt.step(); total+=loss.item()*batch.num_graphs
    return total/len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval(); probs,labs=[],[]
    for batch in loader:
        batch=batch.to(device)
        p=F.softmax(model(batch),dim=-1)[:,1].cpu().numpy()
        probs.append(p); labs.append(batch.y.squeeze().cpu().numpy())
    probs=np.concatenate(probs); labs=np.concatenate(labs)
    auc=roc_auc_score(labs,probs)
    fpr,tpr,_=roc_curve(labs,probs)
    return auc, fpr, tpr

def run_one(graphs, feature_mask, seed, device, epochs, batch_size, hidden, lr):
    """Train on graphs using only features indicated by feature_mask (bool array len 7)."""
    set_seed(seed)
    labels  = np.array([g.y.item() for g in graphs])
    indices = np.arange(len(graphs))
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=seed)

    # Apply feature mask
    def mask_graph(g):
        return Data(x=g.x[:, feature_mask], edge_index=g.edge_index, y=g.y)

    tr = DataLoader([mask_graph(graphs[i]) for i in tr_idx], batch_size=batch_size, shuffle=True)
    te = DataLoader([mask_graph(graphs[i]) for i in te_idx], batch_size=batch_size, shuffle=False)

    in_ch = int(feature_mask.sum())
    model = GNN(in_ch=in_ch, h=hidden).to(device)
    opt   = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    best_auc, best_state = 0.0, None
    for ep in range(1, epochs+1):
        train_epoch(model, tr, opt, device); sched.step()
        if ep % 5 == 0 or ep == epochs:
            auc,_,_ = eval_model(model, te, device)
            if auc > best_auc:
                best_auc  = auc
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return eval_model(model, te, device)

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_ablation(full_auc, ablation_aucs, names, outpath):
    """Bar chart: full AUC + per-feature-dropped AUC, sorted by drop."""
    drops = {n: full_auc - a for n, a in ablation_aucs.items()}
    order = sorted(drops, key=drops.get, reverse=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    colors = ['steelblue' if drops[n] >= 0 else 'tomato' for n in order]
    bars = ax.bar(order, [drops[n] for n in order], color=colors,
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.axhline(0, color='black', linewidth=1)
    for bar, n in zip(bars, order):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + (0.001 if drops[n]>=0 else -0.003),
                f"{drops[n]:+.4f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("AUC drop when feature removed\n(positive = feature was important)", fontsize=11)
    ax.set_title(f"Feature ablation  (full AUC = {full_auc:.4f})", fontsize=13)
    ax.set_xlabel("Feature removed", fontsize=11)
    fig.savefig(outpath, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved  {outpath}")

def plot_roc(fpr, tpr, auc, outpath):
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.plot(fpr, tpr, color='steelblue', linewidth=2,
            label=f'GNN  (AUC = {auc:.4f})')
    ax.plot([0,1],[0,1],'k:',linewidth=1, label='Random')
    ax.set_xlabel('False positive rate', fontsize=12)
    ax.set_ylabel('True positive rate', fontsize=12)
    ax.set_title('ROC — real vs generated (event level)', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    fig.savefig(outpath, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved  {outpath}")

# ── DATA ──────────────────────────────────────────────────────────────────────
if args.data:
    real_events = load_events(os.path.join(BASE, 'guineapig_raw_trimmed.npy'))
    gen_events  = load_events(os.path.join(BASE, args.gen_subdir, 'generated_events.npy'))

    rng = np.random.default_rng(42)
    if args.max_events and len(real_events) > args.max_events:
        real_events = [real_events[i] for i in rng.choice(len(real_events), args.max_events, replace=False)]
    if args.max_events and len(gen_events) > args.max_events:
        gen_events  = [gen_events[i]  for i in rng.choice(len(gen_events),  args.max_events, replace=False)]

    print(f"Building graphs: {len(real_events)} real + {len(gen_events)} generated")
    graphs, skip = [], 0
    for ev in real_events:
        g = event_to_graph(ev, 0, args.k)
        if g: graphs.append(g)
        else: skip += 1
    for ev in gen_events:
        g = event_to_graph(ev, 1, args.k)
        if g: graphs.append(g)
        else: skip += 1

    print(f"  {len(graphs)} graphs built  ({skip} skipped)")
    torch.save(graphs, CACHE)
    print(f"  Saved to {CACHE}")

# ── RUN ───────────────────────────────────────────────────────────────────────
if args.run:
    device = torch.device(args.device)
    print(f"\nDevice: {device}")
    graphs = torch.load(CACHE, weights_only=False)
    print(f"Loaded {len(graphs)} graphs")

    seed = 42
    all_mask = np.ones(7, dtype=bool)

    # ── Full run ──────────────────────────────────────────────────────────────
    print("\n── Full model (all features) ──")
    full_auc, full_fpr, full_tpr = run_one(
        graphs, all_mask, seed, device, args.epochs, args.batch_size, args.hidden, args.lr)
    print(f"  Full AUC = {full_auc:.4f}")

    plot_roc(full_fpr, full_tpr, full_auc,
             os.path.join(OUTDIR, 'gnn_roc.png'))

    # ── Ablation: drop one feature at a time ──────────────────────────────────
    print("\n── Feature ablation ──")
    ablation_aucs = {}
    for i, fname in enumerate(FEATURE_NAMES):
        mask = all_mask.copy(); mask[i] = False
        auc, _, _ = run_one(graphs, mask, seed, device,
                            args.epochs, args.batch_size, args.hidden, args.lr)
        ablation_aucs[fname] = auc
        drop = full_auc - auc
        print(f"  Drop {fname:8s}:  AUC = {auc:.4f}  (drop = {drop:+.4f})")

    plot_ablation(full_auc, ablation_aucs, FEATURE_NAMES,
                  os.path.join(OUTDIR, 'gnn_feature_ablation.png'))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Full model AUC : {full_auc:.4f}")
    print(f"{'='*50}")
    print("Feature importance (by AUC drop):")
    for n, drop in sorted(ablation_aucs.items(), key=lambda x: full_auc-x[1], reverse=True):
        print(f"  {n:8s}: {full_auc-ablation_aucs[n]:+.4f}")
    print(f"\nNote: AUC~0.5 = indistinguishable, AUC~1.0 = easily separated")