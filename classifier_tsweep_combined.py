#!/usr/bin/env python3
"""
Combined RF + GNN classifier sweep over all T_* subdirs.
Produces a combined AUC-vs-T plot with both classifiers.

Usage:
    python classifier_tsweep_combined.py --data --sweep-dir T_sweep_cosine_charge
    python classifier_tsweep_combined.py --run-rf  --sweep-dir T_sweep_cosine_charge
    python classifier_tsweep_combined.py --run-gnn --sweep-dir T_sweep_cosine_charge
    python classifier_tsweep_combined.py --plot    --sweep-dir T_sweep_cosine_charge
    # Or all at once:
    python classifier_tsweep_combined.py --data --run-rf --run-gnn --plot --sweep-dir T_sweep_cosine_charge
"""
import numpy as np
import random
import argparse
import os
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.utils import to_undirected

matplotlib.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "axes.grid": False,
    "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

BASE = "/work/submit/anton100/msci-project/FCC-BB-GenAI"

parser = argparse.ArgumentParser()
parser.add_argument('--data',       action='store_true', help='Prepare data for all T')
parser.add_argument('--run-rf',     action='store_true', help='Run RF classifier for all T')
parser.add_argument('--run-gnn',    action='store_true', help='Run GNN classifier for all T')
parser.add_argument('--plot',       action='store_true', help='Plot combined results')
parser.add_argument('--sweep-dir',  type=str, required=True)
parser.add_argument('--n-runs-rf',  type=int, default=10)
parser.add_argument('--max-T',      type=int, default=500)
parser.add_argument('--max-events', type=int, default=5000, help='GNN events per class')
parser.add_argument('--epochs',     type=int, default=15,   help='GNN epochs')
parser.add_argument('--hidden',     type=int, default=32,   help='GNN hidden dim')
parser.add_argument('--k',          type=int, default=8,    help='GNN kNN neighbours')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--device',     type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

SWEEP_PATH  = os.path.join(BASE, args.sweep_dir)
REAL_PATH   = os.path.join(BASE, "guineapig_raw_trimmed.npy")
RESULTS_NPY = os.path.join(SWEEP_PATH, "combined_classifier_results.npy")


# ── Discover T dirs ───────────────────────────────────────────────────────────
def discover_T_dirs(sweep_path, max_T):
    entries = []
    for name in sorted(os.listdir(sweep_path)):
        path = os.path.join(sweep_path, name)
        if not os.path.isdir(path) or not name.startswith("T_"):
            continue
        try:
            T_val = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if T_val > max_T:
            continue
        if not os.path.exists(os.path.join(path, "generated_events.npy")):
            print(f"  Skipping {name} — no generated_events.npy")
            continue
        entries.append((T_val, path))
    entries.sort(key=lambda x: x[0])
    return entries


# ── Shared data utils ─────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_events(path):
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object: return list(arr)
    if arr.ndim == 3 and arr.shape[-1] >= 4: return [arr[i] for i in range(len(arr))]
    raise ValueError(f"Unrecognized format: {path}")

def sanitize_event(ev):
    ev = np.asarray(ev)
    if ev.ndim == 2 and ev.shape[1] >= 8:
        Eabs = np.abs(ev[:,1].astype(np.float64))
        beta = ev[:,2:5].astype(np.float64)
        x, y, z = ev[:,5].astype(np.float64), ev[:,6].astype(np.float64), ev[:,7].astype(np.float64)
        return Eabs, beta[:,0], beta[:,1], beta[:,2], x, y, z
    if ev.ndim == 2 and ev.shape[1] >= 7:
        Eabs = np.abs(ev[:,0].astype(np.float64))
        beta = ev[:,1:4].astype(np.float64)
        x, y, z = ev[:,4].astype(np.float64), ev[:,5].astype(np.float64), ev[:,6].astype(np.float64)
        return Eabs, beta[:,0], beta[:,1], beta[:,2], x, y, z
    e = np.array([])
    return e, e, e, e, e, e, e

def extract_particle_features(events):
    """Per-particle features for RF: (E, betax, betay, betaz, x, y, z)."""
    cols = [[] for _ in range(7)]
    for ev in events:
        E, bx, by, bz, x, y, z = sanitize_event(ev)
        if len(E):
            for c, arr in zip(cols, [E, bx, by, bz, x, y, z]):
                c.append(arr)
    cat = lambda l: np.concatenate(l) if l else np.array([])
    X = np.column_stack([cat(c) for c in cols])
    return X[np.all(np.isfinite(X), axis=1)]


# ════════════════════════════════════════════════════════════════
# RF CLASSIFIER
# ════════════════════════════════════════════════════════════════
def prepare_rf_data(T_val, t_path, real_events):
    data_path = os.path.join(t_path, "classifier_data.npy")
    gen_events = load_events(os.path.join(t_path, "generated_events.npy"))
    print(f"  T={T_val}: {len(real_events)} real  |  {len(gen_events)} gen")

    X_real = extract_particle_features(real_events)
    X_gen  = extract_particle_features(gen_events)

    MAX = 100000
    if len(X_real) > MAX: X_real = X_real[np.random.choice(len(X_real), MAX, replace=False)]
    if len(X_gen)  > MAX: X_gen  = X_gen[np.random.choice(len(X_gen),  MAX, replace=False)]

    X = np.vstack([X_real, X_gen])
    y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_gen))]).astype(np.int32)
    idx = np.random.permutation(len(X)); X = X[idx]; y = y[idx]

    np.save(data_path, {"X": X, "y": y,
                        "feature_names": ['E','betax','betay','betaz','x','y','z'],
                        "X_real": X_real, "n_real": len(X_real), "n_gen": len(X_gen)})

def run_rf(T_val, t_path, n_runs):
    data_path = os.path.join(t_path, "classifier_data.npy")
    data   = np.load(data_path, allow_pickle=True).item()
    X, y   = data["X"], data["y"]
    X_real = data["X_real"]

    # Baseline
    half   = len(X_real) // 2
    X_base = np.vstack([X_real[:half], X_real[half:2*half]])
    y_base = np.concatenate([np.zeros(half), np.ones(half)]).astype(np.int32)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42, stratify=y_base)
    sc = StandardScaler()
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    clf.fit(sc.fit_transform(Xb_tr), yb_tr)
    baseline_auc = roc_auc_score(yb_te, clf.predict_proba(sc.transform(Xb_te))[:,1])

    auc_scores, fprs, tprs = [], [], []
    for run in range(n_runs):
        seed = random.randint(0, 100000); set_seed(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        sc = StandardScaler()
        clf = RandomForestClassifier(n_estimators=50, max_depth=5,
                                     random_state=seed, n_jobs=-1)
        clf.fit(sc.fit_transform(X_tr), y_tr)
        proba = clf.predict_proba(sc.transform(X_te))[:,1]
        auc   = roc_auc_score(y_te, proba)
        fpr, tpr, _ = roc_curve(y_te, proba)
        auc_scores.append(auc); fprs.append(fpr); tprs.append(tpr)
        print(f"    RF run {run+1}/{n_runs}  AUC={auc:.4f}")

    print(f"  RF T={T_val}: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    return np.mean(auc_scores), np.std(auc_scores), baseline_auc, fprs, tprs, auc_scores


# ════════════════════════════════════════════════════════════════
# GNN CLASSIFIER
# ════════════════════════════════════════════════════════════════
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
    E, bx, by, bz, x, y, z = sanitize_event(ev_raw)
    if len(E) == 0: return None
    X = np.column_stack([E, bx, by, bz, x, y, z])
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) == 0: return None
    return Data(x=torch.tensor(X, dtype=torch.float),
                edge_index=knn_edges(X[:,1:4], k),  # kNN in beta space
                y=torch.tensor([label], dtype=torch.long))

class GNN(nn.Module):
    def __init__(self, in_ch=7, h=32):
        super().__init__()
        def block(a, b):
            mlp = nn.Sequential(nn.Linear(2*a,b), nn.BatchNorm1d(b), nn.ReLU(),
                                 nn.Linear(b,b),   nn.BatchNorm1d(b), nn.ReLU())
            return EdgeConv(mlp, aggr='mean')
        self.c1 = block(in_ch, h)
        self.c2 = block(h, h*2)
        self.head = nn.Sequential(nn.Linear(h*2, h), nn.ReLU(),
                                   nn.Dropout(0.3),   nn.Linear(h, 2))
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        x = self.c1(x, ei); x = self.c2(x, ei)
        return self.head(global_mean_pool(x, b))

def train_epoch_gnn(model, loader, opt, device):
    model.train(); total = 0.0
    for batch in loader:
        batch = batch.to(device); opt.zero_grad()
        loss = F.cross_entropy(model(batch), batch.y.squeeze())
        loss.backward(); opt.step(); total += loss.item()*batch.num_graphs
    return total/len(loader.dataset)

@torch.no_grad()
def eval_gnn(model, loader, device):
    model.eval(); probs, labs = [], []
    for batch in loader:
        batch = batch.to(device)
        p = F.softmax(model(batch), dim=-1)[:,1].cpu().numpy()
        probs.append(p); labs.append(batch.y.squeeze().cpu().numpy())
    probs = np.concatenate(probs); labs = np.concatenate(labs)
    fpr, tpr, _ = roc_curve(labs, probs)
    return roc_auc_score(labs, probs), fpr, tpr

def run_gnn(T_val, t_path, real_events, device, max_events, epochs, hidden, k,
            batch_size, lr):
    gen_events = load_events(os.path.join(t_path, "generated_events.npy"))
    rng = np.random.default_rng(42)

    real_use = [real_events[i] for i in rng.choice(
        len(real_events), min(max_events, len(real_events)), replace=False)]
    gen_use  = [gen_events[i]  for i in rng.choice(
        len(gen_events),  min(max_events, len(gen_events)),  replace=False)]

    graphs, skip = [], 0
    for ev in real_use:
        g = event_to_graph(ev, 0, k)
        if g: graphs.append(g)
        else: skip += 1
    for ev in gen_use:
        g = event_to_graph(ev, 1, k)
        if g: graphs.append(g)
        else: skip += 1
    print(f"  GNN T={T_val}: {len(graphs)} graphs ({skip} skipped)")

    labels  = np.array([g.y.item() for g in graphs])
    indices = np.arange(len(graphs))
    tr_idx, te_idx = train_test_split(indices, test_size=0.2,
                                      stratify=labels, random_state=42)
    tr = DataLoader([graphs[i] for i in tr_idx], batch_size=batch_size, shuffle=True)
    te = DataLoader([graphs[i] for i in te_idx], batch_size=batch_size, shuffle=False)

    model = GNN(in_ch=7, h=hidden).to(device)
    opt   = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    best_auc, best_state = 0.0, None
    for ep in range(1, epochs+1):
        train_epoch_gnn(model, tr, opt, device); sched.step()
        if ep % 5 == 0 or ep == epochs:
            auc, _, _ = eval_gnn(model, te, device)
            print(f"    GNN epoch {ep}/{epochs}  AUC={auc:.4f}")
            if auc > best_auc:
                best_auc  = auc
                best_state = {k2: v.cpu().clone() for k2,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    auc, fpr, tpr = eval_gnn(model, te, device)
    print(f"  GNN T={T_val}: final AUC={auc:.4f}")
    return auc, fpr, tpr


# ════════════════════════════════════════════════════════════════
# COMBINED PLOT
# ════════════════════════════════════════════════════════════════
def plot_combined(results, outdir):
    """
    results: dict keyed by T_val, each with:
      rf_mean, rf_std, rf_fprs, rf_tprs, rf_auc_scores,
      gnn_auc, gnn_fpr, gnn_tpr, baseline_auc
    """
    T_vals = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(T_vals)))

    rf_means   = [results[T]["rf_mean"]  for T in T_vals]
    rf_stds    = [results[T]["rf_std"]   for T in T_vals]
    gnn_aucs   = [results[T]["gnn_auc"]  for T in T_vals]
    baseline   = results[T_vals[0]]["baseline_auc"]

    # ── AUC vs T ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.errorbar(T_vals, rf_means, yerr=rf_stds, fmt="o-", color="steelblue",
                linewidth=2, markersize=8, capsize=5, label="RF classifier (mean ± std)")
    ax.plot(T_vals, gnn_aucs, "s--", color="darkorange",
            linewidth=2, markersize=8, label="GNN classifier")
    ax.axhline(baseline, color="grey", linestyle=":", linewidth=1.5,
               label=f"Baseline ({baseline:.3f})")
    ax.axhline(0.5, color="lightgrey", linestyle=":", linewidth=1)
    ax.set_xlabel("Diffusion steps T", fontsize=13)
    ax.set_ylabel("Classifier AUC", fontsize=13)
    ax.set_title("Generation quality vs T\n(lower AUC = more realistic)", fontsize=14)
    ax.set_xticks(T_vals)
    ax.legend(fontsize=11)
    ymin = min(min(rf_means), min(gnn_aucs)) - 0.05
    ymax = max(max(rf_means), max(gnn_aucs)) + 0.05
    ax.set_ylim(max(0.45, ymin), min(1.0, ymax))
    fig.savefig(os.path.join(outdir, "auc_vs_T_combined.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {os.path.join(outdir, 'auc_vs_T_combined.png')}")

    # ── ROC curves: RF ─────────────────────────────────────────
    fpr_grid = np.linspace(0, 1, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, mode in zip(axes, ["RF", "GNN"]):
        for T, color in zip(T_vals, colors):
            r = results[T]
            if mode == "RF":
                tpr_mat  = np.array([np.interp(fpr_grid, f, t)
                                     for f, t in zip(r["rf_fprs"], r["rf_tprs"])])
                mean_tpr = tpr_mat.mean(0); std_tpr = tpr_mat.std(0)
                auc_str  = f"{r['rf_mean']:.3f}±{r['rf_std']:.3f}"
                ax.plot(fpr_grid, mean_tpr, color=color, linewidth=2,
                        label=f"T={T}  (AUC={auc_str})")
                ax.fill_between(fpr_grid, mean_tpr-std_tpr, mean_tpr+std_tpr,
                                color=color, alpha=0.08)
            else:
                ax.plot(r["gnn_fpr"], r["gnn_tpr"], color=color, linewidth=2,
                        label=f"T={T}  (AUC={r['gnn_auc']:.3f})")
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=1.5,
                   label=f"Baseline ({baseline:.3f})")
        ax.plot([0,1],[0,1],"k:",linewidth=1)
        ax.set_xlabel("False positive rate", fontsize=12)
        ax.set_ylabel("True positive rate", fontsize=12)
        ax.set_title(f"{mode} classifier — ROC by T", fontsize=13)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.savefig(os.path.join(outdir, "roc_combined.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {os.path.join(outdir, 'roc_combined.png')}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
entries = discover_T_dirs(SWEEP_PATH, args.max_T)
if not entries:
    raise RuntimeError(f"No valid T_* dirs in {SWEEP_PATH}")
print(f"Found: {[f'T={T}' for T,_ in entries]}")

# Load real events once — reused for all T
real_events = None
if args.data or args.run_rf or args.run_gnn:
    print("Loading real events...")
    real_events = load_events(REAL_PATH)
    print(f"  {len(real_events)} real events")

if args.data:
    print("\n── Preparing RF data ──")
    for T_val, t_path in entries:
        prepare_rf_data(T_val, t_path, real_events)

# Load or init results dict
results = {}
if os.path.exists(RESULTS_NPY):
    results = np.load(RESULTS_NPY, allow_pickle=True).item()
    print(f"Loaded existing results for T={sorted(results.keys())}")

if args.run_rf:
    print("\n── Running RF classifiers ──")
    for T_val, t_path in entries:
        mean, std, baseline, fprs, tprs, auc_scores = run_rf(T_val, t_path, args.n_runs_rf)
        if T_val not in results: results[T_val] = {}
        results[T_val].update({
            "rf_mean": mean, "rf_std": std,
            "rf_fprs": fprs, "rf_tprs": tprs,
            "rf_auc_scores": auc_scores,
            "baseline_auc": baseline,
        })
        np.save(RESULTS_NPY, results)

if args.run_gnn:
    print(f"\n── Running GNN classifiers (device={args.device}) ──")
    for T_val, t_path in entries:
        auc, fpr, tpr = run_gnn(T_val, t_path, real_events, args.device,
                                args.max_events, args.epochs, args.hidden,
                                args.k, args.batch_size, args.lr)
        if T_val not in results: results[T_val] = {}
        results[T_val].update({"gnn_auc": auc, "gnn_fpr": fpr, "gnn_tpr": tpr})
        np.save(RESULTS_NPY, results)

if args.plot:
    print("\n── Plotting ──")
    # Check we have both RF and GNN for all T
    ready = {T: r for T, r in results.items()
             if "rf_mean" in r and "gnn_auc" in r and T <= args.max_T}
    if not ready:
        print("No complete results to plot yet.")
    else:
        plot_combined(ready, SWEEP_PATH)

print("\nDone.")