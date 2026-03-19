#!/usr/bin/env python3
"""
Train classifier to evaluate sampling quality over multiple random seeds.
Adds: feature importance bar chart, ROC curve, and baseline comparison.

Usage:
    python classifier_eval.py --data --t-dir T_sweep_cosine_charge/T_500
    python classifier_eval.py --run  --t-dir T_sweep_cosine_charge/T_500
"""
import numpy as np
import random
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "axes.grid": False,
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

BASE = "/work/submit/anton100/msci-project/FCC-BB-GenAI"

parser = argparse.ArgumentParser()
parser.add_argument('--data',   action='store_true', help='Prepare classifier data')
parser.add_argument('--run',    action='store_true', help='Run classifier evaluation')
parser.add_argument('--t-dir',  type=str, required=True,
                    help='Subdirectory under BASE, e.g. T_sweep_cosine_charge/T_500')
parser.add_argument('--n-runs', type=int, default=10)
args = parser.parse_args()

T_DIR      = os.path.join(BASE, args.t_dir)
DATA_PATH  = os.path.join(T_DIR, "classifier_data.npy")
OUTDIR     = os.path.join(T_DIR, "classifier_results")
REAL_PATH  = os.path.join(BASE, "guineapig_raw_trimmed.npy")
GEN_PATH   = os.path.join(T_DIR, "generated_events.npy")

os.makedirs(OUTDIR, exist_ok=True)


# ── Utilities ─────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)

def load_events(path):
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object: return list(arr)
    if arr.ndim == 3 and arr.shape[-1] >= 4: return [arr[i] for i in range(len(arr))]
    raise ValueError(f"Unrecognized format: {path}")

def sanitize_event(ev):
    ev = np.asarray(ev)
    if ev.ndim == 2 and ev.shape[1] >= 8:
        Eabs  = np.abs(ev[:,1].astype(np.float64))
        beta  = ev[:,2:5].astype(np.float64)
        pvec  = Eabs[:,None] * beta
        return Eabs, pvec[:,0], pvec[:,1], pvec[:,2], beta[:,0], beta[:,1], beta[:,2]
    if ev.ndim == 2 and ev.shape[1] >= 7:
        Eabs  = np.abs(ev[:,0].astype(np.float64))
        beta  = ev[:,1:4].astype(np.float64)
        pvec  = Eabs[:,None] * beta
        return Eabs, pvec[:,0], pvec[:,1], pvec[:,2], beta[:,0], beta[:,1], beta[:,2]
    e = np.array([])
    return e, e, e, e, e, e, e

def extract_features(events):
    E_l, px_l, py_l, pz_l, bx_l, by_l, bz_l = [], [], [], [], [], [], []
    for ev in events:
        E, px, py, pz, bx, by, bz = sanitize_event(ev)
        if len(E):
            E_l.append(E); px_l.append(px); py_l.append(py); pz_l.append(pz)
            bx_l.append(bx); by_l.append(by); bz_l.append(bz)
    cat = lambda l: np.concatenate(l) if l else np.array([])
    X = np.column_stack([cat(E_l), cat(px_l), cat(py_l), cat(pz_l),
                         cat(bx_l), cat(by_l), cat(bz_l)])
    return X[np.all(np.isfinite(X), axis=1)]

def train_clf(X_tr, y_tr, X_te, y_te, seed):
    clf = RandomForestClassifier(n_estimators=50, max_depth=5,
                                 random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    fpr, tpr, _ = roc_curve(y_te, proba)
    return auc, clf, fpr, tpr


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_feature_importance(importances_all, feature_names, outpath):
    imp = np.array(importances_all)
    means, stds = imp.mean(0), imp.std(0)
    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar([feature_names[i] for i in order], means[order],
           yerr=stds[order], color="grey", edgecolor="black",
           linewidth=0.8, capsize=4, alpha=0.85)
    ax.set_ylabel("Mean feature importance", fontsize=12)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_title("Random Forest feature importance", fontsize=13)
    fig.savefig(outpath, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved  {outpath}")

def plot_roc(fprs, tprs, auc_scores, baseline_auc, outpath):
    fpr_grid = np.linspace(0, 1, 500)
    tpr_mat  = np.array([np.interp(fpr_grid, f, t) for f, t in zip(fprs, tprs)])
    mean_tpr, std_tpr = tpr_mat.mean(0), tpr_mat.std(0)
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.plot(fpr_grid, mean_tpr, color="steelblue", linewidth=2,
            label=f"Mean ROC (AUC = {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f})")
    ax.fill_between(fpr_grid, mean_tpr-std_tpr, mean_tpr+std_tpr,
                    color="steelblue", alpha=0.2, label="±1 std")
    ax.axhline(baseline_auc, color="grey", linestyle="--", linewidth=1.5,
               label=f"Real vs real baseline ({baseline_auc:.3f})")
    ax.plot([0,1],[0,1],"k:",linewidth=1,label="Random (0.5)")
    ax.set_xlabel("False positive rate",fontsize=12); ax.set_ylabel("True positive rate",fontsize=12)
    ax.set_title("ROC curve: real vs generated",fontsize=13)
    ax.legend(fontsize=10,loc="lower right"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.savefig(outpath, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved  {outpath}")

def plot_auc_distribution(auc_scores, baseline_auc, outpath):
    mean_auc, std_auc = np.mean(auc_scores), np.std(auc_scores)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    jitter = np.random.default_rng(0).uniform(-0.05, 0.05, len(auc_scores))
    ax.scatter(np.ones(len(auc_scores))+jitter, auc_scores,
               color="steelblue", s=60, zorder=3, alpha=0.8)
    ax.errorbar([1],[mean_auc],yerr=[std_auc],fmt="D",color="black",
                markersize=8,capsize=6,linewidth=2,zorder=4,
                label=f"Mean ± std\n{mean_auc:.4f} ± {std_auc:.4f}")
    ax.axhline(baseline_auc,color="grey",linestyle="--",linewidth=1.5,
               label=f"Baseline {baseline_auc:.4f}")
    ax.axhline(0.5,color="lightgrey",linestyle=":",linewidth=1)
    ax.set_xlim(0.7,1.3); ax.set_xticks([])
    ax.set_ylabel("AUC score",fontsize=12)
    ax.set_title("Classifier AUC across runs",fontsize=13)
    ax.legend(fontsize=10)
    fig.savefig(outpath, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved  {outpath}")


# ── Data preparation ──────────────────────────────────────────────────────────
if args.data:
    print(f"Loading events from {T_DIR}...")
    real_events = load_events(REAL_PATH)
    gen_events  = load_events(GEN_PATH)
    print(f"  Real: {len(real_events)}  |  Generated: {len(gen_events)}")

    X_real = extract_features(real_events)
    X_gen  = extract_features(gen_events)

    MAX = 100000
    if len(X_real) > MAX: X_real = X_real[np.random.choice(len(X_real), MAX, replace=False)]
    if len(X_gen)  > MAX: X_gen  = X_gen[np.random.choice(len(X_gen),  MAX, replace=False)]

    X = np.vstack([X_real, X_gen])
    y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_gen))]).astype(np.int32)
    idx = np.random.permutation(len(X)); X = X[idx]; y = y[idx]

    feature_names = ['E', 'px', 'py', 'pz', 'betax', 'betay', 'betaz']
    np.save(DATA_PATH, {"X": X, "y": y, "feature_names": feature_names,
                        "X_real": X_real, "n_real": len(X_real), "n_gen": len(X_gen)})
    print(f"Saved to {DATA_PATH}  ({len(X)} samples)")


# ── Training & evaluation ─────────────────────────────────────────────────────
if args.run:
    print(f"Loading data from {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True).item()
    X, y = data["X"], data["y"]
    X_real = data["X_real"]
    feature_names = data["feature_names"]
    print(f"  {len(X)} samples, {len(feature_names)} features")

    # Baseline: real vs real
    print("\nComputing baseline (real vs real)...")
    half = len(X_real) // 2
    X_base = np.vstack([X_real[:half], X_real[half:2*half]])
    y_base = np.concatenate([np.zeros(half), np.ones(half)]).astype(np.int32)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42, stratify=y_base)
    sc = StandardScaler()
    baseline_auc, _, _, _ = train_clf(sc.fit_transform(Xb_tr), yb_tr,
                                      sc.transform(Xb_te), yb_te, 42)
    print(f"  Baseline AUC: {baseline_auc:.4f}")

    # Main runs
    auc_scores, importances_all, fprs, tprs = [], [], [], []
    for run in range(args.n_runs):
        seed = random.randint(0, 100000)
        set_seed(seed)
        print(f"Run {run+1}/{args.n_runs}  seed={seed}", end="  ")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        sc = StandardScaler()
        auc, clf, fpr, tpr = train_clf(sc.fit_transform(X_tr), y_tr,
                                       sc.transform(X_te), y_te, seed)
        auc_scores.append(auc); importances_all.append(clf.feature_importances_)
        fprs.append(fpr); tprs.append(tpr)
        print(f"AUC={auc:.4f}")

    mean_auc, std_auc = np.mean(auc_scores), np.std(auc_scores)
    print(f"\nFINAL: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Baseline: {baseline_auc:.4f}")

    plot_feature_importance(importances_all, feature_names,
                            f"{OUTDIR}/classifier_feature_importance.png")
    plot_roc(fprs, tprs, auc_scores, baseline_auc,
             f"{OUTDIR}/classifier_roc.png")
    plot_auc_distribution(auc_scores, baseline_auc,
                          f"{OUTDIR}/classifier_auc_distribution.png")
    print("\nDone.")