#!/usr/bin/env python3
import os, argparse, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import particle_diffusion_cosine_charge_zplane as pdm
from particle_diffusion_cosine_charge_zplane import (
    CFG, MCPDataset, ParticleDenoiser, DDPM,
    make_linear_gamma_schedule, q_sample_pdg,
    charge_balance_loss, split_indices, set_seed,
    beta_squash_np, beta_unsquash_np,
)
import numpy as np
from scipy.stats import ks_2samp
import optuna
from optuna.samplers import TPESampler

TUNE_EPOCHS = 10
N_WORKERS   = 8

def check_gpu():
    print("\n" + "=" * 60)
    print("GPU CHECK")
    print("=" * 60)
    print(f"CUDA available     : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\n  *** WARNING: No GPU — training will be very slow on CPU! ***")
        print("  Check your SLURM allocation includes --gres=gpu:1")
        print("=" * 60 + "\n")
        return "cpu"
    print(f"Device count       : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}              : {props.name}")
        print(f"  VRAM total       : {props.total_memory / 1024**3:.1f} GB")
        print(f"  VRAM allocated   : {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Compute cap.     : {props.major}.{props.minor}")
    try:
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"  Matmul test      : PASSED  (2048x2048 op completed OK)")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Matmul test      : FAILED — {e}")
        print("  *** GPU visible but not usable — falling back to CPU ***")
        print("=" * 60 + "\n")
        return "cpu"
    print(f"\n  Active device    : {torch.cuda.get_device_name(0)}")
    print("=" * 60 + "\n")
    return "cuda"


# ============================================================
# PEAK METRIC
# ============================================================
def _compute_peak_score(cfg, ddpm, ds_full, val_idx, device, n_sample=200):
    """KS statistic on betax tight zoom (|betax| < 0.007). Lower = better."""
    try:
        from particle_diffusion_cosine_charge_zplane import sample_batch, extract_species
        meta_tmp = {
            "multiplicities": ds_full.multiplicities,
            "feat_mean": ds_full.feat_mean,
            "feat_std":  ds_full.feat_std,
            "feat_dim":  ds_full.feat_dim,
            "me":        cfg.me,
            "n_pdg":     cfg.n_pdg,
            "n_zplane":  cfg.n_zplane,
            "idx_to_pdg": ds_full.idx_to_pdg,
            "max_particles": cfg.max_particles,
            "T": cfg.T,
            "cosine_s": cfg.cosine_s,
        }
        gen_events = sample_batch(meta_tmp, ddpm, device, batch_size=n_sample)

        # real betax from val set — col 1 is u_x, decode via tanh
        real_betax = []
        for idx in val_idx[:min(n_sample, len(val_idx))]:
            ev   = ds_full.events_cont[idx]
            mean = ds_full.feat_mean
            std  = ds_full.feat_std
            cont = ev * std + mean  # denormalise
            u_x  = cont[:, 1]
            bx   = np.tanh(u_x)
            real_betax.append(bx)
        real_betax = np.concatenate(real_betax) if real_betax else np.array([])

        # gen betax — col 2 in output [pdg, E, betax, betay, betaz, x, y, z]
        gen_betax = np.concatenate([ev[:, 2] for ev in gen_events])

        mask_r = np.abs(real_betax) < 0.007
        mask_g = np.abs(gen_betax)  < 0.007
        if mask_r.sum() < 10 or mask_g.sum() < 10:
            return 1.0
        ks_stat, _ = ks_2samp(real_betax[mask_r], gen_betax[mask_g])
        return float(ks_stat)
    except Exception as e:
        print(f"  Peak score failed: {e}")
        return 1.0


def objective(trial, data_path, base_outdir, device):
    lr             = trial.suggest_float("lr",             1e-4, 5e-3,  log=True)
    batch_size     = trial.suggest_categorical("batch_size",     [2, 4, 8, 16, 32])
    d_model        = trial.suggest_categorical("d_model",        [256, 512, 768])
    nhead_opts     = [h for h in [2, 4, 8] if d_model % h == 0]
    nhead          = trial.suggest_categorical("nhead",          nhead_opts)
    num_layers     = trial.suggest_int("num_layers",       2, 6)
    dropout        = trial.suggest_float("dropout",        0.0,  0.3)
    T              = trial.suggest_categorical("T",              [100, 250, 500, 1000])
    cosine_s       = trial.suggest_float("cosine_s",       1e-4, 5e-2, log=True)
    lambda_pdg     = trial.suggest_float("lambda_pdg",     0.05, 2.0)
    lambda_charge  = trial.suggest_float("lambda_charge",  1e-3, 0.5, log=True)
    lambda_zplane  = trial.suggest_float("lambda_zplane",  0.05, 2.0)
    gamma_end      = trial.suggest_float("gamma_end",      0.05, 0.5)
    grad_clip      = trial.suggest_float("grad_clip",      0.5,  5.0)
    gamma_start    = trial.suggest_float("gamma_start",    1e-5, 1e-2, log=True)
    pct_start      = trial.suggest_float("pct_start",      0.05, 0.2)
    div_factor     = trial.suggest_float("div_factor",     10,   100,  log=True)


    cfg = CFG()
    cfg.data_path     = data_path
    cfg.lr            = lr
    cfg.batch_size    = batch_size
    cfg.d_model       = d_model
    cfg.nhead         = nhead
    cfg.num_layers    = num_layers
    cfg.dropout       = dropout
    cfg.T             = T
    cfg.cosine_s      = cosine_s
    cfg.lambda_pdg    = lambda_pdg
    cfg.lambda_charge  = lambda_charge
    cfg.lambda_zplane  = lambda_zplane
    cfg.gamma_end     = gamma_end
    cfg.grad_clip     = grad_clip
    cfg.epochs        = TUNE_EPOCHS
    cfg.gamma_start   = gamma_start
    cfg.pct_start     = pct_start
    cfg.div_factor    = div_factor


    os.makedirs(os.path.join(base_outdir, f"trial_{trial.number}"), exist_ok=True)
    set_seed(cfg.seed)

    try:
        ds_full = MCPDataset(cfg.data_path, max_particles=cfg.max_particles,
                             min_particles=cfg.min_particles, keep_fraction=cfg.keep_fraction)
    except Exception as e:
        print(f"  Trial {trial.number}: dataset error — {e}")
        return float("inf"), float("inf")

    train_idx, val_idx = split_indices(len(ds_full), val_frac=0.1, seed=cfg.seed)
    pin = (device == "cuda")
    dl_train = DataLoader(torch.utils.data.Subset(ds_full, train_idx),
                          batch_size=cfg.batch_size, shuffle=True,
                          num_workers=N_WORKERS, pin_memory=pin, drop_last=False)
    dl_val   = DataLoader(torch.utils.data.Subset(ds_full, val_idx),
                          batch_size=cfg.batch_size, shuffle=False,
                          num_workers=N_WORKERS, pin_memory=pin, drop_last=False)

    model  = ParticleDenoiser(d_model=cfg.d_model, nhead=cfg.nhead,
                               num_layers=cfg.num_layers, dropout=cfg.dropout,
                               n_pdg=cfg.n_pdg, n_zplane=cfg.n_zplane).to(device)
    ddpm   = DDPM(model, cfg.T, device, cosine_s=cfg.cosine_s)
    gammas = make_linear_gamma_schedule(cfg.T, cfg.gamma_start, cfg.gamma_end, device)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(dl_train),
        pct_start=cfg.pct_start,
        anneal_strategy='cos',
        div_factor=cfg.div_factor,
        final_div_factor=1e4,
    )

    best_val = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        total_train, n_train = 0.0, 0
        try:
            for x0, pdg0, zplane0, mask in dl_train:
                x0      = x0.to(device)
                pdg0    = pdg0.to(device)
                zplane0 = zplane0.to(device)
                mask    = mask.to(device)
                B = x0.shape[0]
                t = torch.randint(0, cfg.T, (B,), device=device)
                noise    = torch.randn_like(x0) * mask.unsqueeze(-1)
                x_t      = ddpm.q_sample(x0, t, noise)
                pdg_t    = q_sample_pdg(pdg0,    t, gammas, cfg.n_pdg,    mask)
                zplane_t = q_sample_pdg(zplane0, t, gammas, cfg.n_zplane, mask)
                eps_hat, pdg_logits, zplane_logits = model(x_t, t, pdg_t, zplane_t, mask)
                mse         = (eps_hat - noise).pow(2).sum(dim=-1)
                diff_loss   = (mse * mask).sum() / mask.sum().clamp(min=1)
                pdg_loss    = F.cross_entropy(pdg_logits[mask], pdg0[mask])
                zplane_loss = F.cross_entropy(zplane_logits[mask], zplane0[mask])
                c_loss      = charge_balance_loss(pdg_logits, mask)
                loss        = diff_loss + cfg.lambda_pdg * pdg_loss + cfg.lambda_charge * c_loss + cfg.lambda_zplane * zplane_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                scheduler.step()
                total_train += loss.item(); n_train += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  Trial {trial.number} OOM during train "
                  f"(batch_size={batch_size}, d_model={d_model}, T={T}) — returning worst case.")
            return float("inf"), float("inf")

        train_loss = total_train / max(n_train, 1)

        model.eval()
        total_val, n_val = 0.0, 0
        try:
            with torch.no_grad():
                for x0, pdg0, zplane0, mask in dl_val:
                    x0      = x0.to(device)
                    pdg0    = pdg0.to(device)
                    zplane0 = zplane0.to(device)
                    mask    = mask.to(device)
                    B = x0.shape[0]
                    t = torch.randint(0, cfg.T, (B,), device=device)
                    noise    = torch.randn_like(x0) * mask.unsqueeze(-1)
                    x_t      = ddpm.q_sample(x0, t, noise)
                    pdg_t    = q_sample_pdg(pdg0,    t, gammas, cfg.n_pdg,    mask)
                    zplane_t = q_sample_pdg(zplane0, t, gammas, cfg.n_zplane, mask)
                    eps_hat, pdg_logits, zplane_logits = model(x_t, t, pdg_t, zplane_t, mask)
                    mse         = (eps_hat - noise).pow(2).sum(dim=-1)
                    diff_loss   = (mse * mask).sum() / mask.sum().clamp(min=1)
                    pdg_loss    = F.cross_entropy(pdg_logits[mask], pdg0[mask])
                    zplane_loss = F.cross_entropy(zplane_logits[mask], zplane0[mask])
                    c_loss      = charge_balance_loss(pdg_logits, mask)
                    loss        = diff_loss + cfg.lambda_pdg * pdg_loss + cfg.lambda_charge * c_loss + cfg.lambda_zplane * zplane_loss
                    total_val += loss.item(); n_val += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  Trial {trial.number} OOM during val — returning worst case.")
            return float("inf"), float("inf")

        val_loss = total_val / max(n_val, 1)
        if val_loss < best_val:
            best_val = val_loss

        if device == "cuda":
            mem_alloc    = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved()  / 1024**3
            mem_info = f"| VRAM alloc={mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB"
        else:
            mem_info = "| device=CPU"

        print(f"  Trial {trial.number:3d} | Epoch {epoch+1:2d}/{cfg.epochs} "
              f"| train={train_loss:.5f} | val={val_loss:.5f} {mem_info}")

    peak_score = _compute_peak_score(cfg, ddpm, ds_full, val_idx, device)
    print(f"  Trial {trial.number} | best_val={best_val:.5f} | peak_score={peak_score:.5f}")
    return best_val, peak_score

def print_results(study):
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETE (multi-objective)")
    print("=" * 60)
    pareto = study.best_trials
    print(f"Pareto-optimal trials: {len(pareto)}")
    print("\n{:>6}  {:>12}  {:>12}".format("Trial", "val_loss", "peak_ks"))
    print("-" * 40)
    for t in sorted(pareto, key=lambda t: t.values[0]):
        print(f"  #{t.number:3d}  {t.values[0]:12.6f}  {t.values[1]:12.6f}  {t.params}")
    best_val_t  = min(pareto, key=lambda t: t.values[0])
    best_peak_t = min(pareto, key=lambda t: t.values[1])
    print(f"\nBest val loss  : trial #{best_val_t.number}  val={best_val_t.values[0]:.6f}  peak={best_val_t.values[1]:.6f}")
    print(f"Best peak score: trial #{best_peak_t.number}  val={best_peak_t.values[0]:.6f}  peak={best_peak_t.values[1]:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for particle diffusion")
    parser.add_argument("--data_path",  required=True)
    parser.add_argument("--outdir",     default="tuning_runs")
    parser.add_argument("--n_trials",   type=int, default=30)
    parser.add_argument("--study_name", default="particle_diff")
    parser.add_argument("--storage",    default=None,
                        help="e.g. sqlite:///tuning.db  (enables resume)")
    args = parser.parse_args()

    device = check_gpu()   # <-- runs before anything else

    os.makedirs(args.outdir, exist_ok=True)

    sampler = TPESampler(seed=42)
    pruner  = optuna.pruners.NopPruner()  # pruning not supported with multi-objective

    study = optuna.create_study(
        study_name=args.study_name,
        directions=["minimize", "minimize"],  # (val_loss, peak_ks_score)
        sampler=sampler,
        storage=args.storage, load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args.data_path, args.outdir, device),
        n_trials=args.n_trials, show_progress_bar=True,
    )

    print_results(study)

    best_path = os.path.join(args.outdir, "pareto_front.txt")
    with open(best_path, "w") as f:
        f.write("Pareto-optimal trials (val_loss, peak_ks)\n")
        f.write("=" * 60 + "\n")
        for t in sorted(study.best_trials, key=lambda t: t.values[0]):
            f.write(f"\nTrial #{t.number}  val={t.values[0]:.6f}  peak={t.values[1]:.6f}\n")
            for k, v in t.params.items():
                f.write(f"  {k} = {v}\n")
    print(f"\nPareto front saved to {best_path}")

    try:
        df = study.trials_dataframe()
        csv_path = os.path.join(args.outdir, "all_trials.csv")
        df.to_csv(csv_path, index=False)
        print(f"All trial results saved to {csv_path}")
    except Exception:
        pass

if __name__ == "__main__":
    main()