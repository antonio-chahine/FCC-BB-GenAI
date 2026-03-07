#!/usr/bin/env python3
import os, argparse, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from particle_diffusion_new_data_discretepdgconditioning import (
    CFG, MCPDataset, ParticleDenoiser, DDPM,
    make_linear_gamma_schedule, q_sample_pdg, split_indices, set_seed,
)
import optuna
from optuna.samplers import TPESampler

TUNE_EPOCHS = 10
N_WORKERS   = 4
PRUNING     = True

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

def objective(trial, data_path, base_outdir, device):
    lr         = trial.suggest_float("lr",         1e-5, 5e-3,  log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
    d_model    = trial.suggest_categorical("d_model",    [32, 64, 128, 256])
    nhead_opts = [h for h in [2, 4, 8] if d_model % h == 0]
    nhead      = trial.suggest_categorical("nhead",      nhead_opts)
    num_layers = trial.suggest_int("num_layers",   1, 4)
    dropout    = trial.suggest_float("dropout",    0.0,  0.3)
    T          = trial.suggest_categorical("T",          [100, 200, 500])
    beta_start = trial.suggest_float("beta_start", 1e-5, 1e-3,  log=True)
    beta_end   = trial.suggest_float("beta_end",   5e-3, 5e-2,  log=True)
    lambda_pdg = trial.suggest_float("lambda_pdg", 0.1,  5.0)
    gamma_end  = trial.suggest_float("gamma_end",  0.05, 0.5)
    grad_clip  = trial.suggest_float("grad_clip",  0.1,  5.0)
    gamma_start = trial.suggest_float("gamma_start", 1e-5, 1e-2, log=True)


    if beta_end <= beta_start:
        raise optuna.exceptions.TrialPruned()

    cfg = CFG()
    cfg.data_path  = data_path
    cfg.lr         = lr
    cfg.batch_size = batch_size
    cfg.d_model    = d_model
    cfg.nhead      = nhead
    cfg.num_layers = num_layers
    cfg.dropout    = dropout
    cfg.T          = T
    cfg.beta_start = beta_start
    cfg.beta_end   = beta_end
    cfg.lambda_pdg = lambda_pdg
    cfg.gamma_end  = gamma_end
    cfg.grad_clip  = grad_clip
    cfg.epochs     = TUNE_EPOCHS
    cfg.gamma_start = gamma_start


    os.makedirs(os.path.join(base_outdir, f"trial_{trial.number}"), exist_ok=True)
    set_seed(cfg.seed)

    try:
        ds_full = MCPDataset(cfg.data_path, max_particles=cfg.max_particles,
                             min_particles=cfg.min_particles, keep_fraction=cfg.keep_fraction)
    except Exception as e:
        print(f"  Trial {trial.number}: dataset error — {e}")
        raise optuna.exceptions.TrialPruned()

    train_idx, val_idx = split_indices(len(ds_full), val_frac=0.1, seed=cfg.seed)
    pin = (device == "cuda")
    dl_train = DataLoader(torch.utils.data.Subset(ds_full, train_idx),
                          batch_size=cfg.batch_size, shuffle=True,
                          num_workers=N_WORKERS, pin_memory=pin, drop_last=False)
    dl_val   = DataLoader(torch.utils.data.Subset(ds_full, val_idx),
                          batch_size=cfg.batch_size, shuffle=False,
                          num_workers=N_WORKERS, pin_memory=pin, drop_last=False)

    model  = ParticleDenoiser(d_model=cfg.d_model, nhead=cfg.nhead,
                               num_layers=cfg.num_layers, dropout=cfg.dropout).to(device)
    ddpm   = DDPM(model, cfg.T, cfg.beta_start, cfg.beta_end, device)
    gammas = make_linear_gamma_schedule(cfg.T, cfg.gamma_start, cfg.gamma_end, device)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        total_train, n_train = 0.0, 0
        for x0, pdg0, mask in dl_train:
            x0, pdg0, mask = x0.to(device), pdg0.to(device), mask.to(device)
            B = x0.shape[0]
            t = torch.randint(0, cfg.T, (B,), device=device)
            noise = torch.randn_like(x0) * mask.unsqueeze(-1)
            x_t   = ddpm.q_sample(x0, t, noise)
            pdg_t = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)
            eps_hat, pdg_logits = model(x_t, t, pdg_t, mask)
            mse       = (eps_hat - noise).pow(2).sum(dim=-1)
            diff_loss = (mse * mask).sum() / mask.sum().clamp(min=1)
            pdg_loss  = F.cross_entropy(pdg_logits[mask], pdg0[mask])
            loss      = diff_loss + cfg.lambda_pdg * pdg_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total_train += loss.item(); n_train += 1
        train_loss = total_train / max(n_train, 1)

        model.eval()
        total_val, n_val = 0.0, 0
        with torch.no_grad():
            for x0, pdg0, mask in dl_val:
                x0, pdg0, mask = x0.to(device), pdg0.to(device), mask.to(device)
                B = x0.shape[0]
                t = torch.randint(0, cfg.T, (B,), device=device)
                noise = torch.randn_like(x0) * mask.unsqueeze(-1)
                x_t   = ddpm.q_sample(x0, t, noise)
                pdg_t = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)
                eps_hat, pdg_logits = model(x_t, t, pdg_t, mask)
                mse       = (eps_hat - noise).pow(2).sum(dim=-1)
                diff_loss = (mse * mask).sum() / mask.sum().clamp(min=1)
                pdg_loss  = F.cross_entropy(pdg_logits[mask], pdg0[mask])
                loss      = diff_loss + cfg.lambda_pdg * pdg_loss
                total_val += loss.item(); n_val += 1
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

        trial.report(val_loss, epoch)
        if PRUNING and trial.should_prune():
            print(f"  Trial {trial.number} pruned at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()

    return best_val

def print_results(study):
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)
    print(f"Best val loss : {study.best_value:.6f}")
    print(f"Best trial #  : {study.best_trial.number}")
    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:20s} = {v}")
    print("\nTop 5 trials:")
    for t in sorted([t for t in study.trials if t.value is not None],
                    key=lambda t: t.value)[:5]:
        print(f"  #{t.number:3d}  val={t.value:.6f}  params={t.params}")

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for particle diffusion")
    parser.add_argument("--data_path",  required=True)
    parser.add_argument("--outdir",     default="tuning_runs")
    parser.add_argument("--n_trials",   type=int, default=30)
    parser.add_argument("--study_name", default="particle_diff")
    parser.add_argument("--storage",    default=None,
                        help="e.g. sqlite:///tuning.db  (enables resume)")
    parser.add_argument("--no_pruning", action="store_true")
    args = parser.parse_args()

    global PRUNING
    if args.no_pruning:
        PRUNING = False

    device = check_gpu()   # <-- runs before anything else

    os.makedirs(args.outdir, exist_ok=True)

    sampler = TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=3, interval_steps=1,
    ) if PRUNING else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name, direction="minimize",
        sampler=sampler, pruner=pruner,
        storage=args.storage, load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args.data_path, args.outdir, device),
        n_trials=args.n_trials, show_progress_bar=True,
    )

    print_results(study)

    best_path = os.path.join(args.outdir, "best_params.txt")
    with open(best_path, "w") as f:
        f.write(f"Best val loss: {study.best_value:.6f}\n")
        f.write(f"Best trial:    #{study.best_trial.number}\n\n")
        for k, v in study.best_params.items():
            f.write(f"{k} = {v}\n")
    print(f"\nBest params saved to {best_path}")

    try:
        df = study.trials_dataframe()
        csv_path = os.path.join(args.outdir, "all_trials.csv")
        df.to_csv(csv_path, index=False)
        print(f"All trial results saved to {csv_path}")
    except Exception:
        pass

if __name__ == "__main__":
    main()