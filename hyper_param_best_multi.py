import optuna

study = optuna.load_study(
    study_name='particle_diff_sinh_zplane_v2',
    storage='sqlite:///tuning_sinh_zplane_v2.db'
)

# Pareto front
pareto = study.best_trials
print(f'=== PARETO FRONT ({len(pareto)} trials) ===')
print(f'{"Trial":>6}  {"val_loss":>12}  {"peak_ks":>12}')
print('-' * 40)
for t in sorted(pareto, key=lambda t: t.values[0]):
    print(f'  #{t.number:3d}  {t.values[0]:12.6f}  {t.values[1]:12.6f}')

# Best on each objective
best_val_t  = min(pareto, key=lambda t: t.values[0])
best_peak_t = min(pareto, key=lambda t: t.values[1])

print(f'\n=== BEST VAL LOSS (trial #{best_val_t.number}) ===')
print(f'  val_loss  = {best_val_t.values[0]:.6f}')
print(f'  peak_ks   = {best_val_t.values[1]:.6f}')
for k, v in best_val_t.params.items():
    print(f'  {k} = {v}')

print(f'\n=== BEST PEAK SCORE (trial #{best_peak_t.number}) ===')
print(f'  val_loss  = {best_peak_t.values[0]:.6f}')
print(f'  peak_ks   = {best_peak_t.values[1]:.6f}')
for k, v in best_peak_t.params.items():
    print(f'  {k} = {v}')

# All completed trials sorted by val_loss
print('\n=== TOP 10 BY VAL LOSS ===')
completed = [t for t in study.trials if t.values is not None]
for t in sorted(completed, key=lambda t: t.values[0])[:10]:
    print(f'  #{t.number:3d}  val={t.values[0]:.6f}  peak={t.values[1]:.6f}  {t.params}')

# Save CSV
df = study.trials_dataframe()
df.to_csv('all_trials.csv', index=False)
print('\nFull results saved to all_trials.csv')