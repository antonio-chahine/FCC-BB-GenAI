import optuna
study = optuna.load_study(
    study_name='particle_diff',
    storage='sqlite:///tuning.db'
)

# Best params
print('=== BEST TRIAL ===')
print(f'Val loss: {study.best_value:.6f}')
for k, v in study.best_params.items():
    print(f'  {k} = {v}')

# Top 5
print()
print('=== TOP 5 TRIALS ===')
trials = sorted(
    [t for t in study.trials if t.value is not None],
    key=lambda t: t.value
)
for t in trials[:5]:
    print(f'  #{t.number}  val={t.value:.6f}  {t.params}')

# Save CSV
df = study.trials_dataframe()
df.to_csv('all_trials.csv', index=False)
print()
print('Full results saved to all_trials.csv')
