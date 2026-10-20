import pandas as pd, numpy as np

pso_df = pd.read_parquet('results/runs/eval_runs.parquet')
pso_df = pso_df[pso_df['arm'] == 'PSO'].sort_values('run').reset_index(drop=True)

ba_df = pd.read_parquet('results/runs/eval_runs.parquet')
ba_df = ba_df[ba_df['arm'] == 'BA'].sort_values('run').reset_index(drop=True)

bl_df  = pd.read_parquet('results/runs/baseline_500k_runs.parquet').sort_values('run').reset_index(drop=True)
rand_df = pd.read_parquet('results/runs/random_ctrl_runs.parquet').sort_values('run').reset_index(drop=True)

metrics = ['f1_macro', 'recall_atk', 'accuracy', 'TFP', 'TFD']
arms = ['Baseline-500k', 'Random-k21', 'PSO', 'BA']
dfs = [bl_df, rand_df, pso_df, ba_df]

rank_records = []
for m in metrics:
    matrix = np.column_stack([d[m].values[:30] for d in dfs])
    ranks = np.zeros_like(matrix)
    for b in range(30):
        vals = matrix[b]
        if m in ['f1_macro', 'recall_atk', 'accuracy']:
            order = np.argsort(-vals)
        else:
            order = np.argsort(vals)
        rank_assign = np.zeros(4)
        for r_idx, col_idx in enumerate(order):
            rank_assign[col_idx] = r_idx + 1
        ranks[b] = rank_assign
    
    mean_ranks = ranks.mean(axis=0)
    for arm_name, mr in zip(arms, mean_ranks):
        rank_records.append({'metric': m, 'arm': arm_name, 'mean_rank': round(mr, 2)})

df_ranks = pd.DataFrame(rank_records)
df_ranks.to_csv('results/tables/friedman_rangos_medios.csv', index=False)
pivot = df_ranks.pivot(index='arm', columns='metric', values='mean_rank')
print('=== RANGOS MEDIOS DE FRIEDMAN (1 = Mejor) ===')
print(pivot[['f1_macro', 'recall_atk', 'accuracy', 'TFP', 'TFD']])
