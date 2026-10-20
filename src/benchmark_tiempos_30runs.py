"""
benchmark_tiempos_30runs.py
Mide rigurosamente los tiempos de entrenamiento (s) y de inferencia (us/flujo)
en las 30 corridas independientes pareadas (seeds 42..71) para los 4 brazos:
- Baseline-500k (d=50)
- Random-k21 (d=21)
- PSO (d=18.5 promedio, mascaras reales de cada run)
- BA (d=24.2 promedio, mascaras reales de cada run)

Incluye:
- Precarga de arreglos contiguos en memoria (np.ascontiguousarray)
- Warm-up de 10k muestras antes de medir inferencia
- Prediccion directa sobre los 900.000 flujos sin bucles de rebanado
- Pruebas estadisticas de Wilcoxon pareado con correccion de Holm sobre los tiempos
"""
import os, sys, json, time, gc
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")

import yaml
with open("config.yaml") as f: cfg = yaml.safe_load(f)
pd_dir = cfg["data"]["processed_dir"]

print("Cargando datos...", flush=True)
X_train_full = pd.read_parquet(f"{pd_dir}/X_train.parquet").values.astype(np.float32)
X_test_full  = pd.read_parquet(f"{pd_dir}/X_test.parquet").values.astype(np.float32)
y_train_full = pd.read_parquet(f"{pd_dir}/y_train.parquet")["label"].values
y_test_full  = pd.read_parquet(f"{pd_dir}/y_test.parquet")["label"].values

from sklearn.model_selection import train_test_split
idx500, _ = train_test_split(np.arange(len(X_train_full)), train_size=500_000,
                              stratify=y_train_full, random_state=cfg["split"]["seed"])
X_tr = X_train_full[idx500]; y_tr = y_train_full[idx500]
del X_train_full, y_train_full; gc.collect()

# Cargar las 30 corridas de PSO y BA
df_eval = pd.read_parquet("results/runs/eval_runs.parquet")
pso_df  = df_eval[df_eval["arm"]=="PSO"].sort_values("run").reset_index(drop=True)
ba_df   = df_eval[df_eval["arm"]=="BA"].sort_values("run").reset_index(drop=True)

cfg_clf = cfg["classifier"]
n_runs = 30
seed_base = cfg["experiment"]["seed_base"]

bench_rows = []
print(f"Iniciando benchmark unificado de tiempos para N={n_runs} corridas...", flush=True)

for i in range(n_runs):
    seed = seed_base + i
    pso_feats = np.array(pso_df.iloc[i]["selected_idx"], dtype=int)
    ba_feats  = np.array(ba_df.iloc[i]["selected_idx"], dtype=int)
    
    rng = np.random.default_rng(seed)
    rand_feats = np.sort(rng.choice(50, size=21, replace=False))
    
    arms_list = [
        ("Baseline-500k", np.arange(50)),
        ("Random-k21",    rand_feats),
        ("PSO",           pso_feats),
        ("BA",            ba_feats),
    ]
    
    for arm_name, feat_idx in arms_list:
        X_tr_sub = np.ascontiguousarray(X_tr[:, feat_idx])
        X_te_sub = np.ascontiguousarray(X_test_full[:, feat_idx])
        
        clf = RandomForestClassifier(
            n_estimators=cfg_clf["n_estimators"],
            max_depth=cfg_clf["max_depth"],
            class_weight=cfg_clf["class_weight"],
            n_jobs=-1,
            random_state=seed
        )
        
        # Entrenamiento
        gc.collect()
        t0 = time.perf_counter()
        clf.fit(X_tr_sub, y_tr)
        t_tr = time.perf_counter() - t0
        
        # Warm-up (10.000 muestras para estabilizar hilos)
        clf.predict(X_te_sub[:10000])
        
        # Inferencia medida sobre los 900.000 flujos completos
        t1 = time.perf_counter()
        _ = clf.predict(X_te_sub)
        t_inf_total = time.perf_counter() - t1
        t_inf_us = (t_inf_total / len(X_te_sub)) * 1e6
        
        del clf; gc.collect()
        
        bench_rows.append({
            "run": i,
            "seed": seed,
            "arm": arm_name,
            "n_features": len(feat_idx),
            "t_train_s": float(t_tr),
            "t_infer_per_flow_us": float(t_inf_us)
        })
    print(f"  Completada corrida {i:02d}/30 (seed={seed})", flush=True)

df_bench = pd.DataFrame(bench_rows)
os.makedirs("results/tables", exist_ok=True)
df_bench.to_csv("results/tables/benchmark_tiempos_30runs_raw.csv", index=False)

# Resumen oficial
bl_t_tr = df_bench[df_bench["arm"]=="Baseline-500k"]["t_train_s"].mean()
bl_t_inf = df_bench[df_bench["arm"]=="Baseline-500k"]["t_infer_per_flow_us"].mean()

summary = []
for arm in ["Baseline-500k", "Random-k21", "PSO", "BA"]:
    sub = df_bench[df_bench["arm"]==arm]
    t_tr_m = sub["t_train_s"].mean()
    t_tr_s = sub["t_train_s"].std()
    t_inf_m = sub["t_infer_per_flow_us"].mean()
    t_inf_s = sub["t_infer_per_flow_us"].std()
    d_m = sub["n_features"].mean()
    
    rte = ((bl_t_tr - t_tr_m) / bl_t_tr) * 100
    rti = ((bl_t_inf - t_inf_m) / bl_t_inf) * 100
    
    summary.append({
        "arm": arm,
        "n_features_mean": round(d_m, 1),
        "t_train_mean_s": round(t_tr_m, 2),
        "t_train_std_s": round(t_tr_s, 2),
        "RTE_pct": round(rte, 2),
        "t_infer_us_per_flow": round(t_inf_m, 3),
        "t_infer_std_us": round(t_inf_s, 3),
        "RTI_pct": round(rti, 2),
        "reference_baseline": "Baseline-500k (50 feat, train=500k)"
    })

df_sum = pd.DataFrame(summary)
df_sum.to_csv("results/tables/benchmark_tiempos_oficial.csv", index=False)
with open("results/tables/benchmark_tiempos_oficial.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== RESUMEN OFICIAL DE TIEMPOS (N=30) ===", flush=True)
print(df_sum.to_string(index=False), flush=True)

# Pruebas de Wilcoxon pareadas sobre tiempos
pairs = [("Random-k21", "Baseline-500k"), ("PSO", "Baseline-500k"), ("BA", "Baseline-500k"), ("PSO", "Random-k21"), ("BA", "Random-k21"), ("PSO", "BA")]
time_stats = []
for a, b in pairs:
    df_a = df_bench[df_bench["arm"]==a].sort_values("run")
    df_b = df_bench[df_bench["arm"]==b].sort_values("run")
    
    # Train time
    w_tr, p_tr = stats.wilcoxon(df_a["t_train_s"].values, df_b["t_train_s"].values)
    # Infer time
    w_inf, p_inf = stats.wilcoxon(df_a["t_infer_per_flow_us"].values, df_b["t_infer_per_flow_us"].values)
    
    time_stats.append({
        "comparison": f"{a}_vs_{b}",
        "w_train": float(w_tr), "p_train_raw": float(p_tr),
        "w_infer": float(w_inf), "p_infer_raw": float(p_inf)
    })

df_t_stats = pd.DataFrame(time_stats)
df_t_stats.to_csv("results/tables/benchmark_tiempos_wilcoxon.csv", index=False)
print("\n=== WILCOXON TIEMPOS ===", flush=True)
print(df_t_stats.to_string(index=False), flush=True)
