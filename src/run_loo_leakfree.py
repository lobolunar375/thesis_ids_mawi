"""
run_loo_leakfree.py
Validacion cruzada Leave-One-Capture-Out 100% LIBRE DE FUGA DE INFORMACION (Leak-Free).
En cada pliegue temporal:
- La optimizacion (PSO y BA) se ejecuta EXCLUSIVAMENTE sobre una submuestra de 10k flujos
  del conjunto de ENTRENAMIENTO del pliegue (sin tocar jamas la captura de prueba).
- Los modelos se entrenan sobre 500k flujos del conjunto de ENTRENAMIENTO del pliegue.
- La evaluacion se realiza sobre la captura de prueba completamente no vista.
"""
import os, sys, json, time, gc
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")

import yaml
with open("config.yaml") as f: cfg = yaml.safe_load(f)
pd_dir = cfg["data"]["processed_dir"]

with open(f"{pd_dir}/feature_names.json") as f: feat_names = json.load(f)
n_features = len(feat_names)

def run_pso_opt(X_proxy, y_proxy, seed=42):
    """Ejecuta PSO binario para seleccion de caracteristicas en el proxy del fold."""
    rng = np.random.default_rng(seed)
    pop_size = 10; max_iter = 15
    w = 0.7; c1 = 1.5; c2 = 1.5; vmax = 6.0
    
    # Inicializacion de particulas y velocidades
    X_pos = rng.choice([0, 1], size=(pop_size, n_features)).astype(np.float32)
    # Asegurar que ninguna este vacia
    for p in range(pop_size):
        if X_pos[p].sum() == 0: X_pos[p, rng.choice(n_features)] = 1.0
    V = rng.uniform(-vmax, vmax, size=(pop_size, n_features))
    
    pbest = X_pos.copy()
    pbest_fit = np.full(pop_size, -np.inf)
    gbest = X_pos[0].copy(); gbest_fit = -np.inf
    
    def eval_mask(mask):
        idx = np.where(mask == 1)[0]
        if len(idx) == 0: return 0.0
        clf = RandomForestClassifier(n_estimators=30, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42)
        X_tr_p, X_va_p, y_tr_p, y_va_p = train_test_split(X_proxy[:, idx], y_proxy, test_size=0.3, stratify=y_proxy, random_state=42)
        clf.fit(X_tr_p, y_tr_p)
        preds = clf.predict(X_va_p)
        f1 = float(f1_score(y_va_p, preds, average="macro", zero_division=0))
        rd = (n_features - len(idx)) / n_features
        return 0.95 * f1 + 0.05 * rd
    
    for it in range(max_iter):
        for p in range(pop_size):
            fit = eval_mask(X_pos[p])
            if fit > pbest_fit[p]:
                pbest_fit[p] = fit; pbest[p] = X_pos[p].copy()
            if fit > gbest_fit:
                gbest_fit = fit; gbest = X_pos[p].copy()
        
        # Actualizacion de velocidad y posicion (Sigmoide BPSO)
        r1 = rng.uniform(0, 1, size=(pop_size, n_features))
        r2 = rng.uniform(0, 1, size=(pop_size, n_features))
        V = w * V + c1 * r1 * (pbest - X_pos) + c2 * r2 * (gbest - X_pos)
        V = np.clip(V, -vmax, vmax)
        s_v = 1.0 / (1.0 + np.exp(-V))
        r_trans = rng.uniform(0, 1, size=(pop_size, n_features))
        X_pos = (r_trans < s_v).astype(np.float32)
        for p in range(pop_size):
            if X_pos[p].sum() == 0: X_pos[p, rng.choice(n_features)] = 1.0
            
    best_idx = np.where(gbest == 1)[0]
    return best_idx

def run_ba_opt(X_proxy, y_proxy, seed=42):
    """Ejecuta Bat Algorithm binario para seleccion de caracteristicas en el proxy del fold."""
    rng = np.random.default_rng(seed)
    pop_size = 10; max_iter = 15
    f_min = 0.0; f_max = 2.0; alpha = 0.9; gamma = 0.9
    A = np.ones(pop_size); r = np.full(pop_size, 0.5); r0 = 0.5
    
    X_pos = rng.choice([0, 1], size=(pop_size, n_features)).astype(np.float32)
    for p in range(pop_size):
        if X_pos[p].sum() == 0: X_pos[p, rng.choice(n_features)] = 1.0
    V = np.zeros((pop_size, n_features))
    
    def eval_mask(mask):
        idx = np.where(mask == 1)[0]
        if len(idx) == 0: return 0.0
        clf = RandomForestClassifier(n_estimators=30, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42)
        X_tr_p, X_va_p, y_tr_p, y_va_p = train_test_split(X_proxy[:, idx], y_proxy, test_size=0.3, stratify=y_proxy, random_state=42)
        clf.fit(X_tr_p, y_tr_p)
        preds = clf.predict(X_va_p)
        f1 = float(f1_score(y_va_p, preds, average="macro", zero_division=0))
        rd = (n_features - len(idx)) / n_features
        return 0.95 * f1 + 0.05 * rd
    
    fitnesses = np.array([eval_mask(X_pos[p]) for p in range(pop_size)])
    best_bat = X_pos[np.argmax(fitnesses)].copy()
    best_fit = np.max(fitnesses)
    
    for it in range(max_iter):
        for p in range(pop_size):
            freq = f_min + (f_max - f_min) * rng.uniform(0, 1)
            V[p] = V[p] + (X_pos[p] - best_bat) * freq
            V[p] = np.clip(V[p], -6.0, 6.0)
            s_v = 1.0 / (1.0 + np.exp(-V[p]))
            x_new = (rng.uniform(0, 1, n_features) < s_v).astype(np.float32)
            
            if rng.uniform(0, 1) > r[p]:
                # Vuelo local
                x_new = best_bat.copy()
                rand_dim = rng.choice(n_features, size=3, replace=False)
                x_new[rand_dim] = 1.0 - x_new[rand_dim]
            
            if x_new.sum() == 0: x_new[rng.choice(n_features)] = 1.0
            fit_new = eval_mask(x_new)
            
            if (fit_new > fitnesses[p]) and (rng.uniform(0, 1) < A[p]):
                X_pos[p] = x_new.copy()
                fitnesses[p] = fit_new
                A[p] = A[p] * alpha
                r[p] = r0 * (1.0 - np.exp(-gamma * it))
                
            if fit_new > best_fit:
                best_fit = fit_new
                best_bat = x_new.copy()
                
    best_idx = np.where(best_bat == 1)[0]
    return best_idx

captures = ["20210616", "20210210", "20211013"]
cfg_clf = cfg["classifier"]
loo_results = []

print("=== INICIANDO VALIDACION LOO 100% LEAK-FREE ===", flush=True)

for test_tag in captures:
    print(f"\n--- FOLD LOO (Test en {test_tag}) ---", flush=True)
    X_tr_full = pd.read_parquet(f"{pd_dir}/X_loo_train_{test_tag}.parquet").values.astype(np.float32)
    y_tr_full = pd.read_parquet(f"{pd_dir}/y_loo_train_{test_tag}.parquet")["label"].values
    X_te = pd.read_parquet(f"{pd_dir}/X_loo_test_{test_tag}.parquet").values.astype(np.float32)
    y_te = pd.read_parquet(f"{pd_dir}/y_loo_test_{test_tag}.parquet")["label"].values
    
    # 1. Proxy de 10k exclusivo del train de este pliegue para la busqueda
    idx10k, _ = train_test_split(np.arange(len(X_tr_full)), train_size=10_000, stratify=y_tr_full, random_state=42)
    X_proxy = X_tr_full[idx10k]; y_proxy = y_tr_full[idx10k]
    
    # 2. Submuestra de 500k para entrenar los modelos finales
    if len(X_tr_full) > 500_000:
        idx500, _ = train_test_split(np.arange(len(X_tr_full)), train_size=500_000, stratify=y_tr_full, random_state=42)
        X_tr = X_tr_full[idx500]; y_tr = y_tr_full[idx500]
    else:
        X_tr = X_tr_full; y_tr = y_tr_full
    del X_tr_full, y_tr_full; gc.collect()
    
    print("  Ejecutando busqueda PSO en el train del fold...", flush=True)
    t0 = time.time(); pso_feats = run_pso_opt(X_proxy, y_proxy, seed=42); t_pso_s = time.time()-t0
    print(f"    PSO selecciono {len(pso_feats)} features en {t_pso_s:.1f}s", flush=True)
    
    print("  Ejecutando busqueda BA en el train del fold...", flush=True)
    t0 = time.time(); ba_feats = run_ba_opt(X_proxy, y_proxy, seed=42); t_ba_s = time.time()-t0
    print(f"    BA selecciono {len(ba_feats)} features en {t_ba_s:.1f}s", flush=True)
    
    rng = np.random.default_rng(42)
    rand_feats = np.sort(rng.choice(50, size=21, replace=False))
    
    del X_proxy, y_proxy; gc.collect()
    
    # 3. Clasificador Trivial
    y_pred_triv = np.zeros(len(y_te), dtype=int)
    loo_results.append({
        "fold": test_tag, "arm": "Clasificador-Trivial", "n_features": 0,
        "f1_macro": float(f1_score(y_te, y_pred_triv, average="macro", zero_division=0)),
        "recall_atk": float(recall_score(y_te, y_pred_triv, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, y_pred_triv)), "TFP": 0.0, "TFD": 0.0
    })
    
    fold_arms = [
        ("Baseline", np.arange(50)),
        ("Random-k21", rand_feats),
        ("PSO-Fold", pso_feats),
        ("BA-Fold", ba_feats),
    ]
    
    for arm_name, feat_idx in fold_arms:
        X_tr_sub = np.ascontiguousarray(X_tr[:, feat_idx])
        X_te_sub = np.ascontiguousarray(X_te[:, feat_idx])
        
        clf = RandomForestClassifier(
            n_estimators=cfg_clf["n_estimators"], max_depth=cfg_clf["max_depth"],
            class_weight=cfg_clf["class_weight"], n_jobs=-1, random_state=42
        )
        clf.fit(X_tr_sub, y_tr)
        
        preds = clf.predict(X_te_sub)
        del clf; gc.collect()
        
        cm = confusion_matrix(y_te, preds, labels=[0,1]); tn, fp, fn, tp = cm.ravel()
        tfp = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        tfd = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0
        
        rec = {
            "fold": test_tag,
            "arm": arm_name,
            "n_features": len(feat_idx),
            "f1_macro": float(f1_score(y_te, preds, average="macro", zero_division=0)),
            "recall_atk": float(recall_score(y_te, preds, pos_label=1, zero_division=0)),
            "accuracy": float(accuracy_score(y_te, preds)),
            "TFP": tfp, "TFD": tfd,
            "t_search_s": t_pso_s if "PSO" in arm_name else (t_ba_s if "BA" in arm_name else 0.0)
        }
        loo_results.append(rec)
        print(f"    {arm_name:18s} d={len(feat_idx):02d} | F1={rec['f1_macro']:.4f} Rec={rec['recall_atk']:.4f} Acc={rec['accuracy']:.4f} TFP={rec['TFP']:.4f}", flush=True)

df_leakfree = pd.DataFrame(loo_results)
df_leakfree.to_csv("results/tables/loo_leakfree.csv", index=False)

# Resumen promediado
summary_lf = []
for arm in ["Clasificador-Trivial", "Baseline", "Random-k21", "PSO-Fold", "BA-Fold"]:
    df_arm = df_leakfree[df_leakfree["arm"]==arm]
    summary_lf.append({
        "arm": arm,
        "n_features_mean": round(df_arm["n_features"].mean(), 1),
        "f1_macro_mean": round(df_arm["f1_macro"].mean(), 4),
        "f1_macro_std": round(df_arm["f1_macro"].std(), 4),
        "recall_atk_mean": round(df_arm["recall_atk"].mean(), 4),
        "recall_atk_std": round(df_arm["recall_atk"].std(), 4),
        "accuracy_mean": round(df_arm["accuracy"].mean(), 4),
        "accuracy_std": round(df_arm["accuracy"].std(), 4),
        "TFP_mean": round(df_arm["TFP"].mean(), 4),
        "TFP_std": round(df_arm["TFP"].std(), 4),
        "TFD_mean": round(df_arm["TFD"].mean(), 4),
        "TFD_std": round(df_arm["TFD"].std(), 4),
    })

df_sum_lf = pd.DataFrame(summary_lf)
df_sum_lf.to_csv("results/tables/loo_leakfree_resumen.csv", index=False)
with open("results/tables/loo_leakfree_resumen.json", "w") as f:
    json.dump(summary_lf, f, indent=2)

print("\n=== RESUMEN LOO LEAK-FREE FINAL ===", flush=True)
print(df_sum_lf.to_string(index=False), flush=True)
