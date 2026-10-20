"""
run_baseline_500k.py - Re-ejecuta baseline RF con 500k flujos (misma submuestra que F9).
Garantiza comparacion en igualdad de condiciones con PSO/BA.
"""
import os, sys, json, time, gc, logging
import numpy as np, pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/runs/baseline_500k", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler("results/logs/baseline_500k.log","w","utf-8")])
log = logging.getLogger(__name__)
def p(msg): print(msg, flush=True); log.info(msg)

import yaml
with open("config.yaml") as f: cfg = yaml.safe_load(f)
pd_dir = cfg["data"]["processed_dir"]

p("Cargando datos...")
X_train_full = pd.read_parquet(f"{pd_dir}/X_train.parquet").values.astype(np.float32)
X_test       = pd.read_parquet(f"{pd_dir}/X_test.parquet").values.astype(np.float32)
y_train_full = pd.read_parquet(f"{pd_dir}/y_train.parquet")["label"].values
y_test       = pd.read_parquet(f"{pd_dir}/y_test.parquet")["label"].values

# Misma submuestra 500k que se uso en F9
from sklearn.model_selection import train_test_split
idx500, _ = train_test_split(np.arange(len(X_train_full)), train_size=500_000,
                              stratify=y_train_full, random_state=cfg["split"]["seed"])
X_tr = X_train_full[idx500]; y_tr = y_train_full[idx500]
del X_train_full, y_train_full; gc.collect()
p(f"Train={X_tr.shape} Test={X_test.shape} atk_train={y_tr.mean()*100:.1f}%")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

cfg_clf  = cfg["classifier"]
seed_b   = cfg["experiment"]["seed_base"]
n_runs   = cfg["experiment"]["n_runs"]

def run_one(seed):
    out = f"results/runs/baseline_500k/bl500k_run_{seed-seed_b:03d}.json"
    if os.path.exists(out):
        with open(out) as f: return json.load(f), True
    clf = RandomForestClassifier(
        n_estimators=cfg_clf["n_estimators"], max_depth=cfg_clf["max_depth"],
        class_weight=cfg_clf["class_weight"], n_jobs=-1, random_state=seed)
    t0=time.perf_counter(); clf.fit(X_tr, y_tr); t_tr=time.perf_counter()-t0
    t1=time.perf_counter()
    preds=[clf.predict(X_test[s:s+50000]) for s in range(0,len(X_test),50000)]
    y_pred=np.concatenate(preds); t_inf=time.perf_counter()-t1
    del clf; gc.collect()
    cm=confusion_matrix(y_test,y_pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    m={
        "run":seed-seed_b,"seed":seed,
        "f1_macro":   float(f1_score(y_test,y_pred,average="macro",zero_division=0)),
        "recall_atk": float(recall_score(y_test,y_pred,pos_label=1,zero_division=0)),
        "accuracy":   float(accuracy_score(y_test,y_pred)),
        "TFP": float(fp/(fp+tn)) if (fp+tn)>0 else 0.0,
        "TFD": float(fp/(fp+tp)) if (fp+tp)>0 else 0.0,
        "tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn),
        "t_train_s": float(t_tr),
        "t_infer_per_flow_us": float(t_inf/len(y_test)*1e6),
        "n_features": X_tr.shape[1],
    }
    with open(out,"w") as f2: json.dump(m, f2)
    return m, False

p(f"\n=== BASELINE 500k ({n_runs} corridas) ===")
rows=[]; t_total=time.time()
for i in range(n_runs):
    seed=seed_b+i
    m,cached=run_one(seed)
    rows.append(m)
    tag="(cache)" if cached else f"t={m['t_train_s']:.0f}s"
    p(f"  Run {i:02d} F1={m['f1_macro']:.4f} Rec={m['recall_atk']:.4f} "
      f"TFP={m['TFP']:.4f} Acc={m['accuracy']:.4f} {tag}")

df=pd.DataFrame(rows)
df.to_parquet("results/runs/baseline_500k_runs.parquet",index=False)
df.to_csv("results/tables/baseline_500k_runs.csv",index=False)

ref={
    "f1_macro_mean":    float(df["f1_macro"].mean()),
    "f1_macro_std":     float(df["f1_macro"].std()),
    "recall_atk_mean":  float(df["recall_atk"].mean()),
    "recall_atk_std":   float(df["recall_atk"].std()),
    "accuracy_mean":    float(df["accuracy"].mean()),
    "accuracy_std":     float(df["accuracy"].std()),
    "TFP_mean":         float(df["TFP"].mean()),
    "TFP_std":          float(df["TFP"].std()),
    "TFD_mean":         float(df["TFD"].mean()),
    "TFD_std":          float(df["TFD"].std()),
    "t_train_mean":     float(df["t_train_s"].mean()),
    "t_infer_mean":     float(df["t_infer_per_flow_us"].mean()),
    "n_features":       int(X_tr.shape[1]),
}
with open("results/runs/baseline_500k_ref.json","w") as f: json.dump(ref, f, indent=2)

p(f"\n=== RESUMEN BASELINE 500k (n={n_runs}) ===")
p(f"  F1 macro:   {ref['f1_macro_mean']:.4f} +/- {ref['f1_macro_std']:.4f}")
p(f"  Recall atk: {ref['recall_atk_mean']:.4f} +/- {ref['recall_atk_std']:.4f}")
p(f"  Accuracy:   {ref['accuracy_mean']:.4f} +/- {ref['accuracy_std']:.4f}")
p(f"  TFP:        {ref['TFP_mean']:.4f} +/- {ref['TFP_std']:.4f}")
p(f"  TFD:        {ref['TFD_mean']:.4f} +/- {ref['TFD_std']:.4f}")
p(f"  T_train:    {ref['t_train_mean']:.1f}s")
p(f"Completado en {time.time()-t_total:.0f}s")
p("BASELINE 500k COMPLETO.")
