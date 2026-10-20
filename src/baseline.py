"""
FASE 5: Linea base con Random Forest sobre todos los atributos.
30 semillas independientes. Guarda metricas y tiempos.
"""
import pandas as pd, numpy as np, os, time, json, logging, yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

def load_config():
    with open("config.yaml") as f: return yaml.safe_load(f)

def compute_metrics(y_true, y_pred, t_train, t_infer_total, n_test, n_features):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn,fp,fn,tp = cm.ravel()
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    acc    = accuracy_score(y_true, y_pred)
    tfp    = fp/(fp+tn) if (fp+tn)>0 else 0.0   # FPR
    tfd    = fp/(fp+tp) if (fp+tp)>0 else 0.0   # FDR
    t_per  = t_infer_total / n_test if n_test>0 else 0.0
    return {"f1_macro":f1,"recall_atk":recall,"accuracy":acc,
            "TFP":tfp,"TFD":tfd,"tp":int(tp),"tn":int(tn),
            "fp":int(fp),"fn":int(fn),
            "t_train_s":t_train,"t_infer_per_flow_us":t_per*1e6,
            "n_features":n_features}

def run(cfg=None):
    if cfg is None: cfg = load_config()
    pd_dir = cfg["data"]["processed_dir"]
    out    = "results/runs"
    os.makedirs(out, exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    log.info("=== FASE 5: LINEA BASE ===")
    X_train = pd.read_parquet(f"{pd_dir}/X_train.parquet").values.astype(np.float32)
    X_test  = pd.read_parquet(f"{pd_dir}/X_test.parquet").values.astype(np.float32)
    y_train = pd.read_parquet(f"{pd_dir}/y_train.parquet")["label"].values
    y_test  = pd.read_parquet(f"{pd_dir}/y_test.parquet")["label"].values
    with open(f"{pd_dir}/feature_names.json") as f:
        feat_names = json.load(f)

    n_features = X_train.shape[1]
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {n_features}")

    # Clasificador trivial (todo benigno)
    trivial_acc = accuracy_score(y_test, np.zeros_like(y_test))
    log.info(f"Clasificador trivial (todo benigno) accuracy: {trivial_acc:.4f}")

    cfg_clf = cfg["classifier"]
    n_runs  = cfg["experiment"]["n_runs"]
    seed_b  = cfg["experiment"]["seed_base"]

    results = []
    for i in range(n_runs):
        seed_i = seed_b + i
        log.info(f"  Run {i+1}/{n_runs} (seed={seed_i})")
        clf = RandomForestClassifier(
            n_estimators = cfg_clf["n_estimators"],
            max_depth    = cfg_clf["max_depth"],
            class_weight = cfg_clf["class_weight"],
            n_jobs       = cfg_clf["n_jobs"],
            random_state = seed_i
        )
        t_tr0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t_train = time.perf_counter()-t_tr0

        t_inf0 = time.perf_counter()
        y_pred = clf.predict(X_test)
        t_infer = time.perf_counter()-t_inf0

        m = compute_metrics(y_test, y_pred, t_train, t_infer, len(y_test), n_features)
        m["run"] = i; m["seed"] = seed_i; m["arm"] = "baseline"
        results.append(m)
        log.info(f"    F1={m['f1_macro']:.4f} Rec={m['recall_atk']:.4f} Acc={m['accuracy']:.4f} "
                 f"TFP={m['TFP']:.4f} TFD={m['TFD']:.4f} T_tr={m['t_train_s']:.1f}s")

    df_res = pd.DataFrame(results)
    df_res.to_csv("results/tables/baseline_runs.csv", index=False)
    df_res.to_parquet("results/runs/baseline_runs.parquet", index=False)

    # Resumen
    cols = ["f1_macro","recall_atk","accuracy","TFP","TFD","t_train_s","t_infer_per_flow_us"]
    summary = df_res[cols].agg(["mean","std"])
    log.info("\n=== RESUMEN LINEA BASE ===")
    log.info(f"\n{summary.round(4).to_string()}")
    log.info(f"Trivial accuracy: {trivial_acc:.4f}")
    summary.to_csv("results/tables/baseline_summary.csv")

    # Guardar referencia para comparacion
    baseline_ref = {
        "f1_macro_mean":   float(df_res["f1_macro"].mean()),
        "recall_atk_mean": float(df_res["recall_atk"].mean()),
        "accuracy_mean":   float(df_res["accuracy"].mean()),
        "TFP_mean":        float(df_res["TFP"].mean()),
        "TFD_mean":        float(df_res["TFD"].mean()),
        "t_train_mean":    float(df_res["t_train_s"].mean()),
        "t_infer_mean":    float(df_res["t_infer_per_flow_us"].mean()),
        "n_features":      n_features,
        "trivial_accuracy":trivial_acc,
    }
    with open("results/runs/baseline_ref.json","w") as f: json.dump(baseline_ref, f, indent=2)
    log.info("FASE 5 completa.")
    return df_res, baseline_ref

if __name__ == "__main__":
    import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run()
