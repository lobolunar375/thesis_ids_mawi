"""
FASE 4: Preprocesamiento y particion.
Muestra estratificada por captura (max 3M totales) para ajustarse a la RAM.
Justificacion academica: muestra aleatoria estratificada conserva proporciones de clase.
"""
import pyarrow.parquet as pq
import pandas as pd, numpy as np, os, time, logging, yaml, json
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ID_COLS = {"flow_id","src_ip","dst_ip","src_port","dst_port",
           "start_ts","end_ts","_idx","capture"}

SAMPLE_PER_CAPTURE = {
    "20210616": 1_500_000,
    "20210210": 1_000_000,
    "20211013":   500_000,
}

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def read_and_sample(path, tag, n_sample, seed=42):
    schema    = pq.read_schema(path)
    col_names = schema.names
    keep      = [c for c in col_names if c not in ID_COLS and c != "label"]
    read_cols = keep + ["label"]
    log.info(f"  {tag}: leyendo {len(keep)} features + label")
    df = pd.read_parquet(path, columns=read_cols)
    n_orig = len(df)
    if n_orig > n_sample:
        df, _ = train_test_split(df, train_size=n_sample,
                                 stratify=df["label"], random_state=seed)
        log.info(f"    Muestra: {n_sample:,} de {n_orig:,} ({n_sample/n_orig*100:.1f}%)")
    else:
        log.info(f"    Usando todos: {n_orig:,} filas")
    atk = int((df["label"]==1).sum())
    log.info(f"    Ataque={atk:,} ({atk/len(df)*100:.2f}%)")
    y = df["label"].astype(np.int8).values
    arrays = []
    for col in keep:
        arr = df[col].to_numpy(dtype=np.float32, na_value=0.0)
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        arrays.append(arr)
    X = np.stack(arrays, axis=1)
    del df, arrays
    log.info(f"    X: {X.shape}  RAM={X.nbytes/1e6:.0f} MB")
    return X, y, keep

def run(cfg=None):
    if cfg is None:
        cfg = load_config()
    t0  = time.time()
    out = cfg["data"]["processed_dir"]
    os.makedirs(out, exist_ok=True)
    seed = cfg["split"]["seed"]

    files = {
        "20210616": cfg["data"]["labeled_20210616"],
        "20210210": cfg["data"]["labeled_20210210"],
        "20211013": cfg["data"]["labeled_20211013"],
    }

    X_parts, y_parts, tags_parts, feat_names = [], [], [], None
    for tag, path in files.items():
        log.info(f"Cargando {tag}: {path}")
        n_samp = SAMPLE_PER_CAPTURE.get(tag, 500_000)
        X_p, y_p, feats = read_and_sample(path, tag, n_samp, seed)
        if feat_names is None:
            feat_names = feats
        X_parts.append(X_p)
        y_parts.append(y_p)
        tags_parts.append(np.full(len(y_p), tag))
        del X_p, y_p

    log.info("Concatenando...")
    X    = np.concatenate(X_parts, axis=0)
    y    = np.concatenate(y_parts, axis=0)
    tags = np.concatenate(tags_parts, axis=0)
    del X_parts, y_parts
    log.info(f"Total: {len(X):,} | {X.shape[1]} features | RAM={X.nbytes/1e6:.0f} MB")
    log.info(f"Ataque={int((y==1).sum()):,} ({(y==1).mean()*100:.2f}%)")

    idx_all = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx_all, test_size=cfg["split"]["test_size"],
                                      stratify=y, random_state=seed)
    X_train, X_test = X[idx_tr], X[idx_te]
    y_train, y_test = y[idx_tr], y[idx_te]
    tags_tr, tags_te = tags[idx_tr], tags[idx_te]
    log.info(f"Train={len(X_train):,}  Test={len(X_test):,}")

    n_sub   = min(cfg["split"]["subsample_n"], len(X_train))
    idx_sub, _ = train_test_split(np.arange(len(X_train)), train_size=n_sub,
                                  stratify=y_train, random_state=seed)
    X_sub, y_sub = X_train[idx_sub], y_train[idx_sub]
    log.info(f"Submuestra: {len(X_sub):,} | atk={y_sub.mean()*100:.2f}%")

    log.info("LOO splits...")
    for test_tag in files.keys():
        Xlt = X_train[tags_tr != test_tag]
        ylt = y_train[tags_tr != test_tag]
        Xte = X_test[tags_te == test_tag]
        yte = y_test[tags_te == test_tag]
        if len(Xte) == 0:
            log.warning(f"  LOO {test_tag}: sin muestras en test")
            continue
        log.info(f"  LOO {test_tag}: tr={len(Xlt):,} te={len(Xte):,} atk={yte.mean()*100:.2f}%")
        pd.DataFrame(Xlt, columns=feat_names).to_parquet(f"{out}/X_loo_train_{test_tag}.parquet", index=False)
        pd.DataFrame(Xte, columns=feat_names).to_parquet(f"{out}/X_loo_test_{test_tag}.parquet",  index=False)
        pd.Series(ylt).to_frame("label").to_parquet(f"{out}/y_loo_train_{test_tag}.parquet", index=False)
        pd.Series(yte).to_frame("label").to_parquet(f"{out}/y_loo_test_{test_tag}.parquet",  index=False)

    def sp(arr, cols, path):
        pd.DataFrame(arr, columns=cols).to_parquet(path, compression="snappy", index=False)
        log.info(f"  Guardado: {path} ({os.path.getsize(path)/1e6:.0f} MB)")

    sp(X_train, feat_names, f"{out}/X_train.parquet")
    sp(X_test,  feat_names, f"{out}/X_test.parquet")
    sp(X_sub,   feat_names, f"{out}/X_sub.parquet")
    pd.Series(y_train).to_frame("label").to_parquet(f"{out}/y_train.parquet", index=False)
    pd.Series(y_test).to_frame("label").to_parquet(f"{out}/y_test.parquet",   index=False)
    pd.Series(y_sub).to_frame("label").to_parquet(f"{out}/y_sub.parquet",     index=False)
    with open(f"{out}/feature_names.json","w") as f:
        json.dump(feat_names, f)

    trivial_acc = float((y_test==0).mean())
    stats = {
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "n_features": int(len(feat_names)),
        "pct_attack_train": float((y_train==1).mean()*100),
        "pct_attack_test":  float((y_test==1).mean()*100),
        "trivial_accuracy": trivial_acc,
        "sample_per_capture": SAMPLE_PER_CAPTURE,
    }
    with open(f"{out}/dataset_stats.json","w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Trivial accuracy={trivial_acc:.4f}")
    log.info(f"FASE 4 completa en {time.time()-t0:.0f}s")
    return feat_names, X_train, X_test, y_train, y_test, X_sub, y_sub

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run()
