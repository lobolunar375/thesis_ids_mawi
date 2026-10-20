"""
run_all.py - Pipeline completo Fases 4 a 12.
Ejecutar desde thesis_ids/ con: python src/run_all.py
"""
import os, sys, json, time, logging, yaml
import numpy as np, pandas as pd

# Asegurar que estamos en thesis_ids/
if not os.path.exists("config.yaml"):
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, "src")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/logs/run_all.log", mode="a", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

def load_config():
    with open("config.yaml") as f: return yaml.safe_load(f)

# ============================================================
#  FASE 4: PREPROCESAMIENTO
# ============================================================
def fase4(cfg):
    log.info("\n" + "="*60 + "\nFASE 4: PREPROCESAMIENTO\n" + "="*60)
    pd_dir = cfg["data"]["processed_dir"]
    # Saltar si ya existe la particion procesada
    if os.path.exists(f"{pd_dir}/X_train.parquet") and os.path.exists(f"{pd_dir}/feature_names.json"):
        log.info("  Datos procesados ya existen - saltando Fase 4")
        import json
        with open(f"{pd_dir}/feature_names.json") as f:
            feat = json.load(f)
        log.info(f"  Features: {len(feat)}")
        return feat
    from preprocesamiento import run as preprocess
    feat, Xtr, Xte, ytr, yte, Xs, ys = preprocess(cfg)
    log.info(f"Features ML: {len(feat)}")
    return feat

# ============================================================
#  FASE 5: LINEA BASE
# ============================================================
def fase5(cfg):
    log.info("\n" + "="*60 + "\nFASE 5: LINEA BASE\n" + "="*60)
    if os.path.exists("results/runs/baseline_ref.json"):
        log.info("  Ya existe baseline_ref.json - saltando")
        with open("results/runs/baseline_ref.json") as f:
            return pd.read_parquet("results/runs/baseline_runs.parquet"), json.load(f)
    from baseline import run as base_run
    return base_run(cfg)

# ============================================================
#  FASE 7: CALIBRACION
# ============================================================
def fase7(cfg):
    log.info("\n" + "="*60 + "\nFASE 7: CALIBRACION DEL COSTO\n" + "="*60)
    pd_dir = cfg["data"]["processed_dir"]
    X_sub = pd.read_parquet(f"{pd_dir}/X_sub.parquet").values.astype(np.float32)
    y_sub = pd.read_parquet(f"{pd_dir}/y_sub.parquet")["label"].values
    cfg_clf = cfg["classifier"]
    clf_params = {
        "n_estimators": cfg_clf["n_estimators"],
        "max_depth":    cfg_clf["max_depth"],
        "class_weight": cfg_clf["class_weight"],
        "n_jobs":       1,
        "random_state": cfg_clf["seed"],
    }
    from wrapper import fitness
    # Medir 3 evaluaciones de fitness
    log.info("  Midiendo tiempo de una evaluacion de fitness (3 muestras)...")
    times = []
    mask_half = np.ones(X_sub.shape[1])
    mask_half[::2] = 0
    for i in range(3):
        t0 = time.time()
        fv, f1cv, nsel = fitness(mask_half, X_sub, y_sub, clf_params,
                                  cfg["fitness"]["alpha"], cfg["fitness"]["cv_folds"])
        elapsed = time.time()-t0
        times.append(elapsed)
        log.info(f"    Eval {i+1}: {elapsed:.2f}s | fit={fv:.4f} f1cv={f1cv:.4f} nsel={nsel}")

    t_eval = float(np.median(times))
    pop    = cfg["pso"]["population"]
    iters  = cfg["pso"]["iterations"]
    n_runs = cfg["experiment"]["n_runs"]
    n_algos= 2  # PSO + BA

    t_total_s = t_eval * pop * iters * n_runs * n_algos
    t_total_h = t_total_s / 3600

    log.info(f"\n  === PLAN DE CALIBRACION ===")
    log.info(f"  T por eval    : {t_eval:.2f}s")
    log.info(f"  Poblacion     : {pop}")
    log.info(f"  Iteraciones   : {iters}")
    log.info(f"  Corridas      : {n_runs} x 2 algoritmos")
    log.info(f"  T total est.  : {t_total_h:.1f} horas")

    calib = {
        "t_eval_s": t_eval, "population": pop, "iterations": iters,
        "n_runs": n_runs, "n_algos": n_algos, "t_total_est_h": t_total_h,
        "subsample_n": len(X_sub), "n_features": X_sub.shape[1],
    }
    with open("results/runs/calibration.json","w") as f: json.dump(calib, f, indent=2)
    log.info("\n*** PUNTO DE PARADA 3: Revisa el plan y confirma ***")
    return calib

# ============================================================
#  FASE 8: CORRIDAS PSO + BA
# ============================================================
def fase8(cfg):
    log.info("\n" + "="*60 + "\nFASE 8: CORRIDAS PSO + BA\n" + "="*60)
    pd_dir = cfg["data"]["processed_dir"]
    X_sub = pd.read_parquet(f"{pd_dir}/X_sub.parquet").values.astype(np.float32)
    y_sub = pd.read_parquet(f"{pd_dir}/y_sub.parquet")["label"].values
    cfg_clf = cfg["classifier"]
    cfg_fit = cfg["fitness"]
    clf_params = {
        "n_estimators":        cfg_clf["n_estimators"],
        "n_estimators_search": cfg_fit.get("n_estimators_search", 30),
        "max_depth":           cfg_clf["max_depth"],
        "class_weight":        cfg_clf["class_weight"],
        "n_jobs":              1,
        "random_state":        cfg_clf["seed"],
    }
    # Regenerar submuestra con el nuevo subsample_n del config
    n_sub_cfg = cfg["split"]["subsample_n"]
    if len(X_sub) != n_sub_cfg:
        log.info(f"  Regenerando submuestra: {len(X_sub)} -> {n_sub_cfg}")
        X_train_full = pd.read_parquet(f"{pd_dir}/X_train.parquet").values.astype(np.float32)
        y_train_full = pd.read_parquet(f"{pd_dir}/y_train.parquet")["label"].values
        from sklearn.model_selection import train_test_split as tts
        idx_s, _ = tts(np.arange(len(X_train_full)), train_size=n_sub_cfg,
                        stratify=y_train_full, random_state=cfg["split"]["seed"])
        X_sub = X_train_full[idx_s]; y_sub = y_train_full[idx_s]
        pd.DataFrame(X_sub).to_parquet(f"{pd_dir}/X_sub.parquet", index=False)
        pd.Series(y_sub).to_frame("label").to_parquet(f"{pd_dir}/y_sub.parquet", index=False)
        log.info(f"  Submuestra regenerada: {len(X_sub):,} | atk={y_sub.mean()*100:.1f}%")
        del X_train_full, y_train_full
    out = "results/runs"
    n_runs  = cfg["experiment"]["n_runs"]
    seed_b  = cfg["experiment"]["seed_base"]

    from pso import binary_pso
    from ba  import binary_bat

    pso_results, ba_results = [], []
    for i in range(n_runs):
        seed_i = seed_b + i
        # PSO
        p_file = f"{out}/pso_run_{i:03d}.json"
        if os.path.exists(p_file):
            with open(p_file) as f: pso_results.append(json.load(f))
            log.info(f"  PSO run {i} ya existe - saltando")
        else:
            r = binary_pso(X_sub, y_sub, clf_params, cfg["pso"], cfg["fitness"], i, out, seed=seed_i)
            pso_results.append(r)
        # BA
        b_file = f"{out}/ba_run_{i:03d}.json"
        if os.path.exists(b_file):
            with open(b_file) as f: ba_results.append(json.load(f))
            log.info(f"  BA  run {i} ya existe - saltando")
        else:
            r = binary_bat(X_sub, y_sub, clf_params, cfg["bat"], cfg["fitness"], i, out, seed=seed_i)
            ba_results.append(r)

    log.info("FASE 8 completa.")
    return pso_results, ba_results

# ============================================================
#  FASE 9: EVALUACION FINAL
# ============================================================
def fase9(cfg, pso_results, ba_results, baseline_ref):
    log.info("\n" + "="*60 + "\nFASE 9: EVALUACION FINAL\n" + "="*60)
    pd_dir = cfg["data"]["processed_dir"]
    X_train = pd.read_parquet(f"{pd_dir}/X_train.parquet").values.astype(np.float32)
    X_test  = pd.read_parquet(f"{pd_dir}/X_test.parquet").values.astype(np.float32)
    y_train = pd.read_parquet(f"{pd_dir}/y_train.parquet")["label"].values
    y_test  = pd.read_parquet(f"{pd_dir}/y_test.parquet")["label"].values

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix
    from baseline import compute_metrics
    cfg_clf = cfg["classifier"]
    seed_b  = cfg["experiment"]["seed_base"]
    N_total = X_train.shape[1]

    all_eval = []
    for algo, runs in [("PSO", pso_results), ("BA", ba_results)]:
        for res in runs:
            i    = res["run_id"]
            seed = seed_b + i
            mask = np.array(res["best_mask"])
            idx  = np.where(mask >= 0.5)[0]
            if len(idx) == 0: idx = np.arange(N_total)
            X_tr_sel = X_train[:, idx]
            X_te_sel = X_test[:, idx]
            clf = RandomForestClassifier(
                n_estimators = cfg_clf["n_estimators"],
                max_depth    = cfg_clf["max_depth"],
                class_weight = cfg_clf["class_weight"],
                n_jobs       = 1,   # n_jobs=1 evita fragmentacion de RAM en prediccion
                random_state = seed,
            )
            t_tr0 = time.perf_counter()
            clf.fit(X_tr_sel, y_train)
            t_train = time.perf_counter()-t_tr0
            # Prediccion por lotes para evitar asignacion contigua grande
            t_inf0 = time.perf_counter()
            batch = 50000
            y_parts_pred = []
            for start in range(0, len(X_te_sel), batch):
                y_parts_pred.append(clf.predict(X_te_sel[start:start+batch]))
            y_pred  = np.concatenate(y_parts_pred)
            t_infer = time.perf_counter()-t_inf0
            m = compute_metrics(y_test, y_pred, t_train, t_infer, len(y_test), len(idx))
            # Reducciones
            n_sel = len(idx)
            m["RD"]  = (N_total-n_sel)/N_total*100
            m["RTE"] = (baseline_ref["t_train_mean"]-m["t_train_s"])/baseline_ref["t_train_mean"]*100 if baseline_ref["t_train_mean"]>0 else 0
            m["RTI"] = (baseline_ref["t_infer_mean"]-m["t_infer_per_flow_us"])/baseline_ref["t_infer_mean"]*100 if baseline_ref["t_infer_mean"]>0 else 0
            m["delta_f1"]   = (m["f1_macro"]-baseline_ref["f1_macro_mean"])/baseline_ref["f1_macro_mean"]*100
            m["delta_rec"]  = (m["recall_atk"]-baseline_ref["recall_atk_mean"])/baseline_ref["recall_atk_mean"]*100
            m["delta_acc"]  = (m["accuracy"]-baseline_ref["accuracy_mean"])/baseline_ref["accuracy_mean"]*100
            m["delta_TFP"]  = (m["TFP"]-baseline_ref["TFP_mean"])/baseline_ref["TFP_mean"]*100 if baseline_ref["TFP_mean"]>0 else 0
            m["run"] = i; m["seed"] = seed; m["arm"] = algo
            m["best_fitness"] = res["best_fitness"]
            m["t_search_s"]   = res["t_search_s"]
            m["selected_idx"] = idx.tolist()
            all_eval.append(m)
            log.info(f"  {algo} run={i} n_sel={n_sel} F1={m['f1_macro']:.4f} RD={m['RD']:.1f}% delta_F1={m['delta_f1']:+.2f}%")

    df_eval = pd.DataFrame(all_eval)
    df_eval.to_csv("results/tables/eval_runs.csv", index=False)
    df_eval.to_parquet("results/runs/eval_runs.parquet", index=False)
    log.info("FASE 9 completa.")
    return df_eval

# ============================================================
#  FASE 11: ANALISIS ESTADISTICO
# ============================================================
def fase11(cfg, df_baseline, df_eval):
    log.info("\n" + "="*60 + "\nFASE 11: ANALISIS ESTADISTICO\n" + "="*60)
    from scipy import stats
    from itertools import combinations
    import json

    metrics = ["f1_macro","recall_atk","accuracy","TFP","TFD"]
    pso_df  = df_eval[df_eval["arm"]=="PSO"].sort_values("run").reset_index(drop=True)
    ba_df   = df_eval[df_eval["arm"]=="BA"].sort_values("run").reset_index(drop=True)
    bl_df   = df_baseline.sort_values("run").reset_index(drop=True)

    stat_results = []
    for m in metrics:
        for (a_name, a_vals), (b_name, b_vals) in [
            ("PSO","baseline"), ("BA","baseline"), ("PSO","BA")
        ]:
            if a_name=="PSO":   av = pso_df[m].values
            elif a_name=="BA":  av = ba_df[m].values
            if b_name=="baseline": bv = bl_df[m].values
            elif b_name=="BA": bv = ba_df[m].values

            stat, p = stats.wilcoxon(av, bv, alternative="two-sided", zero_method="wilcox") if len(av)==len(bv) else (np.nan, np.nan)
            n = len(av)
            r_effect = stat / (n*(n+1)/2) if n>0 else np.nan
            stat_results.append({
                "metric": m, "comparison": f"{a_name}_vs_{b_name}",
                "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
                "p_value": float(p) if not np.isnan(p) else None,
                "significant": bool(p < 0.05) if not np.isnan(p) else None,
                "effect_r": float(r_effect) if not np.isnan(r_effect) else None,
                f"{a_name}_mean": float(av.mean()), f"{a_name}_std": float(av.std()),
                f"{b_name}_mean": float(bv.mean()), f"{b_name}_std": float(bv.std()),
            })
            sig = "SI" if (not np.isnan(p) and p<0.05) else "NO"
            log.info(f"  {m:15s} {a_name} vs {b_name}: p={p:.4f} sig={sig} r={r_effect:.3f}")

    df_stat = pd.DataFrame(stat_results)
    df_stat.to_csv("results/tables/estadisticas.csv", index=False)

    # Frecuencia de seleccion por feature
    with open(f"{cfg['data']['processed_dir']}/feature_names.json") as f:
        feat_names = json.load(f)
    n_feat = len(feat_names)
    pso_freq = np.zeros(n_feat); ba_freq = np.zeros(n_feat)
    for _, row in df_eval[df_eval["arm"]=="PSO"].iterrows():
        for idx in row["selected_idx"]: pso_freq[idx] += 1
    for _, row in df_eval[df_eval["arm"]=="BA"].iterrows():
        for idx in row["selected_idx"]: ba_freq[idx] += 1
    n_runs = cfg["experiment"]["n_runs"]
    feat_freq = pd.DataFrame({
        "feature": feat_names,
        "pso_freq_pct": pso_freq/n_runs*100,
        "ba_freq_pct":  ba_freq/n_runs*100,
    }).sort_values("pso_freq_pct", ascending=False)
    feat_freq.to_csv("results/tables/feature_frequency.csv", index=False)
    log.info(f"  Top 10 features PSO:\n{feat_freq.head(10).to_string()}")
    log.info("FASE 11 completa.")
    return df_stat, feat_freq

# ============================================================
#  FASE 12: ENTREGABLES (tablas y graficos)
# ============================================================
def fase12(cfg, df_baseline, df_eval, df_stat, feat_freq, baseline_ref):
    log.info("\n" + "="*60 + "\nFASE 12: ENTREGABLES\n" + "="*60)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import json

    fig_dir = "results/figures"
    tab_dir = "results/tables"
    os.makedirs(fig_dir, exist_ok=True)

    pso_df = df_eval[df_eval["arm"]=="PSO"].sort_values("run").reset_index(drop=True)
    ba_df  = df_eval[df_eval["arm"]=="BA"].sort_values("run").reset_index(drop=True)

    # ------ Tabla resumen metricas ------
    metrics = ["f1_macro","recall_atk","accuracy","TFP","TFD","n_features","t_train_s","t_infer_per_flow_us"]
    rows = []
    for arm, df in [("Baseline",df_baseline),("PSO",pso_df),("BA",ba_df)]:
        r = {"arm": arm}
        for m in metrics:
            if m in df.columns:
                r[f"{m}_mean"] = df[m].mean()
                r[f"{m}_std"]  = df[m].std()
        rows.append(r)
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(f"{tab_dir}/resumen_metricas.csv", index=False)

    # Tabla Markdown
    md_lines = ["# Resumen Metricas por Brazo\n"]
    md_lines.append("| Brazo | F1 macro | Recall atk | Accuracy | TFP | TFD | N features | T_train(s) |")
    md_lines.append("|-------|----------|------------|----------|-----|-----|-----------|-----------|")
    for _, row in df_summary.iterrows():
        md_lines.append(f"| {row['arm']} "
            f"| {row.get('f1_macro_mean',0):.4f}±{row.get('f1_macro_std',0):.4f} "
            f"| {row.get('recall_atk_mean',0):.4f}±{row.get('recall_atk_std',0):.4f} "
            f"| {row.get('accuracy_mean',0):.4f}±{row.get('accuracy_std',0):.4f} "
            f"| {row.get('TFP_mean',0):.4f}±{row.get('TFP_std',0):.4f} "
            f"| {row.get('TFD_mean',0):.4f}±{row.get('TFD_std',0):.4f} "
            f"| {row.get('n_features_mean',0):.1f}±{row.get('n_features_std',0):.1f} "
            f"| {row.get('t_train_s_mean',0):.2f}±{row.get('t_train_s_std',0):.2f} |")
    with open(f"{tab_dir}/resumen_metricas.md","w",encoding="utf-8") as f: f.write("\n".join(md_lines))

    # ------ Boxplots F1 macro y Recall ------
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax, metric, label in [(axes[0],"f1_macro","F1 Macro"),(axes[1],"recall_atk","Recall Ataque")]:
        data_plot = [df_baseline[metric].values, pso_df[metric].values, ba_df[metric].values]
        bp = ax.boxplot(data_plot, labels=["Baseline","PSO","BA"], patch_artist=True)
        colors = ["#4C72B0","#DD8452","#55A868"]
        for patch, color in zip(bp["boxes"], colors): patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_title(label, fontsize=13)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/boxplots_f1_recall.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------ Curvas de convergencia ------
    pso_files = sorted([f for f in os.listdir("results/runs") if f.startswith("pso_run_")])
    ba_files  = sorted([f for f in os.listdir("results/runs") if f.startswith("ba_run_")])
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    for ax, files, color, title in [(axes[0],pso_files,"#DD8452","PSO"),(axes[1],ba_files,"#55A868","BA")]:
        for fn in files:
            with open(f"results/runs/{fn}") as f2: r2=json.load(f2)
            ax.plot(r2["convergence"], alpha=0.3, color=color, linewidth=0.8)
        # Media
        convs = []
        for fn in files:
            with open(f"results/runs/{fn}") as f2: r2=json.load(f2)
            convs.append(r2["convergence"])
        if convs:
            mean_conv = np.mean(convs, axis=0)
            ax.plot(mean_conv, color="black", linewidth=2, label="Media")
        ax.set_title(f"Convergencia {title}", fontsize=13)
        ax.set_xlabel("Iteracion"); ax.set_ylabel("Fitness")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/convergencia.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------ F1 vs N features ------
    fig, ax = plt.subplots(figsize=(8,5))
    for arm, df_a, color, marker in [("PSO",pso_df,"#DD8452","o"),("BA",ba_df,"#55A868","s")]:
        ax.scatter(df_a["n_features"], df_a["f1_macro"], alpha=0.6, color=color, marker=marker, label=arm)
    ax.axhline(df_baseline["f1_macro"].mean(), color="#4C72B0", linestyle="--", label="Baseline media")
    ax.set_xlabel("N features seleccionadas"); ax.set_ylabel("F1 Macro")
    ax.set_title("F1 Macro vs Numero de Features"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/f1_vs_features.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------ Frecuencia de seleccion ------
    fig, ax = plt.subplots(figsize=(14,5))
    x = np.arange(len(feat_freq))
    w = 0.35
    ax.bar(x-w/2, feat_freq["pso_freq_pct"], w, label="PSO", color="#DD8452", alpha=0.7)
    ax.bar(x+w/2, feat_freq["ba_freq_pct"],  w, label="BA",  color="#55A868", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(feat_freq["feature"], rotation=90, fontsize=7)
    ax.set_ylabel("% corridas seleccionada"); ax.set_title("Frecuencia de seleccion por feature")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/feature_frequency.png", dpi=300, bbox_inches="tight")
    plt.close()

    log.info(f"Graficos guardados en {fig_dir}/")

    # ------ Resumen markdown ------
    with open("results/resumen_resultados.md","w",encoding="utf-8") as f:
        f.write("# Resumen de Resultados\n\n")
        f.write("## Hallazgos principales\n\n")
        pso_f1 = pso_df["f1_macro"].mean(); ba_f1 = ba_df["f1_macro"].mean(); bl_f1 = df_baseline["f1_macro"].mean()
        pso_rd = pso_df["RD"].mean();       ba_rd  = ba_df["RD"].mean()
        f.write(f"- **Baseline F1 macro**: {bl_f1:.4f}\n")
        f.write(f"- **PSO F1 macro**: {pso_f1:.4f} (Δ={pso_df['delta_f1'].mean():+.2f}%)\n")
        f.write(f"- **BA  F1 macro**: {ba_f1:.4f} (Δ={ba_df['delta_f1'].mean():+.2f}%)\n")
        f.write(f"- **PSO reduccion dim**: {pso_rd:.1f}%\n")
        f.write(f"- **BA  reduccion dim**: {ba_rd:.1f}%\n\n")
        f.write("## Hipotesis\n\n")
        f.write("- H1 (PSO/BA mejora F1): ver tabla estadisticas.csv\n")
        f.write("- H2 (reduce falsas alarmas): ver TFP/TFD en resumen_metricas.csv\n")
        f.write("- H3 (reduce dimensionalidad): ver RD en eval_runs.csv\n")
        f.write("- H4 (PSO vs BA difieren): ver pso_vs_ba en estadisticas.csv\n\n")
        f.write("## LIMITACIONES\n\n")
        f.write("- Etiquetas MAWILab por consenso de heuristicas (no ground truth absoluto)\n")
        f.write("- MAWILab sin actualizacion desde dic 2024\n")
        f.write("- Solo cabeceras TCP/IP (no payload)\n")
        f.write("- Un unico enlace transpacifico (samplepoint-F)\n")
        f.write("- Desbalance de clases tratado con class_weight=balanced\n")
        f.write("- Emparejamiento de etiquetas por IP/puerto sin ventana temporal precisa\n")
        f.write("- Submuestra durante busqueda puede no representar toda la distribucion\n")
        f.write("- CSV anomalias: nbDetectors ausente en datos (header dice 10 cols, datos tienen 9)\n")
        f.write("- 2021-02-10 extraida parcialmente (~34% del PCAP)\n")

    log.info("FASE 12 completa. Entregables en results/")

# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    cfg = load_config()
    os.makedirs("results/logs", exist_ok=True)

    # F4
    feat = fase4(cfg)

    # F5
    df_baseline, baseline_ref = fase5(cfg)

    # F7 - Calibracion
    calib = fase7(cfg)
    log.info(f"\n*** PUNTO DE PARADA 3 ***")
    log.info(f"Tiempo estimado total: {calib['t_total_est_h']:.1f} horas")
    log.info(f"Parametros: pop={calib['population']} iters={calib['iterations']} runs={calib['n_runs']} sub={calib['subsample_n']}")
    log.info("Revisa results/runs/calibration.json y confirma para continuar con Fase 8.")

    # F8
    pso_res, ba_res = fase8(cfg)

    # F9
    df_eval = fase9(cfg, pso_res, ba_res, baseline_ref)

    # F11
    df_stat, feat_freq = fase11(cfg, df_baseline, df_eval)

    # F12
    fase12(cfg, df_baseline, df_eval, df_stat, feat_freq, baseline_ref)

    log.info("\n" + "="*60)
    log.info("PIPELINE COMPLETO - Revisa results/")
    log.info("="*60)
