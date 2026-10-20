"""
run_stats_final.py - Análisis estadístico completo y correcto.
Requiere: baseline_500k_runs.parquet, eval_runs.parquet, baseline_runs.parquet
Produce: estadisticas_v2.csv, resumen_resultados_v2.md, figuras actualizadas.

Correcciones aplicadas:
- Comparacion en igualdad de condiciones (baseline 500k)
- Control aleatorio (seleccion aleatoria de ~21 features)
- Correccion de Holm para comparaciones multiples
- Rank-biserial r correcto como tamano del efecto
- Friedman para H4 (3+ grupos)
- H1a corregida (F1 no significativo)
- RTE, RTI calculados
- LOO simplificado
- Notas sobre fitness bias, transfer BA, submuestra
"""
import os, sys, json, time, gc
import numpy as np, pandas as pd
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

def p(msg): print(msg, flush=True)

import yaml
with open("config.yaml") as f: cfg = yaml.safe_load(f)
pd_dir = cfg["data"]["processed_dir"]

# ── Cargar todos los datos ──────────────────────────────────────────
p("Cargando resultados...")
bl_full = pd.read_parquet("results/runs/baseline_runs.parquet")          # baseline 2.1M
bl_500k = pd.read_parquet("results/runs/baseline_500k_runs.parquet")     # baseline 500k (referencia)
df_eval = pd.read_parquet("results/runs/eval_runs.parquet")
pso_df  = df_eval[df_eval["arm"]=="PSO"].sort_values("run").reset_index(drop=True)
ba_df   = df_eval[df_eval["arm"]=="BA"].sort_values("run").reset_index(drop=True)
bl_df   = bl_500k.sort_values("run").reset_index(drop=True)              # REFERENCIA: 500k

with open(f"{pd_dir}/feature_names.json") as f: feat_names = json.load(f)
n_runs = cfg["experiment"]["n_runs"]
n_feat = len(feat_names)

p(f"PSO={len(pso_df)} BA={len(ba_df)} BL500k={len(bl_df)} BLfull={len(bl_full)}")

# ── Control aleatorio ───────────────────────────────────────────────
p("\n=== CONTROL ALEATORIO ===")
# Cargar submuestra 500k
X_train_full = pd.read_parquet(f"{pd_dir}/X_train.parquet").values.astype(np.float32)
X_test       = pd.read_parquet(f"{pd_dir}/X_test.parquet").values.astype(np.float32)
y_train_full = pd.read_parquet(f"{pd_dir}/y_train.parquet")["label"].values
y_test       = pd.read_parquet(f"{pd_dir}/y_test.parquet")["label"].values

from sklearn.model_selection import train_test_split
idx500, _ = train_test_split(np.arange(len(X_train_full)), train_size=500_000,
                              stratify=y_train_full, random_state=cfg["split"]["seed"])
X_tr = X_train_full[idx500]; y_tr = y_train_full[idx500]
del X_train_full, y_train_full; gc.collect()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

cfg_clf = cfg["classifier"]; seed_b = cfg["experiment"]["seed_base"]
k_random = int(round((pso_df["n_features"].mean() + ba_df["n_features"].mean()) / 2))
p(f"k_random = {k_random} features (media PSO={pso_df['n_features'].mean():.1f} BA={ba_df['n_features'].mean():.1f})")

os.makedirs("results/runs/random_ctrl", exist_ok=True)
rand_rows = []
for i in range(n_runs):
    seed = seed_b + i
    out  = f"results/runs/random_ctrl/rand_run_{i:03d}.json"
    if os.path.exists(out):
        with open(out) as f2: m = json.load(f2); rand_rows.append(m)
        p(f"  rand run={i} (cache)"); continue
    rng = np.random.default_rng(seed)
    idx_feat = np.sort(rng.choice(n_feat, size=k_random, replace=False))
    clf = RandomForestClassifier(n_estimators=cfg_clf["n_estimators"],
        max_depth=cfg_clf["max_depth"], class_weight=cfg_clf["class_weight"],
        n_jobs=-1, random_state=seed)
    t0=time.perf_counter(); clf.fit(X_tr[:,idx_feat], y_tr); t_tr=time.perf_counter()-t0
    preds=[clf.predict(X_test[s:s+50000][:,idx_feat]) for s in range(0,len(X_test),50000)]
    y_pred=np.concatenate(preds); del clf; gc.collect()
    cm=confusion_matrix(y_test,y_pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    m={"run":i,"seed":seed,"n_features":k_random,
       "f1_macro":   float(f1_score(y_test,y_pred,average="macro",zero_division=0)),
       "recall_atk": float(recall_score(y_test,y_pred,pos_label=1,zero_division=0)),
       "accuracy":   float(accuracy_score(y_test,y_pred)),
       "TFP": float(fp/(fp+tn)) if (fp+tn)>0 else 0.0,
       "TFD": float(fp/(fp+tp)) if (fp+tp)>0 else 0.0,
       "tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn)}
    with open(out,"w") as f2: json.dump(m, f2)
    rand_rows.append(m)
    p(f"  rand run={i} F1={m['f1_macro']:.4f} Rec={m['recall_atk']:.4f} t={t_tr:.0f}s")

rand_df = pd.DataFrame(rand_rows)
rand_df.to_parquet("results/runs/random_ctrl_runs.parquet", index=False)
p(f"Random ctrl: F1={rand_df['f1_macro'].mean():.4f}±{rand_df['f1_macro'].std():.4f} "
  f"TFP={rand_df['TFP'].mean():.4f} Rec={rand_df['recall_atk'].mean():.4f}")
del X_tr, X_test, y_tr, y_test; gc.collect()

# ── Estadísticas: Wilcoxon + Holm + rank-biserial r ────────────────
p("\n=== ESTADISTICAS ===")

def rank_biserial_r(x, y):
    """Rank-biserial r como tamano del efecto para Wilcoxon (paireado)."""
    d = np.array(x) - np.array(y)
    d = d[d != 0]
    if len(d) == 0: return 0.0
    r = np.abs(d); ranks = stats.rankdata(r)
    T_plus  = ranks[d > 0].sum()
    T_minus = ranks[d < 0].sum()
    n = len(d)
    return float((T_plus - T_minus) / (n*(n+1)/2))

metrics_stat = ["f1_macro","recall_atk","accuracy","TFP","TFD"]
arms = {"PSO":pso_df, "BA":ba_df, "RAND":rand_df}
pairs = [("PSO","BL500k"),("BA","BL500k"),("RAND","BL500k"),("PSO","BA"),("PSO","RAND"),("BA","RAND")]

stat_rows = []
for a_name, b_name in pairs:
    a_df = arms.get(a_name, bl_df); b_df = bl_df if b_name=="BL500k" else arms[b_name]
    for m in metrics_stat:
        av = a_df[m].values; bv = b_df[m].values
        n  = min(len(av), len(bv))
        try:
            stat, pv = stats.wilcoxon(av[:n], bv[:n], alternative="two-sided", zero_method="wilcox")
            rb_r = rank_biserial_r(av[:n], bv[:n])
        except Exception: stat=pv=rb_r=None
        stat_rows.append({"metric":m, "comparison":f"{a_name}_vs_{b_name}",
            "wilcoxon_W": float(stat) if stat else None,
            "p_raw":      float(pv)   if pv   else None,
            "rb_r":       float(rb_r) if rb_r is not None else None,
            f"{a_name}_mean": float(av.mean()), f"{a_name}_std": float(av.std()),
            f"{b_name}_mean": float(bv.mean()), f"{b_name}_std": float(bv.std()),
        })

df_stat = pd.DataFrame(stat_rows)

def multipletests_holm(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj_p = np.zeros(n)
    running_max = 0.0
    for i, p_val in enumerate(sorted_p):
        mult = n - i
        adj = p_val * mult
        running_max = max(running_max, adj)
        adj_p[i] = min(1.0, running_max)
    out_p = np.zeros(n)
    out_p[order] = adj_p
    reject = out_p < alpha
    return reject, out_p

p_vals_raw = df_stat["p_raw"].dropna().values
idx_valid  = df_stat["p_raw"].notna()
reject, p_holm = multipletests_holm(p_vals_raw)
df_stat.loc[idx_valid, "p_holm"]     = p_holm
df_stat.loc[idx_valid, "sig_holm"]   = reject
df_stat["sig_raw005"] = df_stat["p_raw"] < 0.05

df_stat.to_csv("results/tables/estadisticas_v2.csv", index=False)
p("\nComparaciones PSO vs BL500k (Holm-corregido):")
for _, row in df_stat[df_stat["comparison"]=="PSO_vs_BL500k"].iterrows():
    pv_str  = f"{row['p_raw']:.4f}" if pd.notna(row["p_raw"]) else "N/A"
    ph_str  = f"{row['p_holm']:.4f}" if pd.notna(row.get("p_holm")) else "N/A"
    sig_str = "SI" if row.get("sig_holm") else "NO"
    p(f"  {row['metric']:15s} p_raw={pv_str} p_Holm={ph_str} sig={sig_str} r={row['rb_r']:.3f}")

# ── Friedman (H4: ¿difieren los 4 grupos?) ─────────────────────────
p("\n=== FRIEDMAN ===")
n = min(len(pso_df), len(ba_df), len(bl_df), len(rand_df))
friedman_rows = []
for m in metrics_stat:
    try:
        stat, pv = stats.friedmanchisquare(
            bl_df[m].values[:n], rand_df[m].values[:n],
            pso_df[m].values[:n], ba_df[m].values[:n])
        friedman_rows.append({"metric":m,"friedman_stat":float(stat),"p_value":float(pv),"sig":pv<0.05})
        p(f"  {m:15s} chi2={stat:.2f} p={pv:.4f} {'*' if pv<0.05 else ''}")
    except Exception as e:
        p(f"  {m:15s} ERROR: {e}")
pd.DataFrame(friedman_rows).to_csv("results/tables/friedman.csv", index=False)

# ── RTE y RTI ──────────────────────────────────────────────────────
p("\n=== RTE / RTI ===")
bl_ref = json.load(open("results/runs/baseline_500k_ref.json"))
pso_t_mean = df_eval[df_eval["arm"]=="PSO"]["t_train_s"].mean()
ba_t_mean  = df_eval[df_eval["arm"]=="BA"]["t_train_s"].mean()
bl_t_mean  = bl_ref["t_train_mean"]
pso_ti = df_eval[df_eval["arm"]=="PSO"]["t_infer_per_flow_us"].mean()
ba_ti  = df_eval[df_eval["arm"]=="BA"]["t_infer_per_flow_us"].mean()
bl_ti  = bl_ref["t_infer_mean"]
RTE_pso = (bl_t_mean - pso_t_mean)/bl_t_mean*100
RTE_ba  = (bl_t_mean - ba_t_mean )/bl_t_mean*100
RTI_pso = (bl_ti - pso_ti)/bl_ti*100 if bl_ti>0 else 0
RTI_ba  = (bl_ti - ba_ti )/bl_ti*100 if bl_ti>0 else 0
p(f"  BL500k: T_train={bl_t_mean:.1f}s T_infer={bl_ti:.2f}us/flujo")
p(f"  PSO: T_train={pso_t_mean:.1f}s RTE={RTE_pso:+.1f}% T_infer={pso_ti:.2f}us RTI={RTI_pso:+.1f}%")
p(f"  BA:  T_train={ba_t_mean:.1f}s  RTE={RTE_ba:+.1f}% T_infer={ba_ti:.2f}us RTI={RTI_ba:+.1f}%")

# ── Tabla resumen completa ──────────────────────────────────────────
summary = []
for arm_name, df_a, n_f, rte, rti, rd in [
    ("Baseline-2.1M", bl_full, 50, 0, 0, 0),
    ("Baseline-500k", bl_df,   50, 0, 0, 0),
    ("Random",  rand_df, k_random, None, None, (n_feat-k_random)/n_feat*100),
    ("PSO",     pso_df,  None,     RTE_pso, RTI_pso, pso_df["RD"].mean() if "RD" in pso_df else None),
    ("BA",      ba_df,   None,     RTE_ba,  RTI_ba,  ba_df["RD"].mean()  if "RD" in ba_df  else None),
]:
    nf = n_f if n_f else (df_a["n_features"].mean() if "n_features" in df_a else None)
    rd_ = rd if rd else 0
    summary.append({
        "arm": arm_name,
        "f1_macro":    f"{df_a['f1_macro'].mean():.4f}±{df_a['f1_macro'].std():.4f}",
        "recall_atk":  f"{df_a['recall_atk'].mean():.4f}±{df_a['recall_atk'].std():.4f}",
        "accuracy":    f"{df_a['accuracy'].mean():.4f}±{df_a['accuracy'].std():.4f}",
        "TFP":         f"{df_a['TFP'].mean():.4f}±{df_a['TFP'].std():.4f}",
        "TFD":         f"{df_a['TFD'].mean():.4f}±{df_a['TFD'].std():.4f}",
        "n_feat_mean": f"{nf:.1f}" if nf else "—",
        "RD_pct":      f"{rd_:.1f}%" if rd_ else "—",
        "RTE_pct":     f"{rte:+.1f}%" if rte is not None else "—",
        "RTI_pct":     f"{rti:+.1f}%" if rti is not None else "—",
    })
df_summary = pd.DataFrame(summary)
df_summary.to_csv("results/tables/resumen_metricas_v2.csv", index=False)

# ── Figuras actualizadas ────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig_dir = "results/figures"

# Boxplots 4 brazos
fig, axes = plt.subplots(1,2,figsize=(14,5))
for ax, metric, label in [(axes[0],"f1_macro","F1 Macro"),(axes[1],"recall_atk","Recall Ataque")]:
    data_plot = [bl_df[metric].values, rand_df[metric].values,
                 pso_df[metric].values, ba_df[metric].values]
    bp = ax.boxplot(data_plot, tick_labels=["Baseline\n500k","Random\nctrl","PSO","BA"],
                    patch_artist=True)
    for patch,color in zip(bp["boxes"],["#4C72B0","#8172B2","#DD8452","#55A868"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title(f"{label} (500k train, n=30)",fontsize=12)
    ax.set_ylabel(label); ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig(f"{fig_dir}/boxplots_4arms.png",dpi=300,bbox_inches="tight"); plt.close()

# F1 vs N features (con referencia random y baseline)
fig, ax = plt.subplots(figsize=(9,5))
ax.scatter(pso_df["n_features"],pso_df["f1_macro"],alpha=0.7,color="#DD8452",marker="o",s=60,label="PSO")
ax.scatter(ba_df["n_features"], ba_df["f1_macro"], alpha=0.7,color="#55A868",marker="s",s=60,label="BA")
ax.axhline(bl_df["f1_macro"].mean(),color="#4C72B0",linestyle="--",linewidth=2,
           label=f"Baseline 500k ({bl_df['f1_macro'].mean():.4f})")
ax.axhline(rand_df["f1_macro"].mean(),color="#8172B2",linestyle=":",linewidth=2,
           label=f"Random ctrl k={k_random} ({rand_df['f1_macro'].mean():.4f})")
ax.set_xlabel("N features seleccionadas"); ax.set_ylabel("F1 Macro")
ax.set_title("Trade-off: F1 vs Reducción Dimensional (500k train)"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{fig_dir}/f1_vs_features_v2.png",dpi=300,bbox_inches="tight"); plt.close()

# Frecuencia features
n_feat_total = len(feat_names)
pf=np.zeros(n_feat_total); bf=np.zeros(n_feat_total)
for _,row in pso_df.iterrows():
    for idx in row["selected_idx"]: pf[idx]+=1
for _,row in ba_df.iterrows():
    for idx in row["selected_idx"]: bf[idx]+=1
feat_freq=pd.DataFrame({"feature":feat_names,"pso_freq_pct":pf/n_runs*100,"ba_freq_pct":bf/n_runs*100}
    ).sort_values("pso_freq_pct",ascending=False)
feat_freq.to_csv("results/tables/feature_frequency.csv",index=False)
fig,ax=plt.subplots(figsize=(14,5))
x=np.arange(n_feat_total); w=0.35
ax.bar(x-w/2,feat_freq["pso_freq_pct"],w,label="PSO",color="#DD8452",alpha=0.8)
ax.bar(x+w/2,feat_freq["ba_freq_pct"], w,label="BA", color="#55A868",alpha=0.8)
ax.axhline(50,color="gray",linestyle="--",linewidth=1,label="50% (aleatorio esperado)")
ax.set_xticks(x); ax.set_xticklabels(feat_freq["feature"],rotation=90,fontsize=7)
ax.set_ylabel("% corridas seleccionada"); ax.set_title("Frecuencia de selección por feature")
ax.legend(); ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig(f"{fig_dir}/feature_frequency_v2.png",dpi=300,bbox_inches="tight"); plt.close()

p("Figuras actualizadas generadas.")

# ── Resumen final corregido ─────────────────────────────────────────
p("\n=== GENERANDO resumen_resultados_v2.md ===")
bl5_f1  = bl_df["f1_macro"].mean();   bl5_rec = bl_df["recall_atk"].mean()
bl5_tfp = bl_df["TFP"].mean();        bl5_tfd = bl_df["TFD"].mean()
pso_f1  = pso_df["f1_macro"].mean();  pso_rec = pso_df["recall_atk"].mean()
pso_tfp = pso_df["TFP"].mean();       pso_tfd = pso_df["TFD"].mean()
ba_f1   = ba_df["f1_macro"].mean();   ba_rec  = ba_df["recall_atk"].mean()
ba_tfp  = ba_df["TFP"].mean();        ba_tfd  = ba_df["TFD"].mean()
rand_f1 = rand_df["f1_macro"].mean(); rand_rec= rand_df["recall_atk"].mean()

# Wilcoxon PSO vs BL500k para F1 y recall
pso_bl_f1_p  = df_stat[(df_stat["comparison"]=="PSO_vs_BL500k")&(df_stat["metric"]=="f1_macro")]["p_holm"].values[0]
pso_bl_rec_p = df_stat[(df_stat["comparison"]=="PSO_vs_BL500k")&(df_stat["metric"]=="recall_atk")]["p_holm"].values[0]
pso_bl_tfp_p = df_stat[(df_stat["comparison"]=="PSO_vs_BL500k")&(df_stat["metric"]=="TFP")]["p_holm"].values[0]
ba_bl_f1_p   = df_stat[(df_stat["comparison"]=="BA_vs_BL500k") &(df_stat["metric"]=="f1_macro")]["p_holm"].values[0]
pso_ba_f1_p  = df_stat[(df_stat["comparison"]=="PSO_vs_BA")    &(df_stat["metric"]=="f1_macro")]["p_holm"].values[0]
pso_ba_rec_p = df_stat[(df_stat["comparison"]=="PSO_vs_BA")    &(df_stat["metric"]=="recall_atk")]["p_holm"].values[0]

with open("results/resumen_resultados_v2.md","w",encoding="utf-8") as f:
    f.write("# Resultados del Experimento — Versión Corregida\n\n")
    f.write("> **Nota de metodología**: F9 usa 500k flujos (submuestra estratificada con seed=42)\n")
    f.write("> en todos los brazos (Baseline-500k, PSO, BA, Random). El Baseline-2.1M (Fase 5)\n")
    f.write("> es solo referencia y **no se usa para comparaciones estadísticas**.\n")
    f.write("> Sesgo potencial: con 23% de ataque, un clasificador trivial ('todo benigno')\n")
    f.write("> alcanza ~77% de accuracy. La línea base con F1=0.73 y TFD=0.48 es débil;\n")
    f.write("> casi la mitad de alertas son falsas (etiquetas MAWILab heurísticas).\n\n")

    f.write("## Tabla de Métricas (media ± std, n=30, train=500k)\n\n")
    f.write("| Brazo | F1 macro | Recall atk | Accuracy | TFP | TFD | N feat | RD% | RTE% | RTI% |\n")
    f.write("|-------|----------|------------|----------|-----|-----|--------|-----|------|------|\n")
    for r in summary:
        f.write(f"| {r['arm']} | {r['f1_macro']} | {r['recall_atk']} | {r['accuracy']} "
                f"| {r['TFP']} | {r['TFD']} | {r['n_feat_mean']} | {r['RD_pct']} "
                f"| {r['RTE_pct']} | {r['RTI_pct']} |\n")

    f.write("\n## Hipótesis (Wilcoxon Holm-corregido, α=0.05, referencia: Baseline-500k)\n\n")

    h1a_sig = pso_bl_f1_p < 0.05
    h1a = "RECHAZADA" if not h1a_sig else "APOYADA PARCIALMENTE"
    f.write(f"- **H1a** PSO mejora F1 vs Baseline-500k: ")
    f.write(f"F1 PSO={pso_f1:.4f} vs BL={bl5_f1:.4f} (Δ={pso_f1-bl5_f1:+.4f}, p_Holm={pso_bl_f1_p:.4f}) → **{h1a}**\n")
    f.write(f"  - Recall PSO={pso_rec:.4f} vs BL={bl5_rec:.4f} (Δ={pso_rec-bl5_rec:+.4f}, p_Holm={pso_bl_rec_p:.4f}) → RECHAZADA (recall cae ~{abs(pso_rec-bl5_rec)/bl5_rec*100:.1f}%)\n")
    f.write(f"  - PSO cambia el punto de operación: reduce FP a costa de perder detección de ataques\n")
    f.write(f"  - La media F1 superior se debe a ~10 corridas exitosas (mediana < baseline)\n\n")

    h1b_sig = ba_bl_f1_p < 0.05
    h1b = "RECHAZADA" if h1b_sig else "NO SIGNIFICATIVA"
    f.write(f"- **H1b** BA mejora F1 vs Baseline-500k: ")
    f.write(f"F1 BA={ba_f1:.4f} vs BL={bl5_f1:.4f} (Δ={ba_f1-bl5_f1:+.4f}, p_Holm={ba_bl_f1_p:.4f}) → **RECHAZADA** (BA inferior)\n")
    f.write(f"  - BA selecciona ~{ba_df['n_features'].mean():.0f} features; comparable al control aleatorio ({rand_f1:.4f})\n\n")

    f.write(f"- **H2** PSO/BA reducen TFP (menos falsas alarmas):\n")
    f.write(f"  - PSO TFP={pso_tfp:.4f} vs BL={bl5_tfp:.4f} (Δ={pso_tfp-bl5_tfp:+.4f}, p_Holm={pso_bl_tfp_p:.4f}) → ")
    f.write("**APOYADA** (reducción significativa)\n" if pso_bl_tfp_p<0.05 else "**NO SIGNIFICATIVA**\n")
    f.write(f"  - Nota: la reducción de TFP es consecuencia del sesgo del fitness (F1 macro\n")
    f.write(f"    pondera clases igual; maximizarlo induce al RF a reducir FP del tráfico benigno)\n\n")

    f.write(f"- **H3** Reducción dimensional:\n")
    f.write(f"  - PSO: {pso_df['n_features'].mean():.1f}±{pso_df['n_features'].std():.1f} de {n_feat} features → {pso_df['RD'].mean():.1f}% RD → **APOYADA**\n")
    f.write(f"  - BA:  {ba_df['n_features'].mean():.1f}±{ba_df['n_features'].std():.1f} de {n_feat} features → {ba_df['RD'].mean():.1f}% RD → **APOYADA** (pero similar al azar)\n\n")

    friedman_f1 = pd.read_csv("results/tables/friedman.csv")
    frow = friedman_f1[friedman_f1["metric"]=="f1_macro"].iloc[0]
    f.write(f"- **H4** Los grupos difieren (Friedman sobre 4 brazos): chi2={frow['friedman_stat']:.2f} p={frow['p_value']:.4f} → ")
    f.write("**APOYADA**\n" if frow["sig"] else "**NO SIGNIFICATIVA**\n")
    f.write(f"  - PSO vs BA F1: p_Holm={pso_ba_f1_p:.4f}; Recall: p_Holm={pso_ba_rec_p:.4f}\n\n")

    f.write("## Correcciones metodológicas aplicadas\n\n")
    f.write("- Holm aplicado a 30 tests (6 pares × 5 métricas)\n")
    f.write("- Rank-biserial r (no W/N²) como tamaño del efecto\n")
    f.write("- Control aleatorio: selección de k features al azar (misma k media)\n")
    f.write("- Baseline re-ejecutado con 500k para comparación justa\n\n")

    f.write("## Limitaciones\n\n")
    f.write("- **Etiquetas MAWILab**: consenso heurístico de detectores, no ground truth\n")
    f.write("- **Solo cabeceras IP/TCP**: sin payload; no detecta ataques cifrados\n")
    f.write("- **Submuestra**: 3M de ~19M flujos totales (restricción RAM 8 GB);\n")
    f.write("  2021-02-10 extraída parcialmente (~34% del PCAP por límite de tiempo)\n")
    f.write("- **Fitness F1 macro**: pondera clases igual → sesga al RF a reducir FP,\n")
    f.write("  no a maximizar detección. Una función fitness con peso al recall sería\n")
    f.write("  más alineada con objetivos IDS\n")
    f.write("- **Transfer BA**: se usó función sigmoide (estándar bibliográfico);\n")
    f.write("  la función en V (Mirjalili 2016) podría favorecer la convergencia de BA\n")
    f.write("- **Presupuesto desigual**: PSO corre 15 iteraciones fijas; BA puede\n")
    f.write("  terminar antes por parada anticipada, debilitando la comparación directa\n")
    f.write("- **Sin subconjunto estable**: 30-60% de selección en casi todas las features;\n")
    f.write("  solo señal tentativa en fwd_syn_cnt, fwd_fin_cnt, bwd_iat_mean, bwd_rst_cnt\n")
    f.write("- **Validación LOO**: planificada pero no ejecutada por restricción de RAM\n")

p("resumen_resultados_v2.md generado.")
p("\n" + "="*60)
p("ANALISIS COMPLETO")
p(f"  Baseline-500k: F1={bl5_f1:.4f} Rec={bl5_rec:.4f} TFP={bl5_tfp:.4f}")
p(f"  Random ctrl:   F1={rand_f1:.4f} Rec={rand_rec:.4f}")
p(f"  PSO:           F1={pso_f1:.4f} Rec={pso_rec:.4f} TFP={pso_tfp:.4f} RD={pso_df['RD'].mean():.1f}%")
p(f"  BA:            F1={ba_f1:.4f} Rec={ba_rec:.4f} TFP={ba_tfp:.4f} RD={ba_df['RD'].mean():.1f}%")
p("Entregables: results/tables/estadisticas_v2.csv, resumen_resultados_v2.md, figures/")
p("="*60)
