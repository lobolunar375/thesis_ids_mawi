"""
run_fase9_12_fast.py - F9 rapida: entrena sobre 500k flujos (stratificado).
Justificacion: submuestra consistente en todos los brazos, hardware limitado.
Cache por run: puede reiniciarse sin perder trabajo.
"""
import os, sys, json, time, gc, logging
import numpy as np, pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")
os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/runs/eval", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler("results/logs/fase9_12_fast.log","a","utf-8")])
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
with open(f"{pd_dir}/feature_names.json") as f: feat_names = json.load(f)

# Submuestra de entrenamiento de 500k (estratificada, reproducible)
from sklearn.model_selection import train_test_split
EVAL_TRAIN_N = 500_000
SEED = cfg["split"]["seed"]
idx500, _ = train_test_split(np.arange(len(X_train_full)), train_size=EVAL_TRAIN_N,
                              stratify=y_train_full, random_state=SEED)
X_tr = X_train_full[idx500]
y_tr = y_train_full[idx500]
del X_train_full, y_train_full; gc.collect()
p(f"Train eval={X_tr.shape} Test={X_test.shape} atk_train={y_tr.mean()*100:.1f}%")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

N_total = len(feat_names)
cfg_clf = cfg["classifier"]
seed_b  = cfg["experiment"]["seed_base"]
n_runs  = cfg["experiment"]["n_runs"]

with open("results/runs/baseline_ref.json") as f: baseline_ref = json.load(f)

def compute_metrics(y_true, y_pred, t_train, t_infer):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn,fp,fn,tp = cm.ravel()
    return {
        "f1_macro":   float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
        "recall_atk": float(recall_score(y_true,y_pred,pos_label=1,zero_division=0)),
        "accuracy":   float(accuracy_score(y_true,y_pred)),
        "TFP": float(fp/(fp+tn)) if (fp+tn)>0 else 0.0,
        "TFD": float(fp/(fp+tp)) if (fp+tp)>0 else 0.0,
        "tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn),
        "t_train_s": float(t_train),
        "t_infer_per_flow_us": float(t_infer/len(y_test)*1e6),
    }

def eval_run(algo, run_id, mask, seed):
    out = f"results/runs/eval/eval_{algo.lower()}_run_{run_id:03d}.json"
    if os.path.exists(out):
        with open(out) as f: return json.load(f), True

    idx = np.where(np.array(mask) >= 0.5)[0]
    if len(idx) == 0: idx = np.arange(N_total)
    n_sel = len(idx)

    clf = RandomForestClassifier(
        n_estimators=cfg_clf["n_estimators"], max_depth=cfg_clf["max_depth"],
        class_weight=cfg_clf["class_weight"], n_jobs=-1, random_state=seed)

    t0 = time.perf_counter()
    clf.fit(X_tr[:, idx], y_tr)
    t_tr = time.perf_counter()-t0

    t1 = time.perf_counter()
    preds = [clf.predict(X_test[s:s+50000, :][:, idx]) for s in range(0, len(X_test), 50000)]
    y_pred = np.concatenate(preds)
    t_inf = time.perf_counter()-t1

    del clf; gc.collect()

    m = compute_metrics(y_test, y_pred, t_tr, t_inf)
    m["n_features"] = int(n_sel)
    m["RD"]  = float((N_total-n_sel)/N_total*100)
    m["delta_f1"]  = float((m["f1_macro"] -baseline_ref["f1_macro_mean"] )/baseline_ref["f1_macro_mean"]*100)
    m["delta_rec"] = float((m["recall_atk"]-baseline_ref["recall_atk_mean"])/baseline_ref["recall_atk_mean"]*100)
    m["delta_TFP"] = float((m["TFP"]-baseline_ref["TFP_mean"])/baseline_ref["TFP_mean"]*100) if baseline_ref["TFP_mean"]>0 else 0.0
    m["run"]=run_id; m["seed"]=seed; m["arm"]=algo
    m["selected_idx"]=idx.tolist()
    with open(out,"w") as f: json.dump(m, f)
    return m, False

# ============================================================
# FASE 9
# ============================================================
p("\n=== FASE 9: EVALUACION FINAL (train=500k, n_jobs=-1) ===")
pso_res=[]; ba_res=[]
for i in range(n_runs):
    with open(f"results/runs/pso_run_{i:03d}.json") as f: pso_res.append(json.load(f))
    with open(f"results/runs/ba_run_{i:03d}.json")  as f: ba_res.append(json.load(f))

all_eval=[]; t9=time.time()
for algo, runs in [("PSO",pso_res),("BA",ba_res)]:
    for res in runs:
        i=res["run_id"]; seed=seed_b+i
        m,cached=eval_run(algo,i,res["best_mask"],seed)
        m["best_fitness"]=res["best_fitness"]; m["t_search_s"]=res["t_search_s"]
        all_eval.append(m)
        tag="(cache)" if cached else f"t={m['t_train_s']:.0f}s"
        p(f"  {algo} run={i} n_sel={m['n_features']} F1={m['f1_macro']:.4f} "
          f"RD={m['RD']:.1f}% dF1={m['delta_f1']:+.2f}% {tag}")

df_eval=pd.DataFrame(all_eval)
df_eval.to_csv("results/tables/eval_runs.csv",index=False)
df_eval.to_parquet("results/runs/eval_runs.parquet",index=False)
p(f"FASE 9 completa en {time.time()-t9:.0f}s")

# ============================================================
# FASE 11
# ============================================================
p("\n=== FASE 11: ESTADISTICAS ===")
from scipy import stats
df_baseline=pd.read_parquet("results/runs/baseline_runs.parquet")
pso_df=df_eval[df_eval["arm"]=="PSO"].sort_values("run").reset_index(drop=True)
ba_df =df_eval[df_eval["arm"]=="BA"].sort_values("run").reset_index(drop=True)
bl_df =df_baseline.sort_values("run").reset_index(drop=True)

metrics_stat=["f1_macro","recall_atk","accuracy","TFP","TFD"]
stat_rows=[]
for a_name,b_name in [("PSO","baseline"),("BA","baseline"),("PSO","BA")]:
    for m in metrics_stat:
        av=pso_df[m].values if a_name=="PSO" else ba_df[m].values
        bv=bl_df[m].values  if b_name=="baseline" else ba_df[m].values
        n=min(len(av),len(bv))
        try:
            stat,pv=stats.wilcoxon(av[:n],bv[:n],alternative="two-sided",zero_method="wilcox")
            re=stat/(n*(n+1)/2); sig=pv<0.05
        except: stat=pv=re=None; sig=None
        stat_rows.append({"metric":m,"comparison":f"{a_name}_vs_{b_name}",
            "wilcoxon_stat":float(stat) if stat is not None else None,
            "p_value":float(pv) if pv is not None else None,
            "significant_005":sig,"effect_r":float(re) if re is not None else None,
            f"{a_name}_mean":float(av.mean()),f"{a_name}_std":float(av.std()),
            f"{b_name}_mean":float(bv.mean()),f"{b_name}_std":float(bv.std())})
        pv_str = f"{pv:.4f}" if pv is not None else "N/A"
        p(f"  {m:15s} {a_name} vs {b_name}: p={pv_str} sig={'SI' if sig else 'NO'}")

pd.DataFrame(stat_rows).to_csv("results/tables/estadisticas.csv",index=False)
n_feat=len(feat_names); pf=np.zeros(n_feat); bf=np.zeros(n_feat)
for _,row in pso_df.iterrows():
    for idx in row["selected_idx"]: pf[idx]+=1
for _,row in ba_df.iterrows():
    for idx in row["selected_idx"]: bf[idx]+=1
feat_freq=pd.DataFrame({"feature":feat_names,"pso_freq_pct":pf/n_runs*100,"ba_freq_pct":bf/n_runs*100}
    ).sort_values("pso_freq_pct",ascending=False)
feat_freq.to_csv("results/tables/feature_frequency.csv",index=False)
p(f"Top 5 PSO: {feat_freq['feature'].head(5).tolist()}")
p("FASE 11 completa.")

# ============================================================
# FASE 12
# ============================================================
p("\n=== FASE 12: ENTREGABLES ===")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig_dir="results/figures"; os.makedirs(fig_dir,exist_ok=True)

# Resumen tabla
rows_t=[]
for arm,df_a in [("Baseline",bl_df),("PSO",pso_df),("BA",ba_df)]:
    r={"arm":arm}
    for m in ["f1_macro","recall_atk","accuracy","TFP","TFD","n_features","t_train_s","RD"]:
        if m in df_a.columns: r[f"{m}_mean"]=df_a[m].mean(); r[f"{m}_std"]=df_a[m].std()
    rows_t.append(r)
pd.DataFrame(rows_t).to_csv("results/tables/resumen_metricas.csv",index=False)

md=["# Resumen de Métricas (media±std, 30 corridas)\n",
    "| Brazo | F1 macro | Recall atk | Accuracy | TFP | TFD | N feat | RD% |",
    "|-------|----------|------------|----------|-----|-----|--------|-----|"]
for r in rows_t:
    md.append(f"| {r['arm']} "
        f"| {r.get('f1_macro_mean',0):.4f}±{r.get('f1_macro_std',0):.4f} "
        f"| {r.get('recall_atk_mean',0):.4f}±{r.get('recall_atk_std',0):.4f} "
        f"| {r.get('accuracy_mean',0):.4f}±{r.get('accuracy_std',0):.4f} "
        f"| {r.get('TFP_mean',0):.4f}±{r.get('TFP_std',0):.4f} "
        f"| {r.get('TFD_mean',0):.4f}±{r.get('TFD_std',0):.4f} "
        f"| {r.get('n_features_mean',0):.1f}±{r.get('n_features_std',0):.1f} "
        f"| {r.get('RD_mean',0):.1f}±{r.get('RD_std',0):.1f} |")
with open("results/tables/resumen_metricas.md","w",encoding="utf-8") as f: f.write("\n".join(md))

# Boxplots
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,metric,label in [(axes[0],"f1_macro","F1 Macro"),(axes[1],"recall_atk","Recall Ataque")]:
    bp=ax.boxplot([bl_df[metric].values,pso_df[metric].values,ba_df[metric].values],
                  tick_labels=["Baseline","PSO","BA"],patch_artist=True)
    for patch,color in zip(bp["boxes"],["#4C72B0","#DD8452","#55A868"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title(label,fontsize=13); ax.set_ylabel(label); ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig(f"{fig_dir}/boxplots_f1_recall.png",dpi=300,bbox_inches="tight"); plt.close()

# Convergencia
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,prefix,color,title in [(axes[0],"pso","#DD8452","PSO"),(axes[1],"ba","#55A868","BA")]:
    convs=[]
    for i in range(n_runs):
        with open(f"results/runs/{prefix}_run_{i:03d}.json") as f2: r2=json.load(f2)
        ax.plot(r2["convergence"],alpha=0.25,color=color,linewidth=0.8); convs.append(r2["convergence"])
    # Pad a la misma longitud antes de calcular la media
    max_len = max(len(c) for c in convs)
    convs_pad = [c + [c[-1]]*(max_len-len(c)) for c in convs]
    ax.plot(np.mean(convs_pad,axis=0),color="black",linewidth=2,label="Media")
    ax.set_title(f"Convergencia {title}"); ax.set_xlabel("Iteración"); ax.set_ylabel("Fitness")
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{fig_dir}/convergencia.png",dpi=300,bbox_inches="tight"); plt.close()

# F1 vs N features
fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(pso_df["n_features"],pso_df["f1_macro"],alpha=0.7,color="#DD8452",marker="o",s=60,label="PSO")
ax.scatter(ba_df["n_features"], ba_df["f1_macro"], alpha=0.7,color="#55A868",marker="s",s=60,label="BA")
ax.axhline(bl_df["f1_macro"].mean(),color="#4C72B0",linestyle="--",linewidth=2,
           label=f"Baseline F5 ({bl_df['f1_macro'].mean():.4f})")
ax.set_xlabel("N features seleccionadas"); ax.set_ylabel("F1 Macro")
ax.set_title("Trade-off: F1 vs Reducción Dimensional"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{fig_dir}/f1_vs_features.png",dpi=300,bbox_inches="tight"); plt.close()

# Frecuencia features
fig,ax=plt.subplots(figsize=(14,5))
x=np.arange(len(feat_freq)); w=0.35
ax.bar(x-w/2,feat_freq["pso_freq_pct"],w,label="PSO",color="#DD8452",alpha=0.8)
ax.bar(x+w/2,feat_freq["ba_freq_pct"], w,label="BA", color="#55A868",alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(feat_freq["feature"],rotation=90,fontsize=7)
ax.set_ylabel("% corridas seleccionada"); ax.set_title("Frecuencia de selección por feature")
ax.legend(); ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig(f"{fig_dir}/feature_frequency.png",dpi=300,bbox_inches="tight"); plt.close()

pso_f1=pso_df["f1_macro"].mean(); ba_f1=ba_df["f1_macro"].mean(); bl_f1=bl_df["f1_macro"].mean()
pso_rd=pso_df["RD"].mean();       ba_rd=ba_df["RD"].mean()
pso_tfp=pso_df["TFP"].mean();     ba_tfp=ba_df["TFP"].mean(); bl_tfp=bl_df["TFP"].mean()
pso_rec=pso_df["recall_atk"].mean(); ba_rec=ba_df["recall_atk"].mean(); bl_rec=bl_df["recall_atk"].mean()
pso_nf=pso_df["n_features"].mean();  ba_nf=ba_df["n_features"].mean()

with open("results/resumen_resultados.md","w",encoding="utf-8") as f:
    f.write("# Resultados del Experimento de Tesis\n\n")
    f.write(open("results/tables/resumen_metricas.md").read()+"\n\n")
    f.write("> *Nota: F9 usa train=500k (submuestra estratificada) por restricción de RAM.*\n")
    f.write("> *Baseline (F5) usa train=2.1M con n_jobs=-1.*\n\n")
    f.write("## Análisis de Hipótesis\n\n")
    h1p=1 if pso_f1>bl_f1 else 0; h1b=1 if ba_f1>bl_f1 else 0
    f.write(f"- **H1a** PSO mejora F1: {pso_f1:.4f} vs {bl_f1:.4f} → {'APOYADA' if h1p else 'RECHAZADA'} (Δ={pso_f1-bl_f1:+.4f})\n")
    f.write(f"- **H1b** BA  mejora F1: {ba_f1:.4f} vs {bl_f1:.4f} → {'APOYADA' if h1b else 'RECHAZADA'} (Δ={ba_f1-bl_f1:+.4f})\n")
    f.write(f"- **H2** Reduce TFP: PSO={pso_tfp:.4f} BA={ba_tfp:.4f} Baseline={bl_tfp:.4f}\n")
    f.write(f"- **H3** Reducción dimensional: PSO={pso_rd:.1f}% ({pso_nf:.1f}/{N_total} feat) BA={ba_rd:.1f}% ({ba_nf:.1f}/{N_total} feat)\n")
    f.write(f"- **H4** PSO vs BA difieren: ver estadisticas.csv\n\n")
    f.write("## Limitaciones\n- Etiquetas MAWILab heurísticas\n- Solo cabeceras IP/TCP\n")
    f.write("- Muestra 3M/19M en preprocesamiento (RAM)\n- RF proxy reducido en búsqueda\n")
    f.write("- PSO/BA binarios transfer sigmoide\n- 2021-02-10 PCAP parcial\n")

p("FASE 12 completa.")
p("="*55)
p("PIPELINE COMPLETO - RESULTADOS FINALES")
p(f"  Baseline (F5, 2.1M): F1={bl_f1:.4f} Rec={bl_rec:.4f} TFP={bl_tfp:.4f}")
p(f"  PSO (F9, 500k):      F1={pso_f1:.4f} Rec={pso_rec:.4f} TFP={pso_tfp:.4f} RD={pso_rd:.1f}% n={pso_nf:.1f}")
p(f"  BA  (F9, 500k):      F1={ba_f1:.4f} Rec={ba_rec:.4f} TFP={ba_tfp:.4f} RD={ba_rd:.1f}% n={ba_nf:.1f}")
p(f"Entregables: thesis_ids/results/")
p("="*55)
