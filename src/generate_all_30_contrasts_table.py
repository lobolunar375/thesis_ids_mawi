"""
generate_all_30_contrasts_table.py
Genera la tabla exhaustiva con los 30 contrastes estadisticos pareados,
incorporando:
- Media y Mediana de cada brazo
- Delta absoluta (Media A - Media B)
- Delta relativa porcentual
- Wilcoxon W, p-valor bruto, p-valor corregido por Holm
- Tamano del efecto Rank-Biserial r
- Matrices de confusion agregadas para fundamentar la hipotesis de TFP vs Recall
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, "src")

# Cargar las 30 corridas de cada brazo
bl_df   = pd.read_parquet("results/runs/baseline_500k_runs.parquet").sort_values("run").reset_index(drop=True)
rand_df = pd.read_parquet("results/runs/random_ctrl_runs.parquet").sort_values("run").reset_index(drop=True)
df_eval = pd.read_parquet("results/runs/eval_runs.parquet")
pso_df  = df_eval[df_eval["arm"]=="PSO"].sort_values("run").reset_index(drop=True)
ba_df   = df_eval[df_eval["arm"]=="BA"].sort_values("run").reset_index(drop=True)

metrics = ["f1_macro", "recall_atk", "accuracy", "TFP", "TFD"]
arms_dict = {"BL500k": bl_df, "RAND": rand_df, "PSO": pso_df, "BA": ba_df}
pairs = [
    ("PSO", "BL500k"),
    ("BA", "BL500k"),
    ("RAND", "BL500k"),
    ("PSO", "BA"),
    ("PSO", "RAND"),
    ("BA", "RAND"),
]

def rank_biserial_r(x, y):
    d = np.array(x) - np.array(y)
    d = d[d != 0]
    if len(d) == 0: return 0.0
    r = np.abs(d); ranks = stats.rankdata(r)
    T_plus  = ranks[d > 0].sum()
    T_minus = ranks[d < 0].sum()
    n = len(d)
    return float((T_plus - T_minus) / (n*(n+1)/2))

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

raw_rows = []
for a_name, b_name in pairs:
    df_a = arms_dict[a_name]
    df_b = arms_dict[b_name]
    for m in metrics:
        va = df_a[m].values[:30]
        vb = df_b[m].values[:30]
        
        stat, pv = stats.wilcoxon(va, vb, alternative="two-sided", zero_method="wilcox")
        rb_r = rank_biserial_r(va, vb)
        
        m_a = float(np.mean(va)); med_a = float(np.median(va)); s_a = float(np.std(va))
        m_b = float(np.mean(vb)); med_b = float(np.median(vb)); s_b = float(np.std(vb))
        
        delta_abs = m_a - m_b
        delta_pct = (delta_abs / m_b * 100) if m_b != 0 else 0.0
        
        raw_rows.append({
            "metric": m,
            "comparison": f"{a_name}_vs_{b_name}",
            "arm_A": a_name,
            "mean_A": m_a,
            "med_A": med_a,
            "std_A": s_a,
            "arm_B": b_name,
            "mean_B": m_b,
            "med_B": med_b,
            "std_B": s_b,
            "delta_abs": delta_abs,
            "delta_pct": delta_pct,
            "wilcoxon_W": float(stat),
            "p_raw": float(pv),
            "rb_r": float(rb_r),
        })

df_all = pd.DataFrame(raw_rows)
p_vals = df_all["p_raw"].values
reject, p_holm = multipletests_holm(p_vals)
df_all["p_holm"] = p_holm
df_all["sig_holm"] = reject

# Redondear para presentacion limpia
df_all_rounded = df_all.copy()
for col in ["mean_A", "med_A", "std_A", "mean_B", "med_B", "std_B", "delta_abs"]:
    df_all_rounded[col] = df_all_rounded[col].round(4)
df_all_rounded["delta_pct"] = df_all_rounded["delta_pct"].round(2)
df_all_rounded["rb_r"] = df_all_rounded["rb_r"].round(3)
df_all_rounded["p_raw"] = df_all_rounded["p_raw"].apply(lambda x: f"{x:.4e}" if x < 0.0001 else f"{x:.4f}")
df_all_rounded["p_holm"] = df_all_rounded["p_holm"].apply(lambda x: f"{x:.4e}" if x < 0.0001 else f"{x:.4f}")

os.makedirs("results/tables", exist_ok=True)
df_all.to_csv("results/tables/tabla_contrastes_30_exhaustiva.csv", index=False)
df_all_rounded.to_csv("results/tables/tabla_contrastes_30_redondeada.csv", index=False)

print(f"Generada tabla con {len(df_all)} contrastes pareados completos.")

# Matrices de confusion promedio y suma
cm_records = []
for name, d in arms_dict.items():
    tp = d["tp"].sum(); fn = d["fn"].sum(); tn = d["tn"].sum(); fp = d["fp"].sum()
    cm_records.append({
        "arm": name,
        "TP_sum": int(tp), "FN_sum": int(fn),
        "TN_sum": int(tn), "FP_sum": int(fp),
        "TP_mean": round(d["tp"].mean(), 1), "FN_mean": round(d["fn"].mean(), 1),
        "TN_mean": round(d["tn"].mean(), 1), "FP_mean": round(d["fp"].mean(), 1),
    })
df_cm = pd.DataFrame(cm_records)
df_cm.to_csv("results/tables/matrices_confusion_agregadas.csv", index=False)
print("\n=== MATRICES DE CONFUSION AGREGADAS (30 corridas) ===")
print(df_cm.to_string(index=False))
