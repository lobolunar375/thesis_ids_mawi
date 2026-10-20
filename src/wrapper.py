"""
Funcion de aptitud (fitness) compartida por PSO y BA.
fitness = alpha*(1-F1_macro_CV) + (1-alpha)*(N_sel/N_total)
Usa n_estimators_search (arboles reducidos) durante la busqueda para velocidad.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import yaml, json

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def fitness(mask, X_sub, y_sub, clf_params, alpha=0.95, cv_folds=2):
    """
    mask: array binario de len=n_features
    clf_params puede incluir 'n_estimators_search' para usar menos arboles
    Retorna: (valor_fitness, f1_cv, n_sel)
    """
    n_total = len(mask)
    idx = np.where(mask >= 0.5)[0]
    n_sel = len(idx)

    # Penalizar subconjunto vacio
    if n_sel == 0:
        return 1.0, 0.0, 0

    X_sel = X_sub[:, idx]

    # Usar n_estimators reducido para busqueda si disponible
    search_params = {k: v for k, v in clf_params.items() if k != "n_estimators_search"}
    if "n_estimators_search" in clf_params:
        search_params["n_estimators"] = clf_params["n_estimators_search"]

    clf = RandomForestClassifier(**search_params)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                          random_state=clf_params.get("random_state", 42))
    scores = cross_val_score(clf, X_sel, y_sub, cv=skf,
                             scoring="f1_macro", n_jobs=1)
    f1_cv   = float(scores.mean())
    fit_val = alpha * (1 - f1_cv) + (1 - alpha) * (n_sel / n_total)
    return fit_val, f1_cv, n_sel