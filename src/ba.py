"""
FASE 6: Bat Algorithm Binario para seleccion de caracteristicas.
"""
import numpy as np, os, json, time, logging
from wrapper import fitness

log = logging.getLogger(__name__)

def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))

def binary_bat(X_sub, y_sub, clf_params, cfg_bat, cfg_fit, run_id, out_dir, seed=42):
    rng   = np.random.default_rng(seed)
    pop   = cfg_bat["population"]
    iters = cfg_bat["iterations"]
    f_min = cfg_bat["f_min"]
    f_max = cfg_bat["f_max"]
    alpha = cfg_bat["alpha"]     # enfriamiento amplitud
    gamma = cfg_bat["gamma"]     # incremento tasa pulso
    A0    = cfg_bat["A0"]
    r0    = cfg_bat["r0"]
    fit_alpha = cfg_fit["alpha"]
    folds     = cfg_fit["cv_folds"]
    n = X_sub.shape[1]

    # Inicializar
    pos = rng.integers(0, 2, size=(pop, n)).astype(float)
    vel = np.zeros((pop, n))
    freq= np.zeros(pop)
    A   = np.full(pop, A0)
    r   = np.full(pop, r0)

    # Evaluar inicial
    fit_vals = np.full(pop, np.inf)
    for p in range(pop):
        fv,_,_ = fitness(pos[p], X_sub, y_sub, clf_params, fit_alpha, folds)
        fit_vals[p] = fv

    gbest_idx = int(np.argmin(fit_vals))
    gbest_pos = pos[gbest_idx].copy()
    gbest_fit = fit_vals[gbest_idx]

    convergence = []
    n_eval = pop
    t0 = time.time()

    for it in range(iters):
        for p in range(pop):
            freq[p]  = f_min + (f_max-f_min)*rng.random()
            vel[p]  += (pos[p] - gbest_pos)*freq[p]
            x_new    = pos[p] + vel[p]

            # Perturbacion local
            if rng.random() > r[p]:
                x_new = gbest_pos + 0.001*rng.standard_normal(n)

            # Binarizar
            prob  = sigmoid(x_new)
            x_bin = (rng.random(n) < prob).astype(float)

            # Evaluar
            fv_new, _, _ = fitness(x_bin, X_sub, y_sub, clf_params, fit_alpha, folds)
            n_eval += 1

            # Aceptar si mejora y amplitud lo permite
            if fv_new < fit_vals[p] and rng.random() < A[p]:
                pos[p]      = x_bin
                fit_vals[p] = fv_new
                A[p]       *= alpha
                r[p]        = r0*(1 - np.exp(-gamma*(it+1)))

            if fv_new < gbest_fit:
                gbest_fit = fv_new
                gbest_pos = x_bin.copy()

        convergence.append(float(gbest_fit))
        elapsed = time.time()-t0
        if (it+1) % 10 == 0 or it == iters-1:
            log.info(f"    BA  run={run_id} iter={it+1}/{iters} best_fit={gbest_fit:.5f} "
                     f"n_sel={int(gbest_pos.sum())} t={elapsed:.0f}s")

    result = {
        "run_id":      run_id,
        "seed":        seed,
        "algorithm":   "BA",
        "best_mask":   gbest_pos.tolist(),
        "best_fitness":gbest_fit,
        "n_selected":  int(gbest_pos.sum()),
        "convergence": convergence,
        "n_eval":      n_eval,
        "t_search_s":  time.time()-t0,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/ba_run_{run_id:03d}.json","w") as f:
        json.dump(result, f, indent=2)
    log.info(f"  BA  run {run_id} DONE: fit={gbest_fit:.5f} n_sel={int(gbest_pos.sum())} t={result['t_search_s']:.0f}s")
    return result
