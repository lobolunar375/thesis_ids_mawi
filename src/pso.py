"""
FASE 6: PSO Binario para seleccion de caracteristicas.
Binarizacion con funcion sigmoide (transfer en V disponible).
"""
import numpy as np, os, json, time, logging
from wrapper import fitness

log = logging.getLogger(__name__)

def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))

def binary_pso(X_sub, y_sub, clf_params, cfg_pso, cfg_fit, run_id, out_dir, seed=42):
    rng  = np.random.default_rng(seed)
    pop  = cfg_pso["population"]
    iters= cfg_pso["iterations"]
    w    = cfg_pso["w"]
    c1   = cfg_pso["c1"]
    c2   = cfg_pso["c2"]
    vmax = cfg_pso["vmax"]
    alpha= cfg_fit["alpha"]
    folds= cfg_fit["cv_folds"]
    n    = X_sub.shape[1]

    # Inicializar posiciones y velocidades
    pos = rng.integers(0, 2, size=(pop, n)).astype(float)
    vel = rng.uniform(-vmax, vmax, size=(pop, n))

    pbest_pos = pos.copy()
    pbest_fit = np.full(pop, np.inf)
    gbest_pos = pos[0].copy()
    gbest_fit = np.inf

    convergence = []
    n_eval = 0
    t0 = time.time()

    for it in range(iters):
        for p in range(pop):
            fv, f1cv, nsel = fitness(pos[p], X_sub, y_sub, clf_params, alpha, folds)
            n_eval += 1
            if fv < pbest_fit[p]:
                pbest_fit[p] = fv
                pbest_pos[p] = pos[p].copy()
            if fv < gbest_fit:
                gbest_fit = fv
                gbest_pos = pos[p].copy()

        # Actualizar velocidad y posicion
        r1 = rng.random((pop, n))
        r2 = rng.random((pop, n))
        vel = (w*vel
               + c1*r1*(pbest_pos - pos)
               + c2*r2*(gbest_pos - pos))
        vel = np.clip(vel, -vmax, vmax)

        prob = sigmoid(vel)
        pos  = (rng.random((pop, n)) < prob).astype(float)

        convergence.append(float(gbest_fit))
        elapsed = time.time()-t0
        if (it+1) % 10 == 0 or it == iters-1:
            log.info(f"    PSO run={run_id} iter={it+1}/{iters} best_fit={gbest_fit:.5f} "
                     f"n_sel={int(gbest_pos.sum())} t={elapsed:.0f}s")

    # Guardar resultado
    result = {
        "run_id":      run_id,
        "seed":        seed,
        "algorithm":   "PSO",
        "best_mask":   gbest_pos.tolist(),
        "best_fitness":gbest_fit,
        "n_selected":  int(gbest_pos.sum()),
        "convergence": convergence,
        "n_eval":      n_eval,
        "t_search_s":  time.time()-t0,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/pso_run_{run_id:03d}.json","w") as f:
        json.dump(result, f, indent=2)
    log.info(f"  PSO run {run_id} DONE: fit={gbest_fit:.5f} n_sel={int(gbest_pos.sum())} t={result['t_search_s']:.0f}s")
    return result
