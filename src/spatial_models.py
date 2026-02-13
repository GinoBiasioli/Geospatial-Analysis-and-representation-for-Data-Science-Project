import time
import numpy as np
import pandas as pd
import geopandas as gpd
import patsy

from libpysal.weights import KNN
from spreg import ML_Error, ML_Lag
from esda.moran import Moran

from .config import METRIC_CRS, K_GRID_MODELS, N_PERM_MODELS


def aic_from_model(m, X_arr_fallback=None):
    ll = getattr(m, "logll", None)
    if ll is None:
        return None

    betas = getattr(m, "betas", None)
    if betas is not None:
        p_betas = int(betas.shape[0])
    elif X_arr_fallback is not None:
        p_betas = int(X_arr_fallback.shape[1])
    else:
        return None

    p = p_betas + 1
    return float(-2.0 * ll + 2.0 * p)


def extract_sar_impacts_block(sar_model):
    s = getattr(sar_model, "summary", "")
    if not isinstance(s, str) or "SPATIAL LAG MODEL IMPACTS" not in s:
        return None
    block = s.split("SPATIAL LAG MODEL IMPACTS", 1)[1]
    block = block.split("================================", 1)[0]
    return block.strip()


def _apply_spreg_summary_patch():
    try:
        import spreg.output as spout
        import spreg.ml_error as mlerr
        import spreg.ml_lag as mllag

        def _no_summary(*args, **kwargs):
            return ""

        # These functions are called during ML_Error / ML_Lag initialization
        spout._nonspat_top = _no_summary
        mlerr._nonspat_top = _no_summary
        mllag._nonspat_top = _no_summary

        # Some versions also call output_start
        if hasattr(spout, "output_start"):
            spout.output_start = _no_summary
        if hasattr(mlerr, "output_start"):
            mlerr.output_start = _no_summary
        if hasattr(mllag, "output_start"):
            mllag.output_start = _no_summary

    except Exception:
        # If patching fails, we just let spreg behave normally
        pass


_apply_spreg_summary_patch()


def run_sem_sar_grid(df_final: gpd.GeoDataFrame, formula_loc: str):
    y_df, X_df = patsy.dmatrices(formula_loc, data=df_final, return_type="dataframe")
    if "Intercept" in X_df.columns:
        X_df = X_df.drop(columns=["Intercept"])

    y_arr = np.asarray(y_df, dtype=float).reshape(-1, 1)
    X_arr = np.asarray(X_df, dtype=float)

    g_metric = df_final.to_crs(METRIC_CRS)

    w_by_k = {}
    sem_by_k = {}
    sar_by_k = {}
    rows = []

    np.random.seed(123)

    for k in K_GRID_MODELS:
        t0 = time.time()
        w = KNN.from_dataframe(g_metric, k=k)
        w.transform = "R"
        t_w = time.time() - t0

        t0 = time.time()
        sem = ML_Error(
            y_arr, X_arr,
            w=w,
            name_y=y_df.columns[0],
            name_x=list(X_df.columns),
        )
        t_sem = time.time() - t0

        t0 = time.time()
        sar = ML_Lag(
            y_arr, X_arr,
            w=w,
            name_y=y_df.columns[0],
            name_x=list(X_df.columns),
        )
        t_sar = time.time() - t0

        mi_sem = Moran(sem.u.flatten(), w, permutations=N_PERM_MODELS)
        mi_sar = Moran(sar.u.flatten(), w, permutations=N_PERM_MODELS)

        sem_aic = aic_from_model(sem, X_arr_fallback=X_arr)
        sar_aic = aic_from_model(sar, X_arr_fallback=X_arr)

        w_by_k[k] = w
        sem_by_k[k] = sem
        sar_by_k[k] = sar

        rows.append({
            "k": k,
            "SEM_lambda": float(getattr(sem, "lam", np.nan)),
            "SEM_logLik": float(getattr(sem, "logll", np.nan)),
            "SEM_AIC": sem_aic,
            "SEM_Moran_I": float(mi_sem.I),
            "SEM_Moran_p": float(mi_sem.p_sim),
            "SAR_rho": float(getattr(sar, "rho", np.nan)),
            "SAR_logLik": float(getattr(sar, "logll", np.nan)),
            "SAR_AIC": sar_aic,
            "SAR_Moran_I": float(mi_sar.I),
            "SAR_Moran_p": float(mi_sar.p_sim),
            "time_W_s": round(t_w, 3),
            "time_SEM_s": round(t_sem, 3),
            "time_SAR_s": round(t_sar, 3),
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df["SAR_clean"] = metrics_df["SAR_Moran_p"] > 0.05

    return metrics_df, w_by_k, sem_by_k, sar_by_k


def print_model_headline(m, name="MODEL"):
    ll = getattr(m, "logll", None)
    aic = aic_from_model(m)
    print(f"\n=== {name} headline ===")
    if hasattr(m, "rho"):
        print("rho:", round(float(m.rho), 6))
    if hasattr(m, "lam"):
        print("lambda:", round(float(m.lam), 6))
    print("logLik:", None if ll is None else round(float(ll), 6))
    print("AIC:", None if aic is None else round(float(aic), 6))

