import numpy as np
import pandas as pd
import geopandas as gpd

from libpysal.weights import KNN
from esda.moran import Moran
from spreg import OLS
import patsy

from .config import METRIC_CRS, K_GRID_DIAG, N_PERM_MORAN


def build_yX(formula: str, data: pd.DataFrame):
    y_df, X_df = patsy.dmatrices(formula, data=data, return_type="dataframe")
    return y_df, X_df


def lm_stat_p(obj):
    try:
        stat, p = obj
        return float(stat), float(p)
    except Exception:
        return (np.nan, np.nan)


def fit_spreg_ols(formula: str, data: gpd.GeoDataFrame, k_neighbors: int):
    y_df, X_df = build_yX(formula, data)

    g = data.loc[y_df.index].copy().to_crs(METRIC_CRS)
    w_sub = KNN.from_dataframe(g, k=k_neighbors)
    w_sub.transform = "R"

    y_arr = y_df.to_numpy()
    X_arr = X_df.to_numpy()

    ols = OLS(
        y_arr, X_arr,
        w=w_sub,
        spat_diag=True,
        moran=True,
        name_y=y_df.columns[0],
        name_x=list(X_df.columns),
    )
    return ols, y_df, X_df, g, w_sub


def run_diagnostics(df_final: gpd.GeoDataFrame, formula_base: str, formula_loc: str):
    y_loc_df, _ = build_yX(formula_loc, df_final)
    idx = y_loc_df.index
    if not idx.equals(df_final.index):
        raise RuntimeError("Frozen sample mismatch: patsy dropped/reordered rows unexpectedly.")

    gdf_diag = df_final.copy().to_crs(METRIC_CRS)
    y = y_loc_df.iloc[:, 0].to_numpy()

    rows = []
    for k in K_GRID_DIAG:
        w = KNN.from_dataframe(gdf_diag, k=k)
        w.transform = "R"

        mi_y = Moran(y, w, permutations=N_PERM_MORAN)

        ols1, _, _, _, w1 = fit_spreg_ols(formula_base, df_final, k)
        ols2, _, _, _, w2 = fit_spreg_ols(formula_loc,  df_final, k)

        mi_u1 = Moran(ols1.u.flatten(), w1, permutations=N_PERM_MORAN)
        mi_u2 = Moran(ols2.u.flatten(), w2, permutations=N_PERM_MORAN)

        lm_error_s, lm_error_p = lm_stat_p(getattr(ols2, "lm_error", None))
        lm_lag_s,   lm_lag_p   = lm_stat_p(getattr(ols2, "lm_lag", None))
        rlm_error_s, rlm_error_p = lm_stat_p(getattr(ols2, "rlm_error", None))
        rlm_lag_s,   rlm_lag_p   = lm_stat_p(getattr(ols2, "rlm_lag", None))
        lm_sarma_s,  lm_sarma_p  = lm_stat_p(getattr(ols2, "lm_sarma", None))

        rows.append({
            "k": k,
            "n_diag": int(w.n),
            "Moran_I_y": float(mi_y.I),
            "p_y_perm": float(mi_y.p_sim),
            "Moran_I_resid_base": float(mi_u1.I),
            "p_resid_base": float(mi_u1.p_sim),
            "Moran_I_resid_loc": float(mi_u2.I),
            "p_resid_loc": float(mi_u2.p_sim),
            "LM_Error_stat": lm_error_s,
            "LM_Error_p": lm_error_p,
            "LM_Lag_stat": lm_lag_s,
            "LM_Lag_p": lm_lag_p,
            "RLM_Error_stat": rlm_error_s,
            "RLM_Error_p": rlm_error_p,
            "RLM_Lag_stat": rlm_lag_s,
            "RLM_Lag_p": rlm_lag_p,
            "LM_SARMA_stat": lm_sarma_s,
            "LM_SARMA_p": lm_sarma_p,
        })

    return pd.DataFrame(rows)

