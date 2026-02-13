import os
import geopandas as gpd

from .config import OUTPUT_DIR
from .build_dataset import build_df1
from .models_ols import freeze_sample, fit_ols, FORMULA_M1, FORMULA_M2_LOC
from .spatial_diagnostics import run_diagnostics
from .spatial_models import run_sem_sar_grid, extract_sar_impacts_block, print_model_headline


def main(run_spatial_diagnostics: bool = True, run_spatial_models: bool = True, k_pick: int = 20):
    df1, _ = build_df1()

    df_final = freeze_sample(df1)
    if not isinstance(df_final, gpd.GeoDataFrame):
        df_final = gpd.GeoDataFrame(df_final, geometry="geometry", crs=df1.crs)

    out_fp = os.path.join(OUTPUT_DIR, "df_final.parquet")
    df_final.to_parquet(out_fp, index=False)
    print("Saved:", out_fp)
    print("df_final shape:", df_final.shape)

    m1, m2 = fit_ols(df_final)
    print(m1.summary())
    print("Cond. No.:", m1.condition_number)
    print(m2.summary())
    print("Cond. No.:", m2.condition_number)

    if run_spatial_diagnostics:
        diag_df = run_diagnostics(df_final, FORMULA_M1, FORMULA_M2_LOC)
        print("\n--- Spatial diagnostics summary table (by k) ---")
        print(diag_df.round(4).to_string(index=False))

    if run_spatial_models:
        metrics_df, _, sem_by_k, sar_by_k = run_sem_sar_grid(df_final, FORMULA_M2_LOC)
        print("\n=== k comparison table (ordered by k) ===")
        print(metrics_df.sort_values("k").reset_index(drop=True).to_string(index=False))

        if k_pick in sar_by_k:
            sar_pick = sar_by_k[k_pick]
            sem_pick = sem_by_k[k_pick]
            print_model_headline(sem_pick, "SEM (ML_Error)")
            print_model_headline(sar_pick, "SAR (ML_Lag)")
            impacts = extract_sar_impacts_block(sar_pick)
            print("\n=== SAR IMPACTS (Direct / Indirect / Total) ===\n")
            print(impacts if impacts is not None else "No impacts block found in sar.summary")
        else:
            print(f"\nSelected k not found: {k_pick}. Available: {sorted(list(sar_by_k.keys()))}")


if __name__ == "__main__":
    main(run_spatial_diagnostics=True, run_spatial_models=True, k_pick=20)

