import geopandas as gpd
import statsmodels.formula.api as smf
import patsy


FORMULA_M1 = """
log_price ~
beds + bathrooms + minimum_nights
+ instant_bookable + host_is_superhost + host_identity_verified
+ ln_host_listings + ln_num_reviews
+ review_scores_rating
+ host_response_rate
+ C(room_type)
"""

FORMULA_M2_LOC = """
log_price ~
beds + bathrooms + minimum_nights
+ instant_bookable + host_is_superhost + host_identity_verified
+ ln_host_listings + ln_num_reviews
+ review_scores_rating
+ host_response_rate
+ ln_pi_dist + ln_dist_subway + green_frac + robbery_rate_z
+ C(room_type)
"""


def freeze_sample(df1: gpd.GeoDataFrame):
    y_df, _ = patsy.dmatrices(FORMULA_M2_LOC, data=df1, return_type="dataframe")
    idx = y_df.index
    df_final = df1.loc[idx].copy()
    return df_final


def fit_ols(df_final: gpd.GeoDataFrame):
    m1 = smf.ols(FORMULA_M1, data=df_final).fit(cov_type="HC1")
    m2 = smf.ols(FORMULA_M2_LOC, data=df_final).fit(cov_type="HC1")
    return m1, m2

