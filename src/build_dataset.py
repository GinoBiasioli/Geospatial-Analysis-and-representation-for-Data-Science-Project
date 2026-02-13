import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box


from .config import (
    DATA_DIR, WGS84_CRS, METRIC_CRS,
    MAKE_PLOTS, PLOT_SAMPLE_N,
    USE_STRATIFIED_SAMPLE, SAMPLE_FRAC, MIN_PER_GROUP, RANDOM_STATE, STRATA_COL,
    YEAR_CRIME, PRICE_CAP_Q, GREEN_CELL_SIZE_M,
)
from .plotting_utils import new_fig_ax, finish_map, show_or_save, annotate_polygons


def p(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def load_core():
    listings = pd.read_csv(p("listings.csv.gz"), low_memory=False)
    neigh = gpd.read_file(p("neighbourhoods.geojson"))
    districts = gpd.read_file(p("caba_comunas.geojson"))
    return listings, neigh, districts


def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    if not USE_STRATIFIED_SAMPLE:
        return df.copy()

    if STRATA_COL not in df.columns:
        raise ValueError(f"Strata column not found: {STRATA_COL}")

    df_full = df.copy()
    idx_keep = []
    for _, g in df_full.dropna(subset=[STRATA_COL]).groupby(STRATA_COL):
        n = len(g)
        take = max(int(np.floor(n * SAMPLE_FRAC)), MIN_PER_GROUP)
        take = min(take, n)
        idx_keep.append(g.sample(n=take, random_state=RANDOM_STATE).index)

    idx_keep = np.concatenate([ix.to_numpy() for ix in idx_keep])
    out = df_full.loc[idx_keep].copy()

    if STRATA_COL not in out.columns:
        raise RuntimeError("Sampling error: stratification column disappeared.")
    return out


def clean_listings(listings: pd.DataFrame) -> pd.DataFrame:
    df = listings.copy()
    df = stratified_sample(df)

    if "id" not in df.columns:
        raise ValueError("No `id` column found in listings.")
    df["listing_id"] = df["id"].astype("int64")

    df["price_num"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .replace("nan", np.nan)
        .astype(float)
    )
    df = df[df["price_num"].notna() & (df["price_num"] > 0)].copy()
    df["log_price"] = np.log(df["price_num"])

    keep = [
        "listing_id",
        "price_num",
        "log_price",
        "latitude", "longitude",
        "neighbourhood_cleansed",
        "room_type", "property_type",
        "accommodates", "bedrooms", "beds", "bathrooms",
        "minimum_nights",
        "instant_bookable", "host_is_superhost", "host_identity_verified",
        "host_total_listings_count",
        "number_of_reviews", "review_scores_rating",
        "host_response_rate",
    ]
    dfm = df[keep].copy()

    dfm = dfm[dfm["property_type"].astype(str) == "Entire rental unit"].copy()

    dfm["host_response_rate"] = (
        dfm["host_response_rate"].astype(str)
        .str.replace("%", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )

    for col in ["instant_bookable", "host_is_superhost", "host_identity_verified"]:
        dfm[col] = dfm[col].map({"t": 1, "f": 0}).astype("float")

    num_cols = [
        "accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights",
        "host_total_listings_count", "number_of_reviews", "review_scores_rating",
    ]
    for c in num_cols:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

    dfm = dfm.dropna(subset=["log_price", "latitude", "longitude", "room_type", "beds"]).copy()
    return dfm


def make_base_gdf(dfm: pd.DataFrame, neigh: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf_base = gpd.GeoDataFrame(
        dfm,
        geometry=gpd.points_from_xy(dfm["longitude"], dfm["latitude"]),
        crs=WGS84_CRS,
    )

    neigh = neigh.to_crs(gdf_base.crs)
    gdf_base = gpd.sjoin(
        gdf_base,
        neigh[["neighbourhood", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    gdf_base = gdf_base.rename(columns={"neighbourhood": "neigh_poly"})
    gdf_base = gdf_base.drop(columns=["neighbourhood_cleansed"], errors="ignore")

    for c in ["neigh_poly", "property_type", "room_type"]:
        if c in gdf_base.columns:
            gdf_base[c] = gdf_base[c].astype("category")

    return gdf_base


def prepare_districts(districts: gpd.GeoDataFrame):
    if districts.crs is None:
        districts = districts.set_crs(WGS84_CRS, allow_override=True)

    if "COMUNAS" in districts.columns:
        districts["comuna"] = districts["COMUNAS"].astype(int)
    elif "comuna" not in districts.columns:
        raise ValueError("Expected 'COMUNAS' or 'comuna' column not found in comuna polygons.")

    districts_m = districts.to_crs(METRIC_CRS)
    aoi_m = districts_m.dissolve()
    aoi_m = aoi_m[aoi_m.geometry.notna()].copy()

    districts_4326 = districts.to_crs(WGS84_CRS)
    aoi_4326 = aoi_m.to_crs(WGS84_CRS)
    return districts, districts_m, aoi_m, districts_4326, aoi_4326


def join_comunas(gdf_base: gpd.GeoDataFrame, districts_4326: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    districts_4326 = districts_4326.to_crs(gdf_base.crs)
    gdf = gpd.sjoin(
        gdf_base,
        districts_4326[["comuna", "BARRIOS", "PERIMETRO", "AREA", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    gdf["comuna_label"] = "Comuna " + gdf["comuna"].astype("Int64").astype(str)
    return gdf


def compute_robbery_rate() -> pd.DataFrame:
    path_snic = p("snic-departamentos-anual.csv")
    try:
        snic = pd.read_csv(path_snic, sep=";", encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        snic = pd.read_csv(path_snic, sep="\t", encoding="latin1", engine="python")

    snic2 = snic.copy()
    snic2["provincia_id"] = snic2["provincia_id"].astype(str).str.zfill(2)
    snic_caba = snic2[snic2["provincia_id"] == "02"].copy()

    snic_caba["departamento_id"] = snic_caba["departamento_id"].astype(str)
    snic_caba = snic_caba[~snic_caba["departamento_id"].str.endswith("999")].copy()

    snic_caba_y = snic_caba[snic_caba["anio"] == YEAR_CRIME].copy()
    snic_caba_y["cod_delito"] = snic_caba_y["cod_delito"].astype(str)
    snic_caba_y_base = snic_caba_y[~snic_caba_y["cod_delito"].str.contains("_", na=False)].copy()

    s = snic_caba_y_base.copy()
    s["cod_delito_num"] = pd.to_numeric(s["cod_delito"], errors="coerce")
    s15 = s[s["cod_delito_num"] == 15].copy()  # robbery

    s15_rate = s15[["departamento_nombre", "tasa_hechos"]].rename(
        columns={"departamento_nombre": "comuna_label", "tasa_hechos": "robbery_rate"}
    )

    mu = s15_rate["robbery_rate"].mean(skipna=True)
    sd = s15_rate["robbery_rate"].std(skipna=True, ddof=0)
    if pd.isna(sd) or sd == 0:
        s15_rate["robbery_rate_z"] = 0.0
    else:
        s15_rate["robbery_rate_z"] = (s15_rate["robbery_rate"] - mu) / sd

    # Make comuna_label match the listing side ("Comuna X")
    s15_rate["comuna_label"] = s15_rate["comuna_label"].astype(str).str.strip()
    if not s15_rate["comuna_label"].str.lower().str.startswith("comuna").any():
        s15_rate["comuna_label"] = "Comuna " + s15_rate["comuna_label"]
    return s15_rate


def add_robbery(gdf_comuna: gpd.GeoDataFrame, robbery_df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gdf_comuna.merge(robbery_df, on="comuna_label", how="left")


def load_green_filtered() -> gpd.GeoDataFrame:
    fp = p("espacio-verde-publico.geojson")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Green areas GeoJSON not found at: {fp}")

    green_areas = gpd.read_file(fp, engine="pyogrio")
    keep_classes = [
        "PLAZOLETA", "PARQUE", "PLAZA", "PATIO RECREATIVO", "PARQUE SEMIPÚBLICO",
        "PASEO", "PATIO DE JUEGOS INCLUSIVO", "JARDÍN BOTÁNICO",
    ]
    green_filt = green_areas[green_areas["clasificac"].isin(keep_classes)].copy()
    return green_filt


def build_green_grid(aoi_m: gpd.GeoDataFrame, green_filt: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    green_m = green_filt.to_crs(METRIC_CRS).copy()

    minx, miny, maxx, maxy = aoi_m.total_bounds
    cell = float(GREEN_CELL_SIZE_M)
    xs = np.arange(minx, maxx, cell)
    ys = np.arange(miny, maxy, cell)

    polys, ids = [], []
    k = 0
    for x in xs:
        for y in ys:
            polys.append(box(x, y, x + cell, y + cell))
            ids.append(k)
            k += 1

    grid = gpd.GeoDataFrame({"cell_id": ids, "geometry": polys}, crs=METRIC_CRS)
    grid = gpd.clip(grid, aoi_m)
    grid = grid[grid.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    grid["cell_area"] = grid.area

    pairs = (
        gpd.sjoin(
            grid[["cell_id", "geometry"]],
            green_m[["geometry"]],
            predicate="intersects",
            how="left",
        )
        .dropna(subset=["index_right"])
        .copy()
    )

    pairs["index_right"] = pairs["index_right"].astype(int)
    pairs = pairs.join(green_m.geometry.rename("green_geom"), on="index_right")
    pairs["green_area"] = pairs.geometry.intersection(pairs["green_geom"]).area

    green_by_cell = pairs.groupby("cell_id")["green_area"].sum()
    grid["green_area"] = grid["cell_id"].map(green_by_cell).fillna(0.0)
    grid["green_frac"] = grid["green_area"] / grid["cell_area"]
    return grid


def assign_green_frac(gdf: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    listings_m = gdf.to_crs(grid.crs)
    grid_small = grid[["cell_id", "green_frac", "geometry"]].copy()

    listings_m = gpd.sjoin(listings_m, grid_small, how="left", predicate="within")
    miss = listings_m["green_frac"].isna()
    if miss.any():
        tmp = gpd.sjoin(
            listings_m[miss].drop(columns=["index_right"], errors="ignore"),
            grid_small,
            how="left",
            predicate="intersects",
        )
        listings_m.loc[miss, "green_frac"] = tmp["green_frac"].values
        listings_m.loc[miss, "cell_id"] = tmp["cell_id"].values

    listings_m = listings_m.drop(columns=["index_right"], errors="ignore")
    return listings_m.to_crs(WGS84_CRS)


def load_subway_stations_active() -> gpd.GeoDataFrame:
    fp = p("estaciones.geojson")
    stations = gpd.read_file(fp)
    active = stations[stations["closure"].isna() | (stations["closure"].astype(str).str.strip() == "")].copy()
    if active.crs is None:
        active = active.set_crs(WGS84_CRS, allow_override=True)
    return active


def add_subway_distance(gdf_4326: gpd.GeoDataFrame, stations_active: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    listings_m = gdf_4326.to_crs(METRIC_CRS)
    stations_m = stations_active.to_crs(METRIC_CRS)

    listings_m = (
        gpd.sjoin_nearest(
            listings_m,
            stations_m[["id", "name", "line", "geometry"]],
            how="left",
            distance_col="dist_subway_m",
        )
        .drop(columns=["index_right"], errors="ignore")
    )
    return listings_m.to_crs(WGS84_CRS)


def poi_geodataframe() -> gpd.GeoDataFrame:
    pois_json = [
        {"name": "Obelisco / Av. 9 de Julio", "lat": -34.603702, "lng": -58.381873},
        {"name": "Plaza de Mayo (Casa Rosada, Cathedral, Cabildo)", "lat": -34.608383, "lng": -58.372395},
        {"name": "Teatro Colón", "lat": -34.601086, "lng": -58.383187},
        {"name": "Puerto Madero (Puente de la Mujer / Docks area)", "lat": -34.599600, "lng": -58.362900},
        {"name": "La Boca (Caminito / Vuelta de Rocha area)", "lat": -34.634828, "lng": -58.363467},
        {"name": "Feria de San Telmo (Plaza Dorrego area)", "lat": -34.620468, "lng": -58.371799},
        {"name": "Recoleta Cemetery", "lat": -34.588056, "lng": -58.393056},
        {"name": "MALBA (Latin American Art Museum of Buenos Aires)", "lat": -34.573958, "lng": -58.403114},
        {"name": "Museo Nacional de Bellas Artes", "lat": -34.583988, "lng": -58.393112},
        {"name": "Bosques de Palermo (main park area)", "lat": -34.571100, "lng": -58.427500},
        {"name": "Planetario Galileo Galilei", "lat": -34.569722, "lng": -58.411667},
        {"name": "Barrio Chino (Chinatown, Belgrano)", "lat": -34.562500, "lng": -58.458333},
        {"name": "Caminito (La Boca)", "lat": -34.635000, "lng": -58.363000},
        {"name": "Estadio Monumental (River Plate)", "lat": -34.545280, "lng": -58.449720},
    ]

    poi_df = pd.DataFrame(pois_json).rename(columns={"name": "poi_name"})
    return gpd.GeoDataFrame(
        poi_df,
        geometry=gpd.points_from_xy(poi_df["lng"], poi_df["lat"]),
        crs=WGS84_CRS,
    )


def add_poi_distance(gdf_4326: gpd.GeoDataFrame, poi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    listings_m = gdf_4326.to_crs(METRIC_CRS)
    poi_m = poi_gdf.to_crs(METRIC_CRS)

    listings_m = (
        gpd.sjoin_nearest(
            listings_m,
            poi_m[["poi_name", "geometry"]],
            how="left",
            distance_col="pi_dist",
        )
        .drop(columns=["index_right"], errors="ignore")
        .rename(columns={"poi_name": "nearest_poi"})
    )
    return listings_m.to_crs(WGS84_CRS)


def finalize_master(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf_master = gdf.copy()

    for needed in ["robbery_rate_z", "green_frac", "dist_subway_m", "pi_dist"]:
        if needed not in gdf_master.columns:
            raise RuntimeError(f"Missing required feature: {needed}")

    gdf_master["price_num"] = pd.to_numeric(gdf_master["price_num"], errors="coerce")
    gdf_master["beds"] = pd.to_numeric(gdf_master["beds"], errors="coerce")

    cap = gdf_master["price_num"].quantile(PRICE_CAP_Q)
    gdf_master = gdf_master[gdf_master["price_num"] <= cap].copy()
    gdf_master["log_price"] = np.log(gdf_master["price_num"])

    gdf_master["ln_pi_dist"] = np.log1p(gdf_master["pi_dist"])
    gdf_master["ln_dist_subway"] = np.log1p(gdf_master["dist_subway_m"])
    gdf_master["ln_host_listings"] = np.log1p(gdf_master["host_total_listings_count"])
    gdf_master["ln_num_reviews"] = np.log1p(gdf_master["number_of_reviews"])

    return gdf_master


def apply_missingness_strategy(gdf_master: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf_ready = gdf_master.copy()

    na_cols = [
        "review_scores_rating",
        "host_response_rate",
        "host_is_superhost",
        "bedrooms", "bathrooms", "beds",
        "host_identity_verified", "host_total_listings_count",
    ]
    for c in na_cols:
        if c in gdf_ready.columns:
            gdf_ready[c + "_isna"] = gdf_ready[c].isna().astype(int)

    gdf_ready["review_scores_rating"] = gdf_ready["review_scores_rating"].fillna(0)

    med_rr = gdf_ready["host_response_rate"].median(skipna=True)
    gdf_ready["host_response_rate"] = gdf_ready["host_response_rate"].fillna(med_rr)

    gdf_ready["host_is_superhost"] = gdf_ready["host_is_superhost"].fillna(0)

    for c in ["bedrooms", "bathrooms", "beds"]:
        med = gdf_ready[c].median(skipna=True)
        gdf_ready[c] = gdf_ready[c].fillna(med)

    gdf_ready["host_identity_verified"] = gdf_ready["host_identity_verified"].fillna(1)

    gdf_ready["host_total_listings_count"] = gdf_ready["host_total_listings_count"].fillna(
        gdf_ready["host_total_listings_count"].median(skipna=True)
    )

    df1 = gdf_ready.copy()
    df1 = df1[df1["log_price"].notna() & df1["beds"].notna() & (df1["beds"] > 0)].copy()

    df1["property_type_simpl"] = "Entire rental unit"
    df1["property_type_simpl"] = df1["property_type_simpl"].astype("category")
    return df1


def optional_plots(gdf_base, districts_4326, aoi_4326, neigh):
    if not MAKE_PLOTS:
        return

    fig, ax = new_fig_ax()
    finish_map(ax, title="Buenos Aires neighborhoods", aoi_4326=aoi_4326, districts_4326=districts_4326)
    if "comuna" in districts_4326.columns:
        annotate_polygons(districts_4326, "comuna", ax=ax, fontsize=8)
    show_or_save(fig, "map_comunas_labels")

    fig, ax = new_fig_ax()
    finish_map(ax, title="Airbnb apartments over Buenos Aires districts", aoi_4326=aoi_4326, districts_4326=districts_4326)
    gdf_plot = gdf_base.copy()
    if len(gdf_plot) > PLOT_SAMPLE_N:
        gdf_plot = gdf_plot.sample(PLOT_SAMPLE_N, random_state=0)
    gdf_plot.plot(ax=ax, markersize=3, alpha=0.25)
    show_or_save(fig, "map_listings_points")

    fig, ax = new_fig_ax()
    finish_map(ax, title="Neighbourhood polygons (InsideAirbnb) + listings (sample)", aoi_4326=aoi_4326, districts_4326=districts_4326)
    try:
        neigh_4326 = neigh.to_crs(WGS84_CRS)
        neigh_4326.plot(ax=ax, facecolor="none", edgecolor="0.7", linewidth=0.6)
    except Exception:
        pass
    gdf_plot.plot(ax=ax, markersize=2, alpha=0.20)
    show_or_save(fig, "map_neighbourhoods_plus_listings")


def build_df1():
    listings, neigh, districts = load_core()
    dfm = clean_listings(listings)
    gdf_base = make_base_gdf(dfm, neigh)

    districts, districts_m, aoi_m, districts_4326, aoi_4326 = prepare_districts(districts)
    optional_plots(gdf_base, districts_4326, aoi_4326, neigh)

    gdf_comuna = join_comunas(gdf_base, districts_4326)

    robbery_df = compute_robbery_rate()
    gdf_comuna = add_robbery(gdf_comuna, robbery_df)

    green_filt = load_green_filtered()
    grid = build_green_grid(aoi_m, green_filt)
    gdf_green = assign_green_frac(gdf_comuna, grid)

    stations_active = load_subway_stations_active()
    gdf_subway = add_subway_distance(gdf_green, stations_active)

    poi_gdf = poi_geodataframe()
    gdf_poi = add_poi_distance(gdf_subway, poi_gdf)

    gdf_master = finalize_master(gdf_poi)
    df1 = apply_missingness_strategy(gdf_master)

    return df1, gdf_master
