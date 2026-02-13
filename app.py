import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import pydeck as pdk

st.set_page_config(page_title="BA Geo Dashboard", layout="wide")
st.title("Buenos Aires – Interactive layers")

MAP_PROVIDER = "carto"
MAP_STYLE = "dark"

APP_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(APP_DIR, "."))
DATA_DIR = os.path.join(PROJECT_DIR, "Data")

MAX_POINTS = 20000

SHOW_GREEN = False
SHOW_LISTINGS = True


@st.cache_data
def load_poi_points_4326() -> pd.DataFrame:
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
    poi_df = poi_df.rename(columns={"lng": "lon"})

    poi_df["lat"] = pd.to_numeric(poi_df["lat"], errors="coerce")
    poi_df["lon"] = pd.to_numeric(poi_df["lon"], errors="coerce")
    poi_df["poi_name"] = poi_df["poi_name"].astype(str)

    _ = gpd.GeoDataFrame(
        poi_df,
        geometry=gpd.points_from_xy(poi_df["lon"], poi_df["lat"]),
        crs="EPSG:4326",
    )

    return poi_df.dropna(subset=["lat", "lon"])


def _find_data_file(keywords_priority, exts=(".geojson", ".json", ".gpkg", ".parquet", ".csv")):
    if not os.path.isdir(DATA_DIR):
        return None

    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            lf = f.lower()
            if any(lf.endswith(ext) for ext in exts):
                all_files.append(os.path.join(root, f))

    def score(fp: str):
        name = os.path.basename(fp).lower()
        for rank, keys in enumerate(keywords_priority):
            if all(k in name for k in keys):
                return rank
        return None

    scored = []
    for fp in all_files:
        s = score(fp)
        if s is not None:
            scored.append((s, fp))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _extract_lon_lat_name(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}

    name_cands = ["name", "nombre", "station", "estacion", "estación", "stop_name", "title", "label"]
    name_col = next((cols[c] for c in name_cands if c in cols), None)

    lon_cands = ["lon", "lng", "long", "longitude", "x"]
    lat_cands = ["lat", "latitude", "y"]
    lon_col = next((cols[c] for c in lon_cands if c in cols), None)
    lat_col = next((cols[c] for c in lat_cands if c in cols), None)

    return lon_col, lat_col, name_col


@st.cache_data
def load_subway_points_4326() -> pd.DataFrame:
    fp = _find_data_file(
        keywords_priority=[
            ("subte",),
            ("subway",),
            ("metro",),
            ("station",),
            ("estacion",),
        ]
    )
    if fp is None:
        raise FileNotFoundError(
            "No subway stations file found in Data/. Add a GeoJSON/CSV with station points (name + lon/lat or geometry)."
        )

    ext = os.path.splitext(fp)[1].lower()
    if ext in (".geojson", ".json", ".gpkg"):
        g = gpd.read_file(fp)
        if g.crs is None:
            g = g.set_crs("EPSG:4326", allow_override=True)
        g = g.to_crs("EPSG:4326")

        geom = g.geometry
        if not g.geometry.geom_type.isin(["Point", "MultiPoint"]).any():
            geom = g.geometry.centroid

        out = pd.DataFrame({"lon": geom.x, "lat": geom.y})

        cols = {c.lower(): c for c in g.columns}
        name_col = None
        for cand in ["name", "nombre", "station", "estacion", "estación", "stop_name", "title", "label"]:
            if cand in cols:
                name_col = cols[cand]
                break
        out["name"] = g[name_col].astype(str) if name_col else "Station"
        return out.dropna(subset=["lon", "lat"])

    if ext == ".parquet":
        df = pd.read_parquet(fp)
    elif ext == ".csv":
        df = pd.read_csv(fp)
    else:
        raise ValueError(f"Unsupported stations file type: {ext}")

    lon_col, lat_col, name_col = _extract_lon_lat_name(df)
    if lon_col is None or lat_col is None:
        raise ValueError(f"Stations file must have lon/lat columns. Found: {list(df.columns)}")

    out = pd.DataFrame({
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "name": (df[name_col].astype(str) if name_col else "Station"),
    }).dropna(subset=["lon", "lat"])

    return out


@st.cache_data
def load_df_final() -> gpd.GeoDataFrame:
    fp = os.path.join(PROJECT_DIR, "df_final.parquet")
    gdf = gpd.read_parquet(fp)
    if gdf.geometry is not None and gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf


@st.cache_data
def load_comunas_4326() -> gpd.GeoDataFrame:
    fp = os.path.join(DATA_DIR, "caba_comunas.geojson")
    d = gpd.read_file(fp).to_crs("EPSG:4326")
    if "COMUNAS" in d.columns and "comuna" not in d.columns:
        d["comuna"] = d["COMUNAS"].astype(int)
    return d


@st.cache_data
def load_green_polys_4326() -> gpd.GeoDataFrame:
    fp = os.path.join(DATA_DIR, "espacio-verde-publico.geojson")
    g = gpd.read_file(fp).to_crs("EPSG:4326")
    if "clasificac" in g.columns:
        keep_classes = [
            "PLAZOLETA", "PARQUE", "PLAZA", "PATIO RECREATIVO", "PARQUE SEMIPÚBLICO",
            "PASEO", "PATIO DE JUEGOS INCLUSIVO", "JARDÍN BOTÁNICO"
        ]
        g = g[g["clasificac"].isin(keep_classes)].copy()
    return g


try:
    gdf = load_df_final()
except Exception as e:
    st.error("Could not load df_final.parquet")
    st.exception(e)
    st.stop()

required = {"latitude", "longitude", "price_num", "beds", "bathrooms", "dist_subway_m", "pi_dist"}
missing = [c for c in required if c not in gdf.columns]
if missing:
    st.error(f"df_final is missing required columns: {missing}")
    st.stop()

gdf["price_num"] = pd.to_numeric(gdf["price_num"], errors="coerce")
gdf["beds"] = pd.to_numeric(gdf["beds"], errors="coerce")
gdf["bathrooms"] = pd.to_numeric(gdf["bathrooms"], errors="coerce")
gdf["dist_subway_m"] = pd.to_numeric(gdf["dist_subway_m"], errors="coerce")
gdf["pi_dist"] = pd.to_numeric(gdf["pi_dist"], errors="coerce")

if "review_scores_rating" in gdf.columns:
    gdf["review_scores_rating"] = pd.to_numeric(gdf["review_scores_rating"], errors="coerce")

st.sidebar.header("Filters")

price_min = int(np.nanmin(gdf["price_num"]))
price_max = int(np.nanmax(gdf["price_num"]))

if not np.isfinite(price_min) or not np.isfinite(price_max) or price_min >= price_max:
    st.sidebar.warning("Price range is not valid; check price_num.")
    price_range = (price_min, price_max)
else:
    price_range = st.sidebar.slider(
        "Nightly price range (ARS)",
        min_value=int(price_min),
        max_value=int(price_max),
        value=(int(price_min), int(price_max)),
        step=max(1, int((price_max - price_min) / 200) if (price_max - price_min) > 0 else 1),
        format="%d ARS",
    )

st.sidebar.header("Review score")
rating_range = None
if "review_scores_rating" in gdf.columns:
    r = gdf["review_scores_rating"]
    if r.notna().any():
        rmin = float(np.nanmin(r.to_numpy(dtype=float)))
        rmax = float(np.nanmax(r.to_numpy(dtype=float)))
        if np.isfinite(rmin) and np.isfinite(rmax) and rmin < rmax:
            step = 1.0 if rmax > 10 else 0.1
            fmt = "%.0f" if step >= 1 else "%.1f"
            rating_range = st.sidebar.slider(
                "Review score rating",
                min_value=float(rmin),
                max_value=float(rmax),
                value=(float(rmin), float(rmax)),
                step=float(step),
                format=fmt,
            )
        else:
            st.sidebar.warning("Review score range is not valid; check review_scores_rating.")
    else:
        st.sidebar.info("No non-missing review scores found; rating filter disabled.")
else:
    st.sidebar.info("Column 'review_scores_rating' not found; rating filter disabled.")

st.sidebar.header("Bedrooms")
beds_opt = st.sidebar.selectbox("Bedrooms", ["(all)", "1", "2", "3", ">3"], index=0)

st.sidebar.header("Bathrooms")
baths_opt = st.sidebar.selectbox("Bathrooms", ["(all)", "1", "2", ">2"], index=0)

st.sidebar.header("Distance filters")
filter_poi = st.sidebar.checkbox("Only apartments within X meters of the nearest POI", False)
poi_max = int(np.nanmax(gdf["pi_dist"])) if np.isfinite(np.nanmax(gdf["pi_dist"])) else 0
poi_max = max(poi_max, 0)
poi_thresh = None
if filter_poi and poi_max > 0:
    poi_thresh = st.sidebar.slider(
        "Max distance to nearest POI (m)",
        min_value=0,
        max_value=poi_max,
        value=min(300, poi_max),
        step=10,
    )

filter_subway = st.sidebar.checkbox("Only apartments within X meters of the nearest subway station", False)
sub_max = int(np.nanmax(gdf["dist_subway_m"])) if np.isfinite(np.nanmax(gdf["dist_subway_m"])) else 0
sub_max = max(sub_max, 0)
sub_thresh = None
if filter_subway and sub_max > 0:
    sub_thresh = st.sidebar.slider(
        "Max distance to nearest subway (m)",
        min_value=0,
        max_value=sub_max,
        value=min(300, sub_max),
        step=10,
    )

st.sidebar.header("Overlays")
show_poi_points = st.sidebar.checkbox("Show points of interest", False)
show_subway_points = st.sidebar.checkbox("Show subway stations", False)

mask = gdf["price_num"].between(price_range[0], price_range[1])

if beds_opt != "(all)":
    if beds_opt == ">3":
        mask = mask & (gdf["beds"].fillna(-1) > 3)
    else:
        mask = mask & (gdf["beds"].fillna(-1) == int(beds_opt))

if baths_opt != "(all)":
    if baths_opt == ">2":
        mask = mask & (gdf["bathrooms"].fillna(-1) > 2)
    else:
        mask = mask & (gdf["bathrooms"].fillna(-1) == int(baths_opt))

if poi_thresh is not None:
    mask = mask & (gdf["pi_dist"].fillna(np.inf) <= poi_thresh)

if sub_thresh is not None:
    mask = mask & (gdf["dist_subway_m"].fillna(np.inf) <= sub_thresh)

if rating_range is not None:
    mask = mask & gdf["review_scores_rating"].between(rating_range[0], rating_range[1])

gdf_f = gdf.loc[mask].copy()

avg_price = float(np.nanmean(gdf_f["price_num"])) if len(gdf_f) else np.nan
avg_price_label = "—" if not np.isfinite(avg_price) else f"{int(round(avg_price)):,} ARS"

gdf_pts = gdf_f.sample(MAX_POINTS, random_state=0) if len(gdf_f) > MAX_POINTS else gdf_f

layers = []
view_state = pdk.ViewState(latitude=-34.6037, longitude=-58.3816, zoom=10.5, pitch=0)

try:
    comunas = load_comunas_4326()
    comunas_geojson = comunas.__geo_interface__
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            comunas_geojson,
            stroked=True,
            filled=True,
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[220, 220, 220, 220],
            line_width_min_pixels=1,
            pickable=False,
        )
    )
except Exception as e:
    st.warning("Failed to load/plot comunas")
    st.exception(e)

if SHOW_GREEN:
    try:
        green = load_green_polys_4326()
        green_geojson = green.__geo_interface__
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                green_geojson,
                stroked=False,
                filled=True,
                get_fill_color=[0, 255, 120, 140],
                pickable=False,
            )
        )
    except Exception as e:
        st.warning("Failed to load/plot green areas")
        st.exception(e)

poi_df = None
subway_df = None

if SHOW_LISTINGS:
    pts_df = pd.DataFrame({
        "lat": gdf_pts["latitude"].astype(float),
        "lon": gdf_pts["longitude"].astype(float),
        "price_num": pd.to_numeric(gdf_pts["price_num"], errors="coerce"),
        "dist_subway_m": pd.to_numeric(gdf_pts.get("dist_subway_m"), errors="coerce"),
        "pi_dist": pd.to_numeric(gdf_pts.get("pi_dist"), errors="coerce"),
    }).dropna(subset=["lat", "lon", "price_num"])

    pts_df["price_label"] = pts_df["price_num"].fillna(0).round(0).astype("int64").astype(str) + " ARS"
    pts_df["subway_label"] = pts_df["dist_subway_m"].fillna(0).round(0).astype("int64").astype(str) + " m"
    pts_df["poi_label"] = pts_df["pi_dist"].fillna(0).round(0).astype("int64").astype(str) + " m"

    pts_df["tooltip_html"] = (
        "<b>Apartment</b><br/>"
        "<b>Price:</b> " + pts_df["price_label"] + "<br/>"
        "<b>Subway distance:</b> " + pts_df["subway_label"] + "<br/>"
        "<b>POI distance:</b> " + pts_df["poi_label"]
    )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            pts_df,
            get_position="[lon, lat]",
            get_fill_color=[0, 200, 255, 200],
            get_radius=35,
            radius_min_pixels=2,
            radius_max_pixels=8,
            pickable=True,
        )
    )

if show_poi_points:
    poi_df = load_poi_points_4326().copy()
    poi_df["tooltip_html"] = poi_df["poi_name"].apply(lambda x: f"<b>Point of interest</b><br/>{x}")
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            poi_df,
            get_position="[lon, lat]",
            get_fill_color=[255, 165, 0, 220],
            get_radius=140,
            radius_min_pixels=7,
            radius_max_pixels=24,
            pickable=True,
        )
    )

if show_subway_points:
    try:
        subway_df = load_subway_points_4326().copy()
        subway_df["name"] = subway_df["name"].fillna("Station").astype(str)
        subway_df["tooltip_html"] = subway_df["name"].apply(lambda x: f"<b>Subway station</b><br/>{x}")
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                subway_df,
                get_position="[lon, lat]",
                get_fill_color=[255, 0, 120, 220],
                get_radius=160,
                radius_min_pixels=7,
                radius_max_pixels=26,
                pickable=True,
            )
        )
    except Exception as e:
        st.sidebar.warning("Could not load subway stations (missing file/columns).")
        st.sidebar.caption("If you want stations on the map, add a stations file into Data/ (geojson/csv).")
        st.exception(e)

tooltip = {"html": "{tooltip_html}", "style": {"backgroundColor": "white", "color": "black"}}

col_map, col_info = st.columns([2, 1])

with col_map:
    st.subheader("Interactive map")
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider=MAP_PROVIDER,
        map_style=MAP_STYLE,
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True, height=460)

with col_info:
    st.subheader("Summary")
    st.metric("Apartments filtered", f"{len(gdf_f):,}")
    st.metric("Average price", avg_price_label)

    st.divider()
    st.write(f"Points drawn (cap {MAX_POINTS:,}): {len(gdf_pts):,}")
    if show_poi_points and poi_df is not None:
        st.write(f"POIs shown: {len(poi_df):,}")
    if show_subway_points and subway_df is not None:
        st.write(f"Stations shown: {len(subway_df):,}")
    st.caption("Filters are applied from the sidebar.")

