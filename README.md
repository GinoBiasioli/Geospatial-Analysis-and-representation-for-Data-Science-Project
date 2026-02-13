# Geospatial Analysis & Representation for Data Science — Airbnb Buenos Aires

This repository contains a geospatial data pipeline and econometric analysis for explaining differences in **nightly Airbnb prices across Buenos Aires**, with a focus on **location effects** (access to attractions/transport, environmental amenities, and neighborhood safety) and **spatial dependence**.

## Repository structure

```
.
├── app.py
├── .gitignore
├── conda_environment.txt
├── Data/
│   ├── listings.csv.gz
│   ├── neighbourhoods.geojson
│   ├── caba_comunas.geojson
│   ├── espacio-verde-publico.geojson
│   ├── estaciones.geojson
│   └── snic-departamentos-anual.csv
└── src/
    ├── main.py
    ├── config.py
    ├── build_dataset.py
    ├── models_ols.py
    ├── spatial_diagnostics.py
    ├── spatial_models.py
    └── plotting_utils.py
```

* **`Data/`**: raw input files (InsideAirbnb + open data layers).
* **`src/`**: analysis pipeline (data build → models → diagnostics → spatial models → plots).
* **`app.py`**: Streamlit dashboard (maps + filters).
* **`conda_environment.txt`**: full list of installed packages and versions exported via `conda list` for reproducibility.

## Requirements

### Python version

* Python **3.10+** (3.11 recommended)

### Library versions 

For full reproducibility, this repository includes:

* `conda_environment.txt`

This file documents the exact package versions used for estimation and can be used to recreate the environment when needed.

To recreate the environment manually:

```bash
conda create -n geo_airbnb python=3.11 -y
conda activate geo_airbnb

# Install packages according to conda_environment.txt
```

Alternatively, the core dependencies can be installed using the setup instructions below.

### Suggested environment setup (conda)

```bash
conda create -n geo_airbnb python=3.11 -y
conda activate geo_airbnb

# Core scientific stack
conda install -y -c conda-forge numpy pandas scipy scikit-learn

# Geospatial
conda install -y -c conda-forge geopandas shapely pyproj fiona rtree

# Spatial econometrics
conda install -y -c conda-forge pysal libpysal esda spreg

# Modeling + plotting
conda install -y -c conda-forge statsmodels patsy matplotlib

# App
conda install -y -c conda-forge streamlit folium streamlit-folium
```

## Data

All inputs are expected in the `Data/` folder (already included in this repository):

* `listings.csv.gz` (InsideAirbnb listings)
* `neighbourhoods.geojson` (neighbourhood polygons)
* `caba_comunas.geojson` (Buenos Aires comunas / districts)
* `espacio-verde-publico.geojson` (public green areas)
* `estaciones.geojson` (subway stations)
* `snic-departamentos-anual.csv` (crime data)

>

## How to run the analysis pipeline

The full pipeline is orchestrated from **`src/main.py`**.

From the repository root:

```bash
python -m src.main
```

If the module invocation fails (for example, if `src` is not treated as a module), run:

```bash
python src/main.py
```

### What the pipeline does (high level)

1. Builds a frozen analysis sample (cleans listings, applies caps/filters if configured).
2. Creates locational features (distances, green coverage, safety measures).
3. Estimates:

   * OLS baseline and OLS with locational controls
   * Spatial diagnostics (Moran’s I, LM tests, etc.)
   * Spatial models (SAR / SEM) for selected spatial weights
4. Exports tables and figures (locations defined in `src/config.py`).

### Configuration

The following settings are defined in `src/config.py`:

* file paths
* CRS used for distance calculations
* caps/filters (e.g., max price, min nights)
* model formula / variable selection
* kNN grid for spatial weights (if testing multiple k)
* output directories for figures/tables

## How to run the Streamlit dashboard

From the repository root:

```bash
streamlit run app.py
```

The local URL displayed in the terminal provides access to the dashboard.

### Notes

* The dashboard reads from the same `Data/` inputs and/or processed outputs defined in the configuration.

##

## Credits / references

This project is inspired by the approach used in:

* Chica-Olmo, J., González-Morales, J. G., & Zafra-Gómez, J. L. (2020). *Effects of location on Airbnb apartment pricing in Málaga*. Tourism Management, 77, 104017.

##
