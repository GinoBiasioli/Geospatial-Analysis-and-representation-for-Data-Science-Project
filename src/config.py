import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))

DATA_DIR = os.environ.get("BA_DATA_DIR", os.path.join(PROJECT_DIR, "Data"))
OUTPUT_DIR = os.environ.get("BA_OUTPUT_DIR", PROJECT_DIR)

METRIC_CRS = "EPSG:32721"   # UTM 21S (meters) for Buenos Aires
WGS84_CRS = "EPSG:4326"

PLOT_DIR = os.path.join(DATA_DIR, "_plots")
SAVE_PLOTS = False
MAKE_PLOTS = False
PLOT_SAMPLE_N = 15000

FIGSIZE = (9, 9)
DPI = 130

# Dataset / feature engineering parameters
YEAR_CRIME = 2024
PRICE_CAP_Q = 0.995
GREEN_CELL_SIZE_M = 800

# Spatial diagnostics defaults
K_GRID_DIAG = [10, 20, 30, 40, 60]
N_PERM_MORAN = 999

# Spatial models defaults
K_GRID_MODELS = [10, 20, 30, 40, 60, 80]
N_PERM_MODELS = 999

# Optional sampling for faster runs
USE_STRATIFIED_SAMPLE = True
SAMPLE_FRAC = 0.10
MIN_PER_GROUP = 5
RANDOM_STATE = 0
STRATA_COL = "neighbourhood_cleansed"
