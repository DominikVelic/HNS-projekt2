

from pathlib import Path

DATASET_PATH = Path(__file__).parents[2] / "dataset"
PROJECT_ROOT = Path(__file__).resolve()
OUTPUT_PATH = PROJECT_ROOT / "vysledky"
DATA_PATH = PROJECT_ROOT.parents[2] / "data"
CMDF_PATH = DATA_PATH / "cmdf"
IMFD_PATH = DATA_PATH / "imfd"

