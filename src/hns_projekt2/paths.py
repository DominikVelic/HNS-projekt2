

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve()
DATA_PATH = PROJECT_ROOT.parents[2] / "data"
CMDF_PATH = DATA_PATH / "cmdf"
IMFD_PATH = DATA_PATH / "imfd"


