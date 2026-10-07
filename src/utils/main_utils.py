import yaml
import os
from src.exception import *
from src.logger import *
from pathlib import Path

def read_config_file(path=None):
    try:
        yaml_file_path = Path(
            path if path is not None
            else Path(__file__).resolve().parents[1] / "config" / "config_file.yaml"
        )

        with open(yaml_file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if os.getenv("LEGALSAATHI_VECTORSTORE_PATH"):
            config["vectorstore"]["db_path"] = os.environ["LEGALSAATHI_VECTORSTORE_PATH"]

        logger.info(f"YAML loaded from {yaml_file_path}")
        return config

    except Exception as e:
        raise MyException(e, sys) from e
    

def ensure_path(path: str):
    """
    Ensures the parent directory of a file path exists.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
