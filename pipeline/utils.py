import os
import yaml
from pathlib import Path
from pydicom.misc import is_dicom


def load_yaml_config(file_path:str) -> dict:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def find_dicom_directories(root_path):
    dicom_dirs = set()
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = Path(os.sep.join([dirpath, filename])).as_posix()
            try:
                if is_dicom(file_path):
                    dicom_dirs.add(Path(dirpath).as_posix())
                    break  
            except Exception:
                continue 
    return dicom_dirs


def del_file(file_path:str, logger):
    try:
        os.remove(file_path)
        logger.debug(f"File {file_path} deleted successfully.")
    except OSError as e:
        logger.error(f"Error deleting file {file_path}: {e.strerror}")