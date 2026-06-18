from pathlib import Path
import torch
import json
import pickle
from loguru import logger
import sys
import pathlib

parent = pathlib.Path(__file__).parent.resolve()
with open(parent / "config.json", 'r') as f:
    CONFIG = dict(json.load(f))

def openpkl(file):
    logger.info(f"Opening file: {file}")
    with open(file, "rb") as f:
        data = pickle.load(f)
        logger.info(f"file size: {sys.getsizeof(data)} bytes")
        return data