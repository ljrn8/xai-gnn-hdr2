from pathlib import Path
import torch
import json
import pickle
from loguru import logger
import sys
import pathlib



def openpkl(file):
    logger.info(f"Opening file: {file}")
    with open(file, "rb") as f:
        data = pickle.load(f)
        logger.info(f"file size: {sys.getsizeof(data)} bytes")
        return data


def savepkl(data, file):
    logger.info(f"Saving file: {file}")
    with open(file, "wb") as f:
        logger.info(f"file size: {sys.getsizeof(data)} bytes")
        pickle.dump(data, f)
