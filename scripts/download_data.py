#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
URL = "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip?download=1"
ZIP_PATH = RAW_DIR / "m5-forecasting-accuracy.zip"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if files already exist
    required_files = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
    if all((RAW_DIR / f).exists() for f in required_files):
        logger.info("All required raw data files already exist in data/raw/.")
        return

    logger.info(f"Downloading M5 dataset from Zenodo to {ZIP_PATH}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading M5 dataset") as t:
            urllib.request.urlretrieve(URL, filename=ZIP_PATH, reporthook=t.update_to)
        logger.success("Download complete!")
    except Exception as e:
        logger.error(f"Failed to download M5 dataset from Zenodo: {e}")
        logger.info("Please download the M5 dataset from Kaggle or another source and extract it to data/raw/")
        sys.exit(1)

    logger.info(f"Extracting zip file to {RAW_DIR}...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # List files in the zip to see structure
            namelist = zip_ref.namelist()
            logger.info(f"Zip contains {len(namelist)} files.")
            zip_ref.extractall(RAW_DIR)
        logger.success("Extraction complete!")
    except Exception as e:
        logger.error(f"Failed to extract zip file: {e}")
        sys.exit(1)

    # Clean up the zip file
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        logger.info("Cleaned up download ZIP archive.")

    # If the zip extracted into a nested folder, move files to RAW_DIR
    extracted_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    for d in extracted_dirs:
        for f in d.iterdir():
            f.rename(RAW_DIR / f.name)
        d.rmdir()
        logger.info(f"Moved files out of nested directory {d.name}")

    # Verify again
    missing = [f for f in required_files if not (RAW_DIR / f).exists()]
    if missing:
        logger.error(f"Extraction verification failed. Missing files: {missing}")
        sys.exit(1)
    
    logger.success("M5 Forecasting dataset is ready for ingestion!")

if __name__ == "__main__":
    download_data()
