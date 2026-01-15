#!/usr/bin/env python3
"""
Download COUGHVID Dataset from Kaggle
Prepares data for training the cough classifier

Usage:
    python scripts/download_coughvid.py --output data/coughvid

Requirements:
    - Kaggle API key: ~/.kaggle/kaggle.json
    - pip install kaggle
"""
import argparse
import logging
import os
import sys
from pathlib import Path
import shutil
import json
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        logger.error("Kaggle credentials not found!")
        logger.info("To set up Kaggle credentials:")
        logger.info("1. Go to https://www.kaggle.com/settings")
        logger.info("2. Click 'Create New Token' under API section")
        logger.info("3. Save kaggle.json to ~/.kaggle/")
        logger.info("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Check permissions
    mode = kaggle_json.stat().st_mode & 0o777
    if mode > 0o600:
        logger.warning(f"kaggle.json has insecure permissions ({oct(mode)})")
        logger.info("Run: chmod 600 ~/.kaggle/kaggle.json")
    
    return True


def download_from_kaggle(dataset: str, output_dir: Path):
    """Download dataset from Kaggle using the Kaggle API"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        logger.info(f"Downloading dataset: {dataset}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and unzip
        api.dataset_download_files(
            dataset,
            path=str(output_dir),
            unzip=True,
            quiet=False
        )
        
        logger.info("✓ Download complete!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download from Kaggle: {e}")
        return False


def organize_coughvid_data(data_dir: Path):
    """Organize COUGHVID data into train/val/test splits"""
    logger.info("Organizing COUGHVID data...")
    
    # Find audio files
    audio_files = list(data_dir.glob("**/*.webm")) + \
                  list(data_dir.glob("**/*.ogg")) + \
                  list(data_dir.glob("**/*.wav"))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Find metadata CSV
    metadata_files = list(data_dir.glob("**/metadata*.csv")) + \
                     list(data_dir.glob("**/labels*.csv"))
    
    if metadata_files:
        logger.info(f"Found metadata: {metadata_files[0]}")
    else:
        logger.warning("No metadata CSV found")
    
    # Create split directories
    for split in ["train", "val", "test"]:
        (data_dir / split).mkdir(exist_ok=True)
    
    # Parse metadata if available
    if metadata_files:
        try:
            import pandas as pd
            df = pd.read_csv(metadata_files[0])
            logger.info(f"Metadata columns: {list(df.columns)}")
            logger.info(f"Total samples in metadata: {len(df)}")
            
            # Look for label column
            label_cols = [c for c in df.columns if "label" in c.lower() or "cough" in c.lower() or "status" in c.lower()]
            if label_cols:
                logger.info(f"Label columns found: {label_cols}")
                logger.info(f"Label distribution:\n{df[label_cols[0]].value_counts()}")
                
        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")
    
    return len(audio_files)


def download_alternative_zenodo(output_dir: Path):
    """Download from Zenodo as alternative to Kaggle"""
    logger.info("Downloading from Zenodo...")
    
    zenodo_url = "https://zenodo.org/records/7024894/files/public_dataset.zip?download=1"
    
    try:
        # Use wget or curl
        zip_file = output_dir / "coughvid.zip"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        subprocess.run([
            "curl", "-L", "-o", str(zip_file), zenodo_url
        ], check=True)
        
        # Unzip
        subprocess.run([
            "unzip", "-o", str(zip_file), "-d", str(output_dir)
        ], check=True)
        
        # Remove zip
        zip_file.unlink()
        
        logger.info("✓ Downloaded from Zenodo")
        return True
        
    except Exception as e:
        logger.error(f"Zenodo download failed: {e}")
        return False


def create_sample_data(output_dir: Path):
    """Create sample synthetic data for testing if download fails"""
    logger.info("Creating sample synthetic data for testing...")
    
    import numpy as np
    import soundfile as sf
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample audio files
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    
    samples = {
        "dry_cough_001": {"freq": 400, "decay": 0.1},
        "dry_cough_002": {"freq": 500, "decay": 0.12},
        "wet_cough_001": {"freq": 200, "decay": 0.2},
        "wet_cough_002": {"freq": 180, "decay": 0.25},
        "normal_cough_001": {"freq": 300, "decay": 0.15},
    }
    
    for name, params in samples.items():
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate synthetic cough-like sound
        envelope = np.exp(-params["decay"] * t * 20) * (1 - np.exp(-t * 100))
        carrier = np.sin(2 * np.pi * params["freq"] * t)
        noise = np.random.randn(len(t)) * 0.2
        
        audio = envelope * (carrier + noise)
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save
        filepath = output_dir / f"{name}.wav"
        sf.write(filepath, audio, sample_rate)
        logger.info(f"Created: {filepath}")
    
    # Create metadata
    metadata = {
        "files": list(samples.keys()),
        "labels": {
            "dry_cough_001": "dry",
            "dry_cough_002": "dry",
            "wet_cough_001": "wet",
            "wet_cough_002": "wet",
            "normal_cough_001": "normal",
        },
        "note": "Synthetic data for testing only"
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✓ Created sample synthetic data")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download COUGHVID dataset")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/coughvid",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "zenodo", "synthetic"],
        default="kaggle",
        help="Data source"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="andradaolteanu/coughvid",
        help="Kaggle dataset name"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    logger.info("=" * 50)
    logger.info("COUGHVID Dataset Downloader")
    logger.info("=" * 50)
    
    success = False
    
    if args.source == "kaggle":
        if check_kaggle_credentials():
            success = download_from_kaggle(args.dataset, output_dir)
        
        if not success:
            logger.info("Trying Zenodo as fallback...")
            success = download_alternative_zenodo(output_dir)
    
    elif args.source == "zenodo":
        success = download_alternative_zenodo(output_dir)
    
    elif args.source == "synthetic":
        success = create_sample_data(output_dir)
    
    if not success:
        logger.warning("Real dataset download failed. Creating synthetic data...")
        success = create_sample_data(output_dir)
    
    if success:
        organize_coughvid_data(output_dir)
        logger.info("=" * 50)
        logger.info("✓ Dataset ready!")
        logger.info(f"  Location: {output_dir}")
        logger.info("=" * 50)
    else:
        logger.error("Failed to prepare dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()
