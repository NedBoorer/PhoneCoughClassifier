#!/usr/bin/env python3
"""
Model Download Script for PhoneCoughClassifier.

Downloads required pretrained model weights for all classifiers.
Run this script to ensure all models work out of the box.

Usage:
    python scripts/download_models.py
"""
import os
import sys
import logging
from pathlib import Path
import urllib.request
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Model configuration
MODELS = {
    "panns_cnn6": {
        "name": "PANNs CNN6 (Respiratory)",
        "url": "https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth",
        "path": "external_models/respiratory_panns/panns/Cnn6_mAP=0.343.pth",
        "md5": None,  # Optional verification
        "size_mb": 19,
    },
    "parkinsons_svm": {
        "name": "Parkinson's SVM",
        "path": "external_models/parkinsons_detector/ml/best_pd_model.pkl",
        "bundled": True,  # Already included in repo
    },
}


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def check_models_status() -> dict:
    """Check which models are present."""
    root = get_project_root()
    status = {}
    
    for model_id, config in MODELS.items():
        model_path = root / config["path"]
        status[model_id] = {
            "name": config["name"],
            "path": str(model_path),
            "exists": model_path.exists(),
            "bundled": config.get("bundled", False),
            "size_mb": config.get("size_mb", 0),
        }
        
        if model_path.exists():
            actual_size = model_path.stat().st_size / (1024 * 1024)
            status[model_id]["actual_size_mb"] = round(actual_size, 1)
    
    return status


def download_file(url: str, dest_path: Path, name: str) -> bool:
    """Download a file with progress indication."""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"⬇️  Downloading {name}...")
        logger.info(f"   From: {url}")
        logger.info(f"   To: {dest_path}")
        
        # Download with progress
        def reporthook(blocknum, blocksize, totalsize):
            downloaded = blocknum * blocksize
            if totalsize > 0:
                percent = min(100, downloaded * 100 / totalsize)
                sys.stdout.write(f"\r   Progress: {percent:.1f}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, str(dest_path), reporthook)
        print()  # newline after progress
        
        logger.info(f"✅ Downloaded {name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {name}: {e}")
        return False


def download_models(force: bool = False) -> dict:
    """Download all required models."""
    root = get_project_root()
    results = {}
    
    for model_id, config in MODELS.items():
        model_path = root / config["path"]
        
        # Skip bundled models
        if config.get("bundled"):
            if model_path.exists():
                logger.info(f"✅ {config['name']}: Already bundled")
                results[model_id] = "bundled"
            else:
                logger.warning(f"⚠️  {config['name']}: Missing (should be bundled)")
                results[model_id] = "missing_bundled"
            continue
        
        # Check if already exists
        if model_path.exists() and not force:
            logger.info(f"✅ {config['name']}: Already downloaded")
            results[model_id] = "exists"
            continue
        
        # Download
        if "url" in config:
            success = download_file(config["url"], model_path, config["name"])
            results[model_id] = "downloaded" if success else "failed"
        else:
            logger.warning(f"⚠️  {config['name']}: No download URL configured")
            results[model_id] = "no_url"
    
    return results


def verify_models() -> bool:
    """Verify all models can be loaded."""
    logger.info("\n🔍 Verifying models...")
    
    all_ok = True
    
    # Test Parkinson's model
    try:
        import pickle
        root = get_project_root()
        pd_path = root / MODELS["parkinsons_svm"]["path"]
        
        with open(pd_path, "rb") as f:
            data = pickle.load(f)
        
        assert "model" in data
        assert "scaler" in data
        assert "selected_features" in data
        
        logger.info(f"✅ Parkinson's SVM: Loaded successfully")
        logger.info(f"   Model type: {type(data['model']).__name__}")
        logger.info(f"   Features: {len(data['selected_features'])}")
        
    except Exception as e:
        logger.error(f"❌ Parkinson's SVM: Failed to load - {e}")
        all_ok = False
    
    # Test PANNs (if exists)
    panns_path = get_project_root() / MODELS["panns_cnn6"]["path"]
    if panns_path.exists():
        try:
            import torch
            state = torch.load(panns_path, map_location="cpu")
            logger.info(f"✅ PANNs CNN6: Valid PyTorch checkpoint")
        except Exception as e:
            logger.error(f"❌ PANNs CNN6: Invalid - {e}")
            all_ok = False
    else:
        logger.info(f"ℹ️  PANNs CNN6: Not downloaded (optional)")
    
    return all_ok


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--force", action="store_true", help="Re-download existing models")
    parser.add_argument("--check", action="store_true", help="Only check status, don't download")
    parser.add_argument("--verify", action="store_true", help="Verify models can be loaded")
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("PhoneCoughClassifier Model Manager")
    logger.info("=" * 50)
    
    # Check status
    status = check_models_status()
    logger.info("\n📦 Model Status:")
    for model_id, info in status.items():
        icon = "✅" if info["exists"] else "❌"
        bundled = " (bundled)" if info["bundled"] else ""
        size = f" [{info.get('actual_size_mb', info['size_mb'])} MB]" if info["exists"] else ""
        logger.info(f"   {icon} {info['name']}{bundled}{size}")
    
    if args.check:
        return
    
    if args.verify:
        verify_models()
        return
    
    # Download missing models
    logger.info("\n" + "=" * 50)
    results = download_models(force=args.force)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Summary:")
    
    downloaded = sum(1 for r in results.values() if r == "downloaded")
    existing = sum(1 for r in results.values() if r in ("exists", "bundled"))
    failed = sum(1 for r in results.values() if r == "failed")
    
    logger.info(f"   Downloaded: {downloaded}")
    logger.info(f"   Already present: {existing}")
    logger.info(f"   Failed: {failed}")
    
    # Verify if everything looks good
    if failed == 0:
        verify_models()
        logger.info("\n✅ All models ready!")
    else:
        logger.error("\n❌ Some models failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()
