#!/usr/bin/env python3
"""
Train Cough Classifier Model
Uses COUGHVID dataset or synthetic data

Usage:
    python scripts/train_model.py --data-dir data/coughvid --output models/cough_classifier.joblib

"""
import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


COUGH_CLASSES = ["dry", "wet", "whooping", "chronic", "normal"]


def load_audio_files(data_dir: Path) -> List[Tuple[Path, str]]:
    """Load audio files and their labels from dataset"""
    samples = []
    
    # Try to load metadata
    metadata_files = list(data_dir.glob("**/metadata.json"))
    
    if metadata_files:
        # Use our synthetic metadata format
        with open(metadata_files[0]) as f:
            metadata = json.load(f)
        
        labels = metadata.get("labels", {})
        for name, label in labels.items():
            # Find audio file
            audio_files = list(data_dir.glob(f"**/{name}.*"))
            if audio_files:
                samples.append((audio_files[0], label))
                
    else:
        # Try to find CSV metadata (COUGHVID format)
        csv_files = list(data_dir.glob("**/*.csv"))
        
        if csv_files:
            try:
                import pandas as pd
                df = pd.read_csv(csv_files[0])
                
                # Find audio column and label column
                audio_col = None
                label_col = None
                
                for col in df.columns:
                    if "uuid" in col.lower() or "file" in col.lower():
                        audio_col = col
                    if "status" in col.lower() or "label" in col.lower() or "cough_type" in col.lower():
                        label_col = col
                
                if audio_col and label_col:
                    for _, row in df.iterrows():
                        # Map COUGHVID labels to our classes
                        label = str(row[label_col]).lower()
                        if "healthy" in label or "covid" not in label:
                            label = "normal"
                        elif "symptomatic" in label:
                            label = "dry"  # Simplification
                        
                        # Find audio file
                        audio_id = str(row[audio_col])
                        audio_files = list(data_dir.glob(f"**/{audio_id}.*"))
                        
                        if audio_files and label in COUGH_CLASSES:
                            samples.append((audio_files[0], label))
                            
            except Exception as e:
                logger.warning(f"Failed to parse CSV metadata: {e}")
        
        # Fallback: infer labels from filenames
        if not samples:
            for audio_file in data_dir.glob("**/*.wav"):
                name = audio_file.stem.lower()
                for label in COUGH_CLASSES:
                    if label in name:
                        samples.append((audio_file, label))
                        break
    
    logger.info(f"Found {len(samples)} labeled samples")
    
    # Log class distribution
    from collections import Counter
    class_counts = Counter(label for _, label in samples)
    logger.info(f"Class distribution: {dict(class_counts)}")
    
    return samples


def extract_features(samples: List[Tuple[Path, str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features from all audio samples"""
    from app.ml.feature_extractor import get_feature_extractor
    
    extractor = get_feature_extractor()
    
    X = []
    y = []
    feature_names = None
    
    for audio_path, label in samples:
        try:
            features, names = extractor.get_feature_vector(str(audio_path))
            X.append(features)
            y.append(COUGH_CLASSES.index(label))
            
            if feature_names is None:
                feature_names = names
                
        except Exception as e:
            logger.warning(f"Failed to extract features from {audio_path}: {e}")
    
    return np.array(X), np.array(y), feature_names


def train_random_forest(X: np.ndarray, y: np.ndarray) -> object:
    """Train Random Forest classifier"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    
    logger.info("Training Random Forest classifier...")
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Cross-validation
    scores = cross_val_score(pipeline, X, y, cv=min(5, len(y) // 2), scoring='accuracy')
    logger.info(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    
    # Train on full data
    pipeline.fit(X, y)
    
    return pipeline


def evaluate_model(model, X: np.ndarray, y: np.ndarray):
    """Evaluate model performance"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_pred = model.predict(X)
    
    logger.info("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=COUGH_CLASSES))
    
    logger.info("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))


def generate_synthetic_data(num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data when no real data is available"""
    logger.info(f"Generating {num_samples} synthetic training samples...")
    
    # Number of features (must match feature extractor)
    num_features = 150  # Approximate
    
    # Generate class-specific features
    X = []
    y = []
    
    samples_per_class = num_samples // len(COUGH_CLASSES)
    
    for class_idx, class_name in enumerate(COUGH_CLASSES):
        # Create class-specific feature distributions
        class_features = np.random.randn(samples_per_class, num_features)
        
        # Add class-specific patterns
        if class_name == "dry":
            class_features[:, 0:20] += 1.5  # Higher MFCCs
            class_features[:, 40:50] += 2.0  # Higher spectral centroid
        elif class_name == "wet":
            class_features[:, 0:20] -= 1.0  # Lower MFCCs
            class_features[:, 40:50] -= 1.5  # Lower spectral centroid
        elif class_name == "whooping":
            class_features[:, 50:60] += 2.5  # Higher ZCR
            class_features[:, 70:80] += 2.0  # More onsets
        elif class_name == "chronic":
            class_features[:, 60:70] += 1.5  # Longer duration patterns
        # Normal stays at baseline
        
        X.extend(class_features)
        y.extend([class_idx] * samples_per_class)
    
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Train cough classifier")
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="data/coughvid",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/cough_classifier.joblib",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data for training"
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    logger.info("=" * 50)
    logger.info("Cough Classifier Training")
    logger.info("=" * 50)
    
    # Load or generate data
    if args.use_synthetic or not data_dir.exists():
        logger.info("Using synthetic training data")
        X, y = generate_synthetic_data(500)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        # Load real data
        samples = load_audio_files(data_dir)
        
        if len(samples) < 10:
            logger.warning(f"Only {len(samples)} samples found. Using synthetic data.")
            X, y = generate_synthetic_data(500)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            X, y, feature_names = extract_features(samples)
    
    logger.info(f"Training set: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train model
    model = train_random_forest(X, y)
    
    # Evaluate
    evaluate_model(model, X, y)
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import joblib
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "classes": COUGH_CLASSES,
        "version": "1.0.0"
    }, output_path)
    
    logger.info(f"✓ Model saved to {output_path}")
    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
