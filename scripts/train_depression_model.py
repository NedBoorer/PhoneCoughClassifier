#!/usr/bin/env python3
"""
Train Depression Voice Classifier
Uses prosodic features from speech samples

Usage:
    python scripts/train_depression_model.py --output models/depression_classifier.joblib
    python scripts/train_depression_model.py --synthetic --samples 500

"""
import argparse
import logging
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

# Aligned with PHQ-9 severity categories
SEVERITY_LEVELS = ["minimal", "mild", "moderate", "moderately_severe", "severe"]

# Feature names for Depression detection
DEPRESSION_FEATURES = [
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_variation_coefficient",
    "speech_rate", "articulation_rate", "num_syllables", "total_duration", "voiced_duration",
    "num_pauses", "total_pause_duration", "mean_pause_duration", "max_pause_duration",
    "pause_ratio", "pause_rate",
    "pitch_range_semitones", "pitch_slope", "pitch_inflection_rate",
    "energy_mean", "energy_std", "energy_variation_coefficient", "energy_range_db"
]


def generate_synthetic_data(num_samples: int = 500):
    """
    Generate synthetic training data for Depression detection.
    Based on prosodic patterns from literature.
    """
    logger.info(f"Generating {num_samples} synthetic samples...")
    
    X = []
    y = []
    
    samples_per_class = num_samples // len(SEVERITY_LEVELS)
    
    for class_idx, severity in enumerate(SEVERITY_LEVELS):
        for _ in range(samples_per_class):
            features = {}
            
            if severity == "minimal":
                # Normal healthy speech patterns
                features["f0_mean"] = np.random.normal(180.0, 40.0)
                features["f0_std"] = np.random.normal(25.0, 8.0)
                features["f0_min"] = np.random.normal(100.0, 30.0)
                features["f0_max"] = np.random.normal(280.0, 50.0)
                features["f0_range"] = np.random.normal(180.0, 40.0)
                features["f0_variation_coefficient"] = np.random.normal(18.0, 4.0)
                
                features["speech_rate"] = np.random.normal(4.5, 0.8)
                features["articulation_rate"] = np.random.normal(5.5, 0.9)
                features["num_syllables"] = np.random.normal(50.0, 15.0)
                features["total_duration"] = np.random.normal(12.0, 3.0)
                features["voiced_duration"] = np.random.normal(10.0, 2.5)
                
                features["num_pauses"] = np.random.normal(5.0, 2.0)
                features["total_pause_duration"] = np.random.normal(1.5, 0.5)
                features["mean_pause_duration"] = np.random.normal(0.3, 0.1)
                features["max_pause_duration"] = np.random.normal(0.6, 0.2)
                features["pause_ratio"] = np.random.normal(0.12, 0.04)
                features["pause_rate"] = np.random.normal(0.4, 0.15)
                
                features["pitch_range_semitones"] = np.random.normal(10.0, 2.5)
                features["pitch_slope"] = np.random.normal(0.0, 5.0)
                features["pitch_inflection_rate"] = np.random.normal(0.4, 0.1)
                
                features["energy_mean"] = np.random.normal(0.05, 0.015)
                features["energy_std"] = np.random.normal(0.02, 0.006)
                features["energy_variation_coefficient"] = np.random.normal(40.0, 10.0)
                features["energy_range_db"] = np.random.normal(30.0, 8.0)
                
            elif severity == "mild":
                # Slight reduction in prosodic variation
                features["f0_mean"] = np.random.normal(170.0, 40.0)
                features["f0_std"] = np.random.normal(20.0, 6.0)
                features["f0_min"] = np.random.normal(110.0, 30.0)
                features["f0_max"] = np.random.normal(250.0, 45.0)
                features["f0_range"] = np.random.normal(140.0, 35.0)
                features["f0_variation_coefficient"] = np.random.normal(13.0, 3.0)
                
                features["speech_rate"] = np.random.normal(4.0, 0.7)
                features["articulation_rate"] = np.random.normal(4.8, 0.8)
                features["num_syllables"] = np.random.normal(45.0, 14.0)
                features["total_duration"] = np.random.normal(12.0, 3.0)
                features["voiced_duration"] = np.random.normal(9.0, 2.3)
                
                features["num_pauses"] = np.random.normal(7.0, 2.5)
                features["total_pause_duration"] = np.random.normal(2.5, 0.8)
                features["mean_pause_duration"] = np.random.normal(0.4, 0.12)
                features["max_pause_duration"] = np.random.normal(0.8, 0.25)
                features["pause_ratio"] = np.random.normal(0.20, 0.05)
                features["pause_rate"] = np.random.normal(0.6, 0.2)
                
                features["pitch_range_semitones"] = np.random.normal(7.0, 2.0)
                features["pitch_slope"] = np.random.normal(-2.0, 4.0)
                features["pitch_inflection_rate"] = np.random.normal(0.3, 0.08)
                
                features["energy_mean"] = np.random.normal(0.04, 0.012)
                features["energy_std"] = np.random.normal(0.015, 0.005)
                features["energy_variation_coefficient"] = np.random.normal(30.0, 8.0)
                features["energy_range_db"] = np.random.normal(25.0, 6.0)
                
            elif severity == "moderate":
                # Noticeable reduction - flatter affect
                features["f0_mean"] = np.random.normal(160.0, 38.0)
                features["f0_std"] = np.random.normal(14.0, 5.0)
                features["f0_min"] = np.random.normal(120.0, 28.0)
                features["f0_max"] = np.random.normal(220.0, 40.0)
                features["f0_range"] = np.random.normal(100.0, 30.0)
                features["f0_variation_coefficient"] = np.random.normal(9.0, 2.5)
                
                features["speech_rate"] = np.random.normal(3.2, 0.6)
                features["articulation_rate"] = np.random.normal(4.0, 0.7)
                features["num_syllables"] = np.random.normal(38.0, 12.0)
                features["total_duration"] = np.random.normal(12.0, 3.0)
                features["voiced_duration"] = np.random.normal(8.0, 2.0)
                
                features["num_pauses"] = np.random.normal(10.0, 3.0)
                features["total_pause_duration"] = np.random.normal(3.5, 1.0)
                features["mean_pause_duration"] = np.random.normal(0.5, 0.15)
                features["max_pause_duration"] = np.random.normal(1.2, 0.35)
                features["pause_ratio"] = np.random.normal(0.28, 0.06)
                features["pause_rate"] = np.random.normal(0.85, 0.25)
                
                features["pitch_range_semitones"] = np.random.normal(5.0, 1.5)
                features["pitch_slope"] = np.random.normal(-4.0, 3.0)
                features["pitch_inflection_rate"] = np.random.normal(0.2, 0.06)
                
                features["energy_mean"] = np.random.normal(0.03, 0.01)
                features["energy_std"] = np.random.normal(0.01, 0.004)
                features["energy_variation_coefficient"] = np.random.normal(22.0, 6.0)
                features["energy_range_db"] = np.random.normal(20.0, 5.0)
                
            elif severity == "moderately_severe":
                # Significant reduction - monotone tendency
                features["f0_mean"] = np.random.normal(150.0, 35.0)
                features["f0_std"] = np.random.normal(10.0, 4.0)
                features["f0_min"] = np.random.normal(130.0, 25.0)
                features["f0_max"] = np.random.normal(190.0, 35.0)
                features["f0_range"] = np.random.normal(60.0, 22.0)
                features["f0_variation_coefficient"] = np.random.normal(6.0, 2.0)
                
                features["speech_rate"] = np.random.normal(2.5, 0.5)
                features["articulation_rate"] = np.random.normal(3.2, 0.6)
                features["num_syllables"] = np.random.normal(30.0, 10.0)
                features["total_duration"] = np.random.normal(12.0, 3.0)
                features["voiced_duration"] = np.random.normal(6.5, 1.8)
                
                features["num_pauses"] = np.random.normal(14.0, 4.0)
                features["total_pause_duration"] = np.random.normal(4.5, 1.2)
                features["mean_pause_duration"] = np.random.normal(0.65, 0.2)
                features["max_pause_duration"] = np.random.normal(1.8, 0.5)
                features["pause_ratio"] = np.random.normal(0.36, 0.08)
                features["pause_rate"] = np.random.normal(1.2, 0.3)
                
                features["pitch_range_semitones"] = np.random.normal(3.0, 1.0)
                features["pitch_slope"] = np.random.normal(-6.0, 2.5)
                features["pitch_inflection_rate"] = np.random.normal(0.12, 0.04)
                
                features["energy_mean"] = np.random.normal(0.025, 0.008)
                features["energy_std"] = np.random.normal(0.007, 0.003)
                features["energy_variation_coefficient"] = np.random.normal(15.0, 5.0)
                features["energy_range_db"] = np.random.normal(15.0, 4.0)
                
            else:  # severe
                # Very flat, slow, many pauses
                features["f0_mean"] = np.random.normal(140.0, 30.0)
                features["f0_std"] = np.random.normal(6.0, 3.0)
                features["f0_min"] = np.random.normal(130.0, 22.0)
                features["f0_max"] = np.random.normal(160.0, 28.0)
                features["f0_range"] = np.random.normal(30.0, 15.0)
                features["f0_variation_coefficient"] = np.random.normal(3.5, 1.5)
                
                features["speech_rate"] = np.random.normal(1.8, 0.4)
                features["articulation_rate"] = np.random.normal(2.5, 0.5)
                features["num_syllables"] = np.random.normal(22.0, 8.0)
                features["total_duration"] = np.random.normal(12.0, 3.0)
                features["voiced_duration"] = np.random.normal(5.0, 1.5)
                
                features["num_pauses"] = np.random.normal(18.0, 5.0)
                features["total_pause_duration"] = np.random.normal(6.0, 1.5)
                features["mean_pause_duration"] = np.random.normal(0.85, 0.25)
                features["max_pause_duration"] = np.random.normal(2.5, 0.7)
                features["pause_ratio"] = np.random.normal(0.48, 0.1)
                features["pause_rate"] = np.random.normal(1.6, 0.4)
                
                features["pitch_range_semitones"] = np.random.normal(1.5, 0.6)
                features["pitch_slope"] = np.random.normal(-8.0, 2.0)
                features["pitch_inflection_rate"] = np.random.normal(0.06, 0.03)
                
                features["energy_mean"] = np.random.normal(0.02, 0.006)
                features["energy_std"] = np.random.normal(0.004, 0.002)
                features["energy_variation_coefficient"] = np.random.normal(8.0, 3.0)
                features["energy_range_db"] = np.random.normal(10.0, 3.0)
            
            # Ensure positivity where needed
            for key in features:
                if key not in ["pitch_slope"]:
                    features[key] = abs(features[key])
            
            feature_vector = [features.get(name, 0.0) for name in DEPRESSION_FEATURES]
            X.append(feature_vector)
            y.append(class_idx)
    
    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Train the best performing model"""
    logger.info("Training Depression classifier...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    }
    
    best_score = 0
    best_name = None
    best_clf = None
    
    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        mean_score = scores.mean()
        
        logger.info(f"  {name}: CV accuracy = {mean_score:.3f} (+/- {scores.std():.3f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_clf = pipeline
    
    logger.info(f"\nBest model: {best_name} with CV accuracy = {best_score:.3f}")
    
    best_clf.fit(X_train, y_train)
    
    y_pred = best_clf.predict(X_test)
    
    logger.info("\nTest Set Evaluation:")
    logger.info(f"Accuracy: {(y_pred == y_test).mean():.3f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=SEVERITY_LEVELS))
    
    return best_clf


def main():
    parser = argparse.ArgumentParser(description="Train Depression Voice Classifier")
    parser.add_argument("--data-dir", type=str, help="Path to training data directory")
    parser.add_argument("--output", type=str, default="models/depression_classifier.joblib",
                        help="Output path for trained model")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic training data")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of synthetic samples")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.data_dir:
        logger.warning("Real data loading not implemented yet. Using synthetic data.")
        X, y = generate_synthetic_data(args.samples)
    else:
        X, y = generate_synthetic_data(args.samples)
    
    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    model = train_model(X, y)
    
    joblib.dump(model, output_path)
    logger.info(f"\nModel saved to: {output_path}")
    
    metadata = {
        "model_type": "depression_classifier",
        "feature_names": DEPRESSION_FEATURES,
        "class_names": SEVERITY_LEVELS,
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "synthetic_data": True if not args.data_dir else False
    }
    
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
