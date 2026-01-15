#!/usr/bin/env python3
"""
Train Parkinson's Disease Voice Classifier
Uses acoustic features from sustained vowel recordings

Usage:
    python scripts/train_parkinsons_model.py --output models/parkinsons_classifier.joblib
    python scripts/train_parkinsons_model.py --synthetic --samples 500

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
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

RISK_LEVELS = ["normal", "low_risk", "moderate_risk", "elevated_risk"]

# Feature names for Parkinson's detection
PARKINSONS_FEATURES = [
    "jitter_absolute", "jitter_relative", "jitter_rap", "jitter_ppq5",
    "shimmer_absolute", "shimmer_relative", "shimmer_apq3", "shimmer_apq5", "shimmer_dda",
    "hnr_mean", "hnr_std", "hnr_min", "hnr_max",
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_variation_coefficient"
]


def generate_synthetic_data(num_samples: int = 500):
    """
    Generate synthetic training data for Parkinson's detection.
    Based on clinical feature distributions from literature.
    """
    logger.info(f"Generating {num_samples} synthetic samples...")
    
    X = []
    y = []
    
    samples_per_class = num_samples // len(RISK_LEVELS)
    
    for class_idx, risk_level in enumerate(RISK_LEVELS):
        for _ in range(samples_per_class):
            features = {}
            
            if risk_level == "normal":
                # Normal healthy voice patterns
                features["jitter_absolute"] = np.random.normal(0.0003, 0.0001)
                features["jitter_relative"] = np.random.normal(0.5, 0.2)
                features["jitter_rap"] = np.random.normal(0.3, 0.1)
                features["jitter_ppq5"] = np.random.normal(0.35, 0.1)
                
                features["shimmer_absolute"] = np.random.normal(0.02, 0.01)
                features["shimmer_relative"] = np.random.normal(2.0, 0.5)
                features["shimmer_apq3"] = np.random.normal(1.0, 0.3)
                features["shimmer_apq5"] = np.random.normal(1.2, 0.3)
                features["shimmer_dda"] = np.random.normal(3.0, 0.9)
                
                features["hnr_mean"] = np.random.normal(25.0, 3.0)
                features["hnr_std"] = np.random.normal(2.0, 0.5)
                features["hnr_min"] = np.random.normal(18.0, 3.0)
                features["hnr_max"] = np.random.normal(30.0, 3.0)
                
                features["f0_mean"] = np.random.normal(150.0, 30.0)
                features["f0_std"] = np.random.normal(3.0, 1.0)
                features["f0_min"] = np.random.normal(120.0, 20.0)
                features["f0_max"] = np.random.normal(180.0, 30.0)
                features["f0_range"] = np.random.normal(60.0, 15.0)
                features["f0_variation_coefficient"] = np.random.normal(2.0, 0.5)
                
            elif risk_level == "low_risk":
                # Slight elevation in perturbation measures
                features["jitter_absolute"] = np.random.normal(0.0005, 0.0002)
                features["jitter_relative"] = np.random.normal(1.2, 0.3)
                features["jitter_rap"] = np.random.normal(0.6, 0.2)
                features["jitter_ppq5"] = np.random.normal(0.7, 0.2)
                
                features["shimmer_absolute"] = np.random.normal(0.04, 0.015)
                features["shimmer_relative"] = np.random.normal(4.0, 1.0)
                features["shimmer_apq3"] = np.random.normal(2.0, 0.5)
                features["shimmer_apq5"] = np.random.normal(2.5, 0.6)
                features["shimmer_dda"] = np.random.normal(6.0, 1.5)
                
                features["hnr_mean"] = np.random.normal(18.0, 3.0)
                features["hnr_std"] = np.random.normal(3.0, 0.8)
                features["hnr_min"] = np.random.normal(12.0, 3.0)
                features["hnr_max"] = np.random.normal(24.0, 3.0)
                
                features["f0_mean"] = np.random.normal(140.0, 35.0)
                features["f0_std"] = np.random.normal(5.0, 1.5)
                features["f0_min"] = np.random.normal(110.0, 25.0)
                features["f0_max"] = np.random.normal(175.0, 35.0)
                features["f0_range"] = np.random.normal(65.0, 20.0)
                features["f0_variation_coefficient"] = np.random.normal(3.5, 1.0)
                
            elif risk_level == "moderate_risk":
                # Moderate elevation - concerning patterns
                features["jitter_absolute"] = np.random.normal(0.001, 0.0003)
                features["jitter_relative"] = np.random.normal(2.0, 0.5)
                features["jitter_rap"] = np.random.normal(1.0, 0.3)
                features["jitter_ppq5"] = np.random.normal(1.2, 0.3)
                
                features["shimmer_absolute"] = np.random.normal(0.06, 0.02)
                features["shimmer_relative"] = np.random.normal(6.0, 1.5)
                features["shimmer_apq3"] = np.random.normal(3.0, 0.8)
                features["shimmer_apq5"] = np.random.normal(4.0, 1.0)
                features["shimmer_dda"] = np.random.normal(9.0, 2.4)
                
                features["hnr_mean"] = np.random.normal(13.0, 3.0)
                features["hnr_std"] = np.random.normal(4.0, 1.0)
                features["hnr_min"] = np.random.normal(8.0, 3.0)
                features["hnr_max"] = np.random.normal(18.0, 3.0)
                
                features["f0_mean"] = np.random.normal(130.0, 40.0)
                features["f0_std"] = np.random.normal(7.0, 2.0)
                features["f0_min"] = np.random.normal(95.0, 30.0)
                features["f0_max"] = np.random.normal(170.0, 40.0)
                features["f0_range"] = np.random.normal(75.0, 25.0)
                features["f0_variation_coefficient"] = np.random.normal(5.0, 1.5)
                
            else:  # elevated_risk
                # High perturbation - significant concern
                features["jitter_absolute"] = np.random.normal(0.002, 0.0005)
                features["jitter_relative"] = np.random.normal(3.5, 0.8)
                features["jitter_rap"] = np.random.normal(1.8, 0.5)
                features["jitter_ppq5"] = np.random.normal(2.0, 0.5)
                
                features["shimmer_absolute"] = np.random.normal(0.1, 0.03)
                features["shimmer_relative"] = np.random.normal(9.0, 2.0)
                features["shimmer_apq3"] = np.random.normal(5.0, 1.2)
                features["shimmer_apq5"] = np.random.normal(6.0, 1.5)
                features["shimmer_dda"] = np.random.normal(15.0, 3.6)
                
                features["hnr_mean"] = np.random.normal(8.0, 3.0)
                features["hnr_std"] = np.random.normal(5.0, 1.5)
                features["hnr_min"] = np.random.normal(3.0, 2.0)
                features["hnr_max"] = np.random.normal(14.0, 3.0)
                
                features["f0_mean"] = np.random.normal(120.0, 45.0)
                features["f0_std"] = np.random.normal(10.0, 3.0)
                features["f0_min"] = np.random.normal(80.0, 30.0)
                features["f0_max"] = np.random.normal(165.0, 45.0)
                features["f0_range"] = np.random.normal(85.0, 30.0)
                features["f0_variation_coefficient"] = np.random.normal(8.0, 2.0)
            
            # Ensure all values are positive where appropriate
            for key in features:
                if key not in ["hnr_min"]:  # HNR can be negative
                    features[key] = abs(features[key])
            
            # Create feature vector
            feature_vector = [features[name] for name in PARKINSONS_FEATURES]
            X.append(feature_vector)
            y.append(class_idx)
    
    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Train the best performing model using cross-validation"""
    logger.info("Training Parkinson's classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Try different classifiers
    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
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
        
        # Cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        mean_score = scores.mean()
        
        logger.info(f"  {name}: CV accuracy = {mean_score:.3f} (+/- {scores.std():.3f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_clf = pipeline
    
    logger.info(f"\nBest model: {best_name} with CV accuracy = {best_score:.3f}")
    
    # Train final model on all training data
    best_clf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_clf.predict(X_test)
    
    logger.info("\nTest Set Evaluation:")
    logger.info(f"Accuracy: {(y_pred == y_test).mean():.3f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=RISK_LEVELS))
    
    return best_clf


def main():
    parser = argparse.ArgumentParser(description="Train Parkinson's Disease Voice Classifier")
    parser.add_argument("--data-dir", type=str, help="Path to training data directory")
    parser.add_argument("--output", type=str, default="models/parkinsons_classifier.joblib",
                        help="Output path for trained model")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic training data")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of synthetic samples to generate")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get training data
    if args.data_dir:
        # Load real data - would need implementation for specific dataset
        logger.warning("Real data loading not implemented yet. Using synthetic data.")
        X, y = generate_synthetic_data(args.samples)
    else:
        X, y = generate_synthetic_data(args.samples)
    
    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    joblib.dump(model, output_path)
    logger.info(f"\nModel saved to: {output_path}")
    
    # Save metadata
    metadata = {
        "model_type": "parkinsons_classifier",
        "feature_names": PARKINSONS_FEATURES,
        "class_names": RISK_LEVELS,
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
