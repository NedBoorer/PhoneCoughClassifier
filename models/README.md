# ML Models Documentation

This directory contains trained machine learning models for voice health screening.

## Model Status

### ✅ Available Models

| Model | File | Purpose | Status |
|-------|------|---------|--------|
| **Cough Classifier** | `cough_classifier.joblib` | Primary cough sound classification | ✅ Present |
| **Parkinson's Detector** | `parkinsons_classifier.joblib` | Voice-based Parkinson's screening | ✅ Present |
| **Depression Screener** | `depression_classifier.joblib` | Speech pattern analysis for mood | ✅ Present |
| **PANNs Respiratory** | `../external_models/respiratory_panns/panns/Cnn6_mAP=0.343.pth` | Deep learning respiratory sound classification | ✅ Present |
| **Parkinson's SVM** | `../external_models/parkinsons_detector/ml/best_pd_model.pkl` | SVM model for Parkinson's detection | ✅ Present |

### ⚠️ Optional/Missing Models

| Model | Purpose | Status | Fallback Behavior |
|-------|---------|--------|-------------------|
| **Google HeAR** | Health Acoustic Representations (foundation model) | ⚠️ Not publicly available | Uses PANNs CNN instead |
| **Depression CNN** | Deep learning depression detection | ⚠️ Weights not included | Uses feature-based heuristics |

## Model Descriptions

### 1. Cough Classifier (`cough_classifier.joblib`)
- **Type**: Supervised ML classifier
- **Input**: Audio features (MFCCs, spectral features)
- **Output**: Cough classification (healthy, COPD, asthma, etc.)
- **Training Data**: COUGHVID dataset + augmentations

### 2. Respiratory Classifier (PANNs CNN6)
- **Type**: Pretrained Audio Neural Network
- **Architecture**: CNN6 (6-layer convolutional network)
- **Input**: Mel spectrograms (64 bins @ 16kHz)
- **Output**: 4 classes (normal, crackle, wheeze, both)
- **Performance**: mAP = 0.343 on respiratory sound dataset

### 3. Parkinson's Detector
- **Type**: SVM with voice biomarker features
- **Features**: Jitter, shimmer, HNR, RPDE, DFA, PPE
- **Input**: Voice recording (sustained phonation or speech)
- **Output**: Parkinson's probability + biomarker analysis
- **Threshold**: 0.7 confidence for detection

### 4. Depression Screener
- **Type**: Feature-based heuristic model
- **Features**: Pitch variability, energy, speaking rate, pause ratio
- **Input**: Speech sample (10-15 seconds)
- **Output**: Risk score based on speech indicators
- **Research Basis**: Clinical depression voice markers

## Model Files

```
models/
├── cough_classifier.joblib          # Main cough classifier
├── parkinsons_classifier.joblib     # Parkinson's model wrapper
├── parkinsons_classifier.json       # Model metadata
├── depression_classifier.joblib     # Depression model wrapper
├── depression_classifier.json       # Model metadata
└── README.md                        # This file

external_models/
├── respiratory_panns/
│   ├── panns/
│   │   └── Cnn6_mAP=0.343.pth      # PANNs weights
│   └── models/                      # CNN architecture code
└── parkinsons_detector/
    └── ml/
        └── best_pd_model.pkl        # SVM + scaler + features
```

## Fallback Behavior

The application gracefully handles missing models:

1. **Missing Cough Classifier**: Falls back to rule-based classification using audio features
2. **Missing PANNs Weights**: Initializes CNN from scratch (lower accuracy)
3. **Missing Parkinson's Model**: Screening disabled, returns "unknown" status
4. **Missing Depression Model**: Uses basic feature thresholds (reduced accuracy)

## Training New Models

### Cough Classifier

```python
# Training script not included - proprietary dataset
# Expected input format: WAV files @ 16kHz, labeled
# Output: joblib file with sklearn classifier
```

### Fine-tuning PANNs

```python
# Requires respiratory sound dataset (e.g., ICBHI 2017)
# See external_models/respiratory_panns/ for training code
```

### Parkinson's Detector

```python
# Uses Oxford Parkinson's Dataset or similar
# Features extracted via librosa
# SVM training with selected features
```

## Performance Benchmarks

| Model | Accuracy | Precision | Recall | Notes |
|-------|----------|-----------|--------|-------|
| Cough Classifier | TBD | TBD | TBD | Requires validation set |
| PANNs Respiratory | 0.343 mAP | - | - | On original dataset |
| Parkinson's SVM | ~85% | ~80% | ~90% | Cross-validated |
| Depression Heuristic | N/A | N/A | N/A | Research-based thresholds |

## Model Updates

To update or replace a model:

1. Train new model using appropriate dataset
2. Save as `.joblib` (sklearn) or `.pth` (PyTorch)
3. Update `app/config.py` with new path if needed
4. Test with `/test/model-status` endpoint
5. Validate with sample audio files

## Required Dependencies

All model dependencies are specified in `requirements.txt`:
- `torch` - PyTorch for PANNs
- `librosa` - Audio feature extraction
- `scikit-learn` - Traditional ML models
- `numpy` - Numerical operations

## Security Notes

- Models are loaded at runtime (not at import)
- Model files should be verified for integrity
- Avoid loading untrusted model files (pickle vulnerability)
- All models run in CPU mode by default (set `SCREENING_MODEL_DEVICE=cuda` for GPU)

## Contact

For questions about model training or integration, see the main project documentation.
