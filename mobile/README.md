# Mobile App Integration Guide

## Overview
This directory is reserved for the offline-capable mobile application.
The mobile app uses the same ML logic as the backend but runs it directly on the device using **ONNX Runtime**.

## Prerequisites
- [Flutter SDK](https://flutter.dev/docs/get-started/install) installed.

## Setup Instructions

1.  **Initialize Flutter App:**
    Run this command inside the `mobile/` directory:
    ```bash
    flutter create .
    ```

2.  **Add Dependencies:**
    Add these to `pubspec.yaml`:
    ```yaml
    dependencies:
      flutter:
        sdk: flutter
      # For Audio Recording
      flutter_sound: ^9.2.13
      permission_handler: ^10.2.0
      # For ML Inference (Offline)
      onnxruntime: ^1.15.0
      # For Audio Processing (Feature Extraction equivalent to Librosa)
      fftea: ^1.2.0 
    ```

3.  **Export the Model:**
    You must convert the Python model to ONNX format:
    ```bash
    cd ../scripts
    python export_to_onnx.py
    ```
    This creates `models/cough_classifier.onnx`.

4.  **Copy Model to App:**
    Create a `assets/` folder in `mobile/` and copy the `.onnx` file there.
    Update `pubspec.yaml` to include assets:
    ```yaml
    flutter:
      assets:
        - assets/cough_classifier.onnx
    ```

## The Hard Part: Feature Extraction
The Python backend uses `librosa` to convert audio into numbers (Mel Spectrograms, MFCCs).
**Flutter does not have librosa.**
You must re-implement the feature extraction in Dart or use a C++ library via FFI.

**Strategy for Hackathon:**
If re-implementing MFCCs in Dart is too hard, use the **Online Mode** (hit the FastAPI endpoint) as the default, and keep "Offline Mode" as a "Coming Soon" or simple heuristic feature.
