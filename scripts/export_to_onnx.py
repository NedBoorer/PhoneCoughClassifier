import joblib
import onnx
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import sys
import os

# Add parent directory to path to import app modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def export_classifier_to_onnx(model_path, output_path):
    print(f"Loading model from {model_path}...")
    
    try:
        model_data = joblib.load(model_path)
        
        # Handle different saving structures (sometimes it's the model directly, sometimes a dict)
        if isinstance(model_data, dict) and "model" in model_data:
            clf = model_data["model"]
            print("Extracted model from dictionary.")
        else:
            clf = model_data
            
        print(f"Model type: {type(clf)}")

        # Define input shape
        # This depends heavily on your feature extraction. 
        # Assuming the input is a 1D float array of features.
        # You need to check classifier.n_features_in_ to be exact.
        
        try:
            n_features = clf.n_features_in_
            print(f"Model expects {n_features} input features.")
        except AttributeError:
            print("Could not determine n_features_in_, defaulting to 1 (Risk of failure).")
            n_features = 1

        initial_type = [('float_input', FloatTensorType([None, n_features]))]

        # Convert
        print("Converting to ONNX...")
        onx = convert_sklearn(clf, initial_types=initial_type)

        # Save
        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())
            
        print(f"✓ Successfully exported to {output_path}")
        print("NOTE: You must implement the EXACT same feature extraction (librosa logic) in Dart/Flutter.")

    except Exception as e:
        print(f"X Failed to export: {e}")

if __name__ == "__main__":
    # Default paths
    input_model = "../models/cough_classifier.joblib"
    output_model = "../models/cough_classifier.onnx"
    
    if len(sys.argv) > 1:
        input_model = sys.argv[1]
    
    export_classifier_to_onnx(input_model, output_model)
