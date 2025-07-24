import sys
import os
import json
import argparse
# Make sure prediction_tool.py is in the same directory or accessible
from prediction_tool import AmyloidPredictor 

# --- Define Paths ---
# These paths should point to where you've stored the models and scaler
MODEL_DIRECTORY = "champion_v8_ensemble_modelv3"
SCALER_FILE_PATH = os.path.join(MODEL_DIRECTORY, "scalar_scaler_v8_ensemble.joblib")

def main():
    """
    Main function to run prediction from the command line.
    Parses arguments, initializes the predictor, runs the prediction,
    and prints the result as a JSON string.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Amyloid Protein Prediction.")
    parser.add_argument("--pdb", required=True, help="Path to the input .pdb file.")
    parser.add_argument("--json", required=True, help="Path to the input .json file.")
    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.pdb):
        print(json.dumps({"error": f"PDB file not found at: {args.pdb}"}))
        sys.exit(1)
    if not os.path.exists(args.json):
        print(json.dumps({"error": f"JSON file not found at: {args.json}"}))
        sys.exit(1)
        
    try:
        # --- Initialize the Predictor ---
        # This is the most time-consuming step, but it only happens once per script execution.
        predictor = AmyloidPredictor(model_dir=MODEL_DIRECTORY, scaler_path=SCALER_FILE_PATH)
        
        # --- Run Prediction ---
        result = predictor.predict(pdb_path=args.pdb, json_path=args.json)
        
        # --- Print Result to Standard Output ---
        # The Deno process will capture this JSON string.
        print(json.dumps(result))

    except Exception as e:
        # Print any exceptions as a JSON error object
        print(json.dumps({"error": f"An unexpected error occurred in Python: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()