import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from Insurance import build_feature_matrix

def main():
  parser = argparse.ArgumentParser(
      description="Predict insurance charges from personal information")
  parser.add_argument("--model", required=True,
                      help="Path to joblib model file")
  parser.add_argument("--age", required=True, type=int, help="Age (years)")
  parser.add_argument("--sex", required=True,
                      choices=['male', 'female'], help="Sex")
  parser.add_argument("--bmi", required=True, type=float,
                      help="BMI (Body Mass Index)")
  parser.add_argument("--children", required=True,
                      type=int, help="Number of children")
  parser.add_argument("--smoker", required=True,
                      choices=['yes', 'no'], help="Smoker status")
  parser.add_argument("--region", required=True,
                      choices=['northwest', 'northeast', 'southwest', 'southeast'], help="Region")
  args = parser.parse_args()

  model_path = Path(args.model)

  # Load model
  try:
    model = joblib.load(model_path)
  except Exception as e:
    print(f"Failed to load model: {e}", file=sys.stderr)
    sys.exit(1)

  # Create raw feature DataFrame with charges column
  raw_df = pd.DataFrame({
      "age": [args.age],
      "sex": [args.sex],
      "bmi": [args.bmi],
      "children": [args.children],
      "smoker": [args.smoker],
      "region": [args.region],
      "charges": [0]
  })

  # Use the function from Insurance.py to build features
  X, _ = build_feature_matrix(raw_df)

  # Predict
  try:
    prediction_log = model.predict(X)[0]
    prediction = np.exp(prediction_log)

    # Determine BMI category for display
    if args.bmi < 18.5:
      bmi_category = 'underweight'
    elif args.bmi < 25:
      bmi_category = 'normal'
    elif args.bmi < 30:
      bmi_category = 'overweight'
    elif args.bmi < 35:
      bmi_category = 'obese'
    else:
      bmi_category = 'morbid_obese'

    print(f"\nPredicted insurance charges: ${prediction:.2f}")
    print(f"\nInput Summary:")
    print(f"  Age: {args.age} years")
    print(f"  Sex: {args.sex}")
    print(f"  BMI: {args.bmi:.1f} ({bmi_category})")
    print(f"  Children: {args.children}")
    print(f"  Smoker: {args.smoker}")
    print(f"  Region: {args.region}")

  except Exception as e:
    print(f"Prediction failed: {e}", file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
  main()
