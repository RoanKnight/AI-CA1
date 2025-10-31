import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

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

  # Create engineered features
  bmi_children = args.bmi * args.children
  age_children = args.age * args.children

  # Create BMI category
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

  # Create region_smoker feature
  region_smoker = f"{args.region}_{args.smoker}"

  # Create feature DataFrame with engineered features
  X = pd.DataFrame({
      "age": [args.age],
      "sex": [args.sex],
      "bmi": [args.bmi],
      "children": [args.children],
      "smoker": [args.smoker],
      "region": [args.region],
      "bmi_children": [bmi_children],
      "age_children": [age_children],
      "bmi_category": [bmi_category],
      "region_smoker": [region_smoker]
  })

  # Predict
  try:
    prediction_log = model.predict(X)[0]
    prediction = np.exp(prediction_log)

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
