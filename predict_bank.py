import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from Bank import build_feature_matrix

def main():
  parser = argparse.ArgumentParser(
      description="Predict bank deposit from customer information")
  parser.add_argument("--model", required=True,
                      help="Path to joblib model file")
  parser.add_argument("--age", required=True, type=int, help="Age (years)")
  parser.add_argument("--job", required=True, help="Job type")
  parser.add_argument("--marital", required=True, help="Marital status")
  parser.add_argument("--education", required=True, help="Education level")
  parser.add_argument("--default", required=True,
                      choices=['yes', 'no'], help="Has credit in default?")
  parser.add_argument("--balance", required=True,
                      type=float, help="Account balance")
  parser.add_argument("--housing", required=True,
                      choices=['yes', 'no'], help="Has housing loan?")
  parser.add_argument("--loan", required=True,
                      choices=['yes', 'no'], help="Has personal loan?")
  parser.add_argument("--contact", required=True,
                      help="Contact communication type")
  parser.add_argument("--day", required=True, type=int,
                      help="Last contact day of month")
  parser.add_argument("--month", required=True, help="Last contact month")
  parser.add_argument("--duration", required=True, type=int,
                      help="Last contact duration (seconds)")
  parser.add_argument("--campaign", required=True, type=int,
                      help="Number of contacts this campaign")
  parser.add_argument("--pdays", required=True, type=int,
                      help="Days since last contact (-1 = not contacted)")
  parser.add_argument("--previous", required=True, type=int,
                      help="Number of contacts before this campaign")
  parser.add_argument("--poutcome", required=True,
                      help="Outcome of previous campaign")
  args = parser.parse_args()

  model_path = Path(args.model)

  # Load model
  try:
    model = joblib.load(model_path)
  except Exception as e:
    print(f"Failed to load model: {e}", file=sys.stderr)
    sys.exit(1)

  # Create raw feature DataFrame with deposit column
  raw_df = pd.DataFrame({
      "age": [args.age],
      "job": [args.job],
      "marital": [args.marital],
      "education": [args.education],
      "default": [args.default],
      "balance": [args.balance],
      "housing": [args.housing],
      "loan": [args.loan],
      "contact": [args.contact],
      "day": [args.day],
      "month": [args.month],
      "duration": [args.duration],
      "campaign": [args.campaign],
      "pdays": [args.pdays],
      "previous": [args.previous],
      "poutcome": [args.poutcome],
      "deposit": ["no"]
  })

  # Use the function from Bank.py to build features
  X, _ = build_feature_matrix(raw_df)

  # Predict
  try:
    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0]

    result = "YES - Will make a deposit" if prediction == 1 else "NO - Will not make a deposit"

    print(f"\nPrediction: {result}")
    print(f"\nProbabilities:")
    print(f"  No Deposit: {prediction_proba[0]:.1%}")
    print(f"  Deposit: {prediction_proba[1]:.1%}")

    print(f"\nInput Summary:")
    print(f"  Age: {args.age} years")
    print(f"  Job: {args.job}")
    print(f"  Marital: {args.marital}")
    print(f"  Education: {args.education}")
    print(f"  Balance: ${args.balance:.2f}")
    print(f"  Housing Loan: {args.housing}")
    print(f"  Personal Loan: {args.loan}")
    print(f"  Last Contact Duration: {args.duration} seconds")
    print(f"  Campaign Contacts: {args.campaign}")

  except Exception as e:
    print(f"Prediction failed: {e}", file=sys.stderr)
    print(f"Error details: {type(e).__name__}", file=sys.stderr)
    sys.exit(2)

if __name__ == "__main__":
  main()
