import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

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

  # Create engineered features
  age_balance = args.age * args.balance
  duration_campaign = args.duration * args.campaign

  # Create balance category
  if args.balance < 0:
    balance_category = 'negative'
  elif args.balance < 500:
    balance_category = 'low'
  elif args.balance < 1000:
    balance_category = 'medium'
  elif args.balance < 2000:
    balance_category = 'high'
  else:
    balance_category = 'very_high'

  # Create age category
  if args.age <= 30:
    age_category = 'young'
  elif args.age <= 45:
    age_category = 'middle'
  elif args.age <= 60:
    age_category = 'senior'
  else:
    age_category = 'elder'

  # Create job_marital feature
  job_marital = f"{args.job}_{args.marital}"

  # Create balance_log
  balance_log = np.log1p(max(0, args.balance))

  # Convert binary to numeric (matching training preprocessing)
  default_num = 1 if args.default == 'yes' else 0
  housing_num = 1 if args.housing == 'yes' else 0
  loan_num = 1 if args.loan == 'yes' else 0

  # Create feature DataFrame with all features
  X = pd.DataFrame({
      "age": [args.age],
      "job": [args.job],
      "marital": [args.marital],
      "education": [args.education],
      "default": [default_num],
      "balance": [args.balance],
      "housing": [housing_num],
      "loan": [loan_num],
      "contact": [args.contact],
      "day": [args.day],
      "month": [args.month],
      "duration": [args.duration],
      "campaign": [args.campaign],
      "pdays": [args.pdays],
      "previous": [args.previous],
      "poutcome": [args.poutcome],
      "age_balance": [age_balance],
      "duration_campaign": [duration_campaign],
      "balance_category": [balance_category],
      "job_marital": [job_marital],
      "age_category": [age_category],
      "balance_log": [balance_log]
  })

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
    print(f"  Age: {args.age} years ({age_category})")
    print(f"  Job: {args.job}")
    print(f"  Marital: {args.marital}")
    print(f"  Education: {args.education}")
    print(f"  Balance: ${args.balance:.2f} ({balance_category})")
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
