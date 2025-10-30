import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
  """Create an enriched design matrix with informative derived features."""
  feature_engineered_df = df.copy()
  feature_engineered_df['bmi_children'] = feature_engineered_df['bmi'] * feature_engineered_df['children']
  feature_engineered_df['age_children'] = feature_engineered_df['age'] * feature_engineered_df['children']
  feature_engineered_df['bmi_category'] = pd.cut(
      feature_engineered_df['bmi'], bins=[0, 18.5, 25, 30, 35, np.inf],
      labels=['underweight', 'normal', 'overweight', 'obese', 'morbid_obese']
  )
  feature_engineered_df['region_smoker'] = feature_engineered_df['region'] + '_' + feature_engineered_df['smoker']

  X = pd.get_dummies(feature_engineered_df.drop('charges', axis=1), drop_first=True)
  y = feature_engineered_df['charges']

  return X, y

def main():
  insurance = pd.read_csv('DataFiles/insurance.csv')
  X, y = build_feature_matrix(insurance)

  print(
      f"\nUsing {len(X)} rows with {len(X.columns)} features (including engineered).")

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.1, random_state=0)
  print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

  # Log-transform target for better modeling of skewed charges
  y_train_log = np.log(y_train)

  # HGB hyperparameter tuning
  hgb_param_grid = {
      'max_iter': [100, 200, 300],
      'learning_rate': [0.05, 0.1, 0.15],
      'max_depth': [3, 5, 7],
      'l2_regularization': [0.0, 0.1, 1.0]
  }

  hgb = HistGradientBoostingRegressor(random_state=0)
  hgb_grid_search = GridSearchCV(
      hgb, hgb_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
  hgb_grid_search.fit(X_train, y_train_log)

  print(f"\nBest HGB parameters: {hgb_grid_search.best_params_}")
  print(f"Best HGB CV MAE: {-hgb_grid_search.best_score_:.2f}")

  hgb_model = hgb_grid_search.best_estimator_

  y_pred_log = hgb_model.predict(X_test)
  y_pred = np.exp(y_pred_log)

  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  print(f"\nHist Gradient Boosting: MAE: {mae:.2f}, R2: {r2:.3f}")

  # Compute permutation importance
  scorer = make_scorer(lambda y_true, y_pred: mean_absolute_error(
      y_true, np.exp(y_pred)), greater_is_better=False)
  perm_importance = permutation_importance(
      hgb_model, X_test, y_test, n_repeats=10, random_state=0, scoring=scorer)
  feature_names = X.columns
  importance_df = pd.DataFrame({
      'Feature': feature_names,
      'Importance': perm_importance.importances_mean,
      'Std': perm_importance.importances_std
  })
  importance_df = importance_df.sort_values(by='Importance', ascending=False)

  print("\nTop 10 Feature Importances (Permutation):")
  print(importance_df.head(10))

  # Plot feature importance
  plt.figure(figsize=(10, 6))
  plt.barh(importance_df['Feature'][:10], importance_df['Importance']
           [:10], xerr=importance_df['Std'][:10], capsize=5)
  plt.xlabel('Permutation Importance (MAE Impact)')
  plt.title('Top 10 Feature Importances')
  plt.gca().invert_yaxis()
  plt.tight_layout()
  plt.show()

  # Plot
  plt.scatter(y_test, y_pred, alpha=0.5)
  plt.xlabel('Actual Charges')
  plt.ylabel('Predicted Charges')
  plt.title('Predicted vs Actual')
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
