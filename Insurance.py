import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
  """Create an enriched design matrix with informative derived features."""
  feature_engineered_df = df.copy()
  feature_engineered_df['bmi_children'] = feature_engineered_df['bmi'] * \
      feature_engineered_df['children']
  feature_engineered_df['age_children'] = feature_engineered_df['age'] * \
      feature_engineered_df['children']
  feature_engineered_df['bmi_category'] = pd.cut(
      feature_engineered_df['bmi'], bins=[0, 18.5, 25, 30, 35, np.inf],
      labels=['underweight', 'normal', 'overweight', 'obese', 'morbid_obese']
  )
  feature_engineered_df['region_smoker'] = feature_engineered_df['region'] + \
      '_' + feature_engineered_df['smoker']

  X = pd.get_dummies(feature_engineered_df.drop(
      'charges', axis=1), drop_first=True)
  y = feature_engineered_df['charges']

  return X, y

def main():
  insurance = pd.read_csv('DataFiles/insurance.csv')

  # Get engineered features
  feature_engineered_df = insurance.copy()
  feature_engineered_df['bmi_children'] = feature_engineered_df['bmi'] * \
      feature_engineered_df['children']
  feature_engineered_df['age_children'] = feature_engineered_df['age'] * \
      feature_engineered_df['children']
  feature_engineered_df['bmi_category'] = pd.cut(
      feature_engineered_df['bmi'], bins=[0, 18.5, 25, 30, 35, np.inf],
      labels=['underweight', 'normal', 'overweight', 'obese', 'morbid_obese']
  )
  feature_engineered_df['region_smoker'] = feature_engineered_df['region'] + \
      '_' + feature_engineered_df['smoker']

  X = feature_engineered_df.drop('charges', axis=1)
  y = feature_engineered_df['charges']

  # Identify numeric and categorical columns
  numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
  categorical_cols = X.select_dtypes(
      include=['object', 'category']).columns.tolist()

  print(
      f"\nUsing {len(X)} rows with {len(numeric_cols)} numeric + {len(categorical_cols)} categorical features.")

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.1, random_state=0)
  print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

  # Log-transform target for better modeling of skewed charges
  y_train_log = np.log(y_train)
  y_test_log = np.log(y_test)

  # Build preprocessing pipeline with ColumnTransformer
  preprocessor = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numeric_cols),
          ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
      ],
      remainder='passthrough'
  )

  # Create full pipeline with preprocessor and estimator
  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('regressor', HistGradientBoostingRegressor(random_state=0))
  ])

  # HGB hyperparameter tuning
  hgb_param_grid = {
      'regressor__max_iter': [100, 200, 300],
      'regressor__learning_rate': [0.05, 0.1, 0.15],
      'regressor__max_depth': [3, 5, 7],
      'regressor__l2_regularization': [0.0, 0.1, 1.0]
  }

  hgb_grid_search = GridSearchCV(
      pipeline, hgb_param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
  hgb_grid_search.fit(X_train, y_train_log)

  print(f"\nBest HGB parameters: {hgb_grid_search.best_params_}")
  print(f"Best HGB CV MAE (log scale): {-hgb_grid_search.best_score_:.2f}")

  best_model = hgb_grid_search.best_estimator_

  # Cross-validation scores on training set
  cv_mae_scores = -cross_val_score(best_model, X_train,
                                   y_train_log, cv=5, scoring='neg_mean_absolute_error')
  cv_r2_scores = cross_val_score(
      best_model, X_train, y_train_log, cv=5, scoring='r2')
  print(
      f"\n5-Fold CV MAE (log scale): {cv_mae_scores.mean():.2f} (+/- {cv_mae_scores.std():.2f})")
  print(
      f"5-Fold CV R²: {cv_r2_scores.mean():.3f} (+/- {cv_r2_scores.std():.3f})")

  # Predictions on test set
  y_pred_log = best_model.predict(X_test)
  y_pred = np.exp(y_pred_log)

  # Evaluate with both metrics
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  print(f"\nTest Set Metrics:")
  print(f"  MAE: ${mae:.2f}")
  print(f"  R²: {r2:.3f}")

  # Persist best model
  joblib.dump(best_model, 'insurance_model.pkl')
  print(f"\nBest model saved to 'insurance_model.pkl'")

  # Compute permutation importance on preprocessed test data
  X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
  perm_importance = permutation_importance(
      best_model.named_steps['regressor'], X_test_transformed, y_test_log,
      n_repeats=10, random_state=0, scoring='r2')

  # Get feature names after preprocessing
  feature_names_numeric = numeric_cols
  feature_names_cat = best_model.named_steps['preprocessor'].named_transformers_[
      'cat'].get_feature_names_out(categorical_cols).tolist()
  all_feature_names = feature_names_numeric + feature_names_cat

  importance_df = pd.DataFrame({
      'Feature': all_feature_names,
      'Importance': perm_importance.importances_mean,
      'Std': perm_importance.importances_std
  })
  importance_df = importance_df.sort_values(by='Importance', ascending=False)

  print("\nTop 10 Feature Importances (Permutation, R² Impact):")
  print(importance_df.head(10))

  # Plot feature importance
  plt.figure(figsize=(10, 6))
  plt.barh(importance_df['Feature'][:10], importance_df['Importance']
           [:10], xerr=importance_df['Std'][:10], capsize=5)
  plt.xlabel('Permutation Importance (R² Impact)')
  plt.title('Top 10 Feature Importances')
  plt.gca().invert_yaxis()
  plt.tight_layout()
  plt.show()

  # Plot predicted vs actual
  plt.scatter(y_test, y_pred, alpha=0.5)
  plt.xlabel('Actual Charges')
  plt.ylabel('Predicted Charges')
  plt.title('Predicted vs Actual (Test Set)')
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
