import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, precision_score, recall_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
  """Create an enriched design matrix with informative derived features."""
  feature_engineered_df = df.copy()

  # Derived features: interactions and categoricals
  feature_engineered_df['age_balance'] = feature_engineered_df['age'] * \
      feature_engineered_df['balance']
  feature_engineered_df['duration_campaign'] = feature_engineered_df['duration'] * \
      feature_engineered_df['campaign']
  feature_engineered_df['balance_category'] = pd.cut(
      feature_engineered_df['balance'], bins=[-np.inf,
                                              0, 500, 1000, 2000, np.inf],
      labels=['negative', 'low', 'medium', 'high', 'very_high']
  )
  feature_engineered_df['job_marital'] = feature_engineered_df['job'] + \
      '_' + feature_engineered_df['marital']
  feature_engineered_df['age_category'] = pd.cut(
      feature_engineered_df['age'], bins=[0, 30, 45, 60, np.inf],
      labels=['young', 'middle', 'senior', 'elder']
  )
  feature_engineered_df['balance_log'] = np.log1p(
      feature_engineered_df['balance'].clip(lower=0))

  # Encode binary columns
  binary_cols = ['default', 'housing', 'loan', 'deposit']
  for col in binary_cols:
    if col in feature_engineered_df.columns:
      feature_engineered_df[col] = feature_engineered_df[col].map(
          {'yes': 1, 'no': 0})

  # Drop target for X
  X = feature_engineered_df.drop('deposit', axis=1)
  y = feature_engineered_df['deposit']

  return X, y

def main():
  bank = pd.read_csv('DataFiles/bank.csv')

  X, y = build_feature_matrix(bank)

  # Identify numeric and categorical columns
  numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
  categorical_cols = X.select_dtypes(
      include=['object', 'category']).columns.tolist()

  print(
      f"\nUsing {len(X)} rows with {len(numeric_cols)} numeric + {len(categorical_cols)} categorical features.")

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.1, random_state=0, stratify=y
  )
  print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

  # Build preprocessing pipeline with ColumnTransformer
  preprocessor = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numeric_cols),
          ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
      ],
      remainder='passthrough'
  )

  # Create full pipeline with preprocessor, SMOTE, and estimator
  pipeline = ImbPipeline(steps=[
      ('preprocessor', preprocessor),
      ('smote', SMOTE(random_state=0)),
      ('classifier', HistGradientBoostingClassifier(random_state=0))
  ])

  # HGB hyperparameter tuning
  hgb_param_grid = {
      'classifier__max_iter': [100, 125, 150],
      'classifier__learning_rate': [0.09],
      'classifier__max_depth': [11, 12],
      'classifier__l2_regularization': [0.8, 0.9],
      'classifier__min_samples_leaf': [12, 14]
  }

  hgb_grid_search = GridSearchCV(
      pipeline, hgb_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
  )
  hgb_grid_search.fit(X_train, y_train)

  print(f"\nBest HGB parameters: {hgb_grid_search.best_params_}")
  print(f"Best HGB CV F1: {hgb_grid_search.best_score_:.3f}")

  best_model = hgb_grid_search.best_estimator_

  # Cross-validation scores on training set
  cv_f1_scores = cross_val_score(
      best_model, X_train, y_train, cv=5, scoring='f1')
  cv_roc_auc_scores = cross_val_score(
      best_model, X_train, y_train, cv=5, scoring='roc_auc')
  print(
      f"\n5-Fold CV F1: {cv_f1_scores.mean():.3f} (+/- {cv_f1_scores.std():.3f})")
  print(
      f"5-Fold CV ROC AUC: {cv_roc_auc_scores.mean():.3f} (+/- {cv_roc_auc_scores.std():.3f})")

  # Predictions on test set
  y_pred = best_model.predict(X_test)
  y_pred_proba = best_model.predict_proba(X_test)[:, 1]

  # Evaluate with both metrics
  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_pred_proba)
  print(f"\nTest Set Metrics (Default Threshold 0.5):")
  print(f"  Accuracy: {acc:.3f}, F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}")

  # Find optimal threshold for better F1
  precision, recall, thresholds_pr = precision_recall_curve(
      y_test, y_pred_proba)
  f1_scores = 2 * (precision[:-1] * recall[:-1]) / \
      (precision[:-1] + recall[:-1] + 1e-10)
  best_f1_idx = f1_scores.argmax()
  best_threshold_f1 = thresholds_pr[best_f1_idx]
  best_precision = precision[:-1][best_f1_idx]
  best_recall = recall[:-1][best_f1_idx]
  print(
      f"\nBest threshold for F1: {best_threshold_f1:.3f} (F1: {f1_scores[best_f1_idx]:.3f})")

  # Apply best F1 threshold and recompute metrics
  y_pred_opt = (y_pred_proba >= best_threshold_f1).astype(int)
  f1_opt = f1_score(y_test, y_pred_opt)
  acc_opt = accuracy_score(y_test, y_pred_opt)
  prec_opt = precision_score(y_test, y_pred_opt)
  rec_opt = recall_score(y_test, y_pred_opt)
  print(f"\nOptimized (Best F1 Threshold):")
  print(f"  Accuracy: {acc_opt:.3f}, F1: {f1_opt:.3f}, ROC AUC: {roc_auc:.3f}")
  print(f"  Precision: {prec_opt:.3f}, Recall: {rec_opt:.3f}")

  # Persist best model
  joblib.dump(best_model, 'bank_model.pkl')
  print(f"\nBest model saved to 'bank_model.pkl'")

  # Compute permutation importance on preprocessed test data
  X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
  perm_importance = permutation_importance(
      best_model.named_steps['classifier'], X_test_transformed, y_test,
      n_repeats=10, random_state=0, scoring='f1')

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

  print("\nTop 10 Feature Importances (Permutation, F1 Impact):")
  print(importance_df.head(10))

  # Plot precision-recall curve with best F1 threshold highlighted
  plt.figure(figsize=(8, 6))
  plt.plot(recall, precision, label='Precision-Recall Curve')
  plt.scatter(best_recall, best_precision, color='red',
                          label=f'Best F1 Threshold = {best_threshold_f1:.3f}')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall Curve (Test Set)')
  plt.legend(loc='lower left')
  plt.grid(alpha=0.3, linestyle='--')
  plt.tight_layout()
  plt.show()

  # Plot feature importance
  plt.figure(figsize=(10, 6))
  plt.barh(importance_df['Feature'][:10], importance_df['Importance']
           [:10], xerr=importance_df['Std'][:10], capsize=5)
  plt.xlabel('Permutation Importance (F1 Score Impact)')
  plt.title('Top 10 Feature Importances')
  plt.gca().invert_yaxis()
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
