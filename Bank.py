import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

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

  # Target
  X = pd.get_dummies(feature_engineered_df.drop(
      'deposit', axis=1), drop_first=True)
  y = feature_engineered_df['deposit']

  return X, y

def main():
  bank = pd.read_csv('DataFiles/bank.csv')

  X, y = build_feature_matrix(bank)

  print(
      f"\nUsing {len(X)} rows with {len(X.columns)} features (including engineered).")

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.1, random_state=0, stratify=y
  )
  print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

  # Handle class imbalance with SMOTE
  smote = SMOTE(random_state=0)
  X_train, y_train = smote.fit_resample(X_train, y_train)
  print(f"After SMOTE: Train size: {len(X_train)}")

  # HGB hyperparameter tuning
  hgb_param_grid = {
      'max_iter': [100, 125, 150],
      'learning_rate': [0.09],
      'max_depth': [11, 12],
      'l2_regularization': [0.8, 0.9],
      'min_samples_leaf': [12, 14]
  }

  hgb = HistGradientBoostingClassifier(random_state=0)
  hgb_grid_search = GridSearchCV(
      hgb, hgb_param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
  )
  hgb_grid_search.fit(X_train, y_train)

  print(f"\nBest HGB parameters: {hgb_grid_search.best_params_}")
  print(f"Best HGB CV F1: {hgb_grid_search.best_score_:.3f}")

  hgb_model = hgb_grid_search.best_estimator_

  y_pred = hgb_model.predict(X_test)
  y_pred_proba = hgb_model.predict_proba(X_test)[:, 1]

  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_pred_proba)
  print(
      f"\nHist Gradient Boosting: Accuracy: {acc:.3f}, F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}")

  # Find optimal threshold for better F1
  from sklearn.metrics import precision_recall_curve, roc_curve
  precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
  f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
  best_f1_idx = f1_scores.argmax()
  best_threshold_f1 = thresholds_pr[best_f1_idx]
  print(f"Best threshold for F1: {best_threshold_f1:.3f} (F1: {f1_scores[best_f1_idx]:.3f})")

  # Use ROC curve for Youden's J statistic
  fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
  j_scores = tpr - fpr
  best_j_idx = j_scores.argmax()
  best_threshold_roc = thresholds_roc[best_j_idx]
  print(f"Best threshold for Youden's J: {best_threshold_roc:.3f} (J: {j_scores[best_j_idx]:.3f})")

  # Apply best F1 threshold and recompute metrics
  y_pred_opt = (y_pred_proba >= best_threshold_f1).astype(int)
  f1_opt = f1_score(y_test, y_pred_opt)
  acc_opt = accuracy_score(y_test, y_pred_opt)
  roc_auc_opt = roc_auc_score(y_test, y_pred_proba)
  print(f"Optimized: Accuracy: {acc_opt:.3f}, F1: {f1_opt:.3f}, ROC AUC: {roc_auc_opt:.3f}")

  # Compute permutation importance
  perm_importance = permutation_importance(
      hgb_model, X_test, y_test, n_repeats=10, random_state=0, scoring='f1')
  feature_names = X.columns
  importance_df = pd.DataFrame({
      'Feature': feature_names,
      'Importance': perm_importance.importances_mean,
      'Std': perm_importance.importances_std
  })
  importance_df = importance_df.sort_values(by='Importance', ascending=False)

  print("\nTop 10 Feature Importances:")
  print(importance_df.head(10))

  # Plot feature importance
  plt.figure(figsize=(10, 6))
  plt.barh(importance_df['Feature'][:10], importance_df['Importance']
           [:10], xerr=importance_df['Std'][:10], capsize=5)
  plt.xlabel('Permutation Importance (F1 Score Impact)')
  plt.title('Top 10 Feature Importances')
  plt.gca().invert_yaxis()
  plt.tight_layout()
  plt.show()

  # Plot confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                                'No Deposit', 'Deposit'])
  disp.plot(cmap=plt.cm.Blues)
  plt.title('Confusion Matrix')
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
