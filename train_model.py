"""
Title: LightGBM and TreeSHAP Pipeline for Sustainable Supply Chain Risk Assessment
Description: Trains a LightGBM classification model on supply chain transaction features,
             evaluates risk probabilities, and generates local/global TreeSHAP explanations.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report

def load_and_preprocess_data():
    """
    Simulates loading and preprocessing of the DataCo Global Supply Chain dataset.
    In practice, load your cleaned CSV dataset here.
    """
    np.random.seed(42)
    n_samples = 5000
    n_features = 12
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i+1}" for i in range(n_features)]
    )
   
    y = np.random.randint(0, 2, size=n_samples)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """
    Trains the LightGBM model with optimized hyperparameters for risk classification.
    """
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    return model

def evaluate_and_explain(model, X_test, y_test):
    """
    Evaluates model performance and computes TreeSHAP explanations for interpretability.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"--- Model Evaluation Benchmarks ---")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    return shap_values

if __name__ == "__main__":
    print("Initializing Supply Chain Risk Pipeline...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Training LightGBM Classifier...")
    model = train_lightgbm_model(X_train, y_train, X_test, y_test)
    
    print("Evaluating Performance & Generating TreeSHAP Explanations...")
    shap_values = evaluate_and_explain(model, X_test, y_test)
    print("Pipeline execution completed successfully.")
