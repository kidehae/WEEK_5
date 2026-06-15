# src/modeling_pipeline.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model_cv(X, y, model, clf_name="Model", apply_smote=False):
    """
    Executes a 5-Fold Stratified Cross-Validation loop.
    Applies SMOTE strictly inside the training folds to ensure no data leakage.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Feature matrix.
    y : pd.Series or np.array
        Target binary labels.
    model : estimator
        Scikit-learn compatible classifier instance.
    apply_smote : bool
        Whether to balance the training fold using SMOTE.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_pr_scores = []
    f1_scores = []
    conf_matrices = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Apply resampling strictly within the training split only
        if apply_smote:
            smote = SMOTE(sampling_strategy=0.3, random_state=42)
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
            
        # Fit the classifier
        model.fit(X_tr, y_tr)
        
        # Predict class probabilities and hard classes
        y_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)
        y_preds = model.predict(X_val)
        
        # Compute metrics
        precision, recall, _ = precision_recall_curve(y_val, y_probs)
        auc_pr = auc(recall, precision)
        f1 = f1_score(y_val, y_preds)
        cm = confusion_matrix(y_val, y_preds)
        
        auc_pr_scores.append(auc_pr)
        f1_scores.append(f1)
        conf_matrices.append(cm)
        
    mean_auc_pr = np.mean(auc_pr_scores)
    std_auc_pr = np.std(auc_pr_scores)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    total_cm = np.sum(conf_matrices, axis=0) # Sum across folds for global view
    
    logging.info(f"[{clf_name}] CV AUC-PR: {mean_auc_pr:.4f} ± {std_auc_pr:.4f} | Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    return {
        "model_name": clf_name,
        "mean_auc_pr": mean_auc_pr,
        "std_auc_pr": std_auc_pr,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "confusion_matrix": total_cm,
        "fitted_model": model
    }