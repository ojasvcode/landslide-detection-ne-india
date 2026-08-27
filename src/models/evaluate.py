import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, 
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates ML models for landslide detection."""

    def __init__(self):
        logger.info("ModelEvaluator initialized.")

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str = 'Model') -> Dict[str, Any]:
        """Computes comprehensive evaluation metrics."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        }
        
        logger.info(f"--- {model_name} Evaluation ---")
        logger.info(f"AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
        return metrics

    def plot_roc_curve(self, models_dict: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> plt.Figure:
        """Plots ROC curves for multiple models on the same axes."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, model in models_dict.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
            
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> plt.Figure:
        """Plots a heatmap of the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['No Landslide', 'Landslide'], 
                    yticklabels=['No Landslide', 'Landslide'])
        ax.set_title(f'Confusion Matrix: {model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        return fig

    def plot_precision_recall_curve(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> plt.Figure:
        """Plots Precision-Recall curve."""
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=model_name, color='purple')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve: {model_name}')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def compare_models(self, models_dict: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Compares models side by side and returns a summary DataFrame."""
        results = []
        for name, model in models_dict.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            results.append({
                'model_name': name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'roc_auc': metrics['roc_auc']
            })
        return pd.DataFrame(results)

    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """Performs k-fold cross validation."""
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        return {
            'cv_scores': scores.tolist(),
            'mean_auc': float(np.mean(scores)),
            'std_auc': float(np.std(scores))
        }
