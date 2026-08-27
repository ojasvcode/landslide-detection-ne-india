import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Provides SHAP-based explainability for models."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        # Note: TreeExplainer requires Tree-based models (RF, XGBoost)
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Initialized SHAP TreeExplainer.")
        except Exception as e:
            logger.warning(f"Could not initialize TreeExplainer, falling back to KernelExplainer. Error: {e}")
            # Requires a background dataset to use KernelExplainer properly, 
            # omitted here for brevity as requirements imply tree models (RF, XGB).
            self.explainer = None 

    def compute_shap_values(self, X_data: np.ndarray) -> np.ndarray:
        """Computes SHAP values."""
        if not self.explainer:
            raise ValueError("Explainer not initialized properly.")
        shap_values = self.explainer.shap_values(X_data)
        # For some models/versions, shap_values is a list for multi-class. Extract positive class if so.
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values

    def plot_shap_summary(self, X_data: np.ndarray, max_display: int = 14) -> plt.Figure:
        """Plots SHAP beeswarm/summary plot."""
        shap_values = self.compute_shap_values(X_data)
        fig = plt.figure()
        shap.summary_plot(shap_values, X_data, feature_names=self.feature_names, 
                          max_display=max_display, show=False)
        plt.tight_layout()
        return fig

    def plot_shap_bar(self, X_data: np.ndarray) -> plt.Figure:
        """Plots SHAP feature importance bar chart."""
        shap_values = self.compute_shap_values(X_data)
        fig = plt.figure()
        shap.summary_plot(shap_values, X_data, feature_names=self.feature_names, 
                          plot_type='bar', show=False)
        plt.tight_layout()
        return fig

    def explain_single_prediction(self, features: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Provides SHAP breakdown for a single prediction."""
        if not self.explainer:
            raise ValueError("Explainer not initialized.")
            
        fn = feature_names if feature_names else self.feature_names
        
        # Calculate
        shap_values = self.compute_shap_values(features)
        base_value = self.explainer.expected_value
        if isinstance(base_value, list) or isinstance(base_value, np.ndarray):
            base_value = base_value[1] # positive class
            
        contributions = dict(zip(fn, shap_values[0]))
        
        return {
            'base_value': float(base_value),
            'shap_values': shap_values[0].tolist(),
            'feature_contributions': contributions
        }

    def plot_feature_importance(self, model: Optional[Any] = None) -> plt.Figure:
        """Standard model feature importance (not SHAP)."""
        target_model = model if model else self.model
        if not hasattr(target_model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute.")
            
        importances = target_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(importances)), importances[indices])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([self.feature_names[i] for i in indices], rotation=90)
        ax.set_title("Standard Feature Importances")
        plt.tight_layout()
        return fig

    def get_top_contributing_features(self, shap_values_single: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        """Returns top N features contributing to a specific prediction."""
        # Pair feature names with their SHAP values
        paired = list(zip(self.feature_names, shap_values_single))
        # Sort by absolute SHAP value magnitude
        paired.sort(key=lambda x: abs(x[1]), reverse=True)
        return paired[:n]
