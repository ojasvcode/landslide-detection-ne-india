import os
import joblib
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE

from .evaluate import ModelEvaluator
from src.feature_engineering.composite_features import CompositeFeatureBuilder

logger = logging.getLogger(__name__)

class LandslideModelTrainer:
    """Trains machine learning models for landslide prediction."""

    def __init__(self, model_dir: str = 'data/processed/models', random_state: int = 42):
        self.model_dir = model_dir
        self.random_state = random_state
        self.evaluator = ModelEvaluator()
        self.feature_names = CompositeFeatureBuilder.FEATURE_COLUMNS
        
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"LandslideModelTrainer initialized. Model directory: {self.model_dir}")

    def prepare_data(self, feature_df: pd.DataFrame, target_col: str = 'label') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """Splits, balances, and scales data."""
        X = feature_df[self.feature_names]
        y = feature_df[target_col]
        
        # Split
        X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_df)
        X_test_scaled = scaler.transform(X_test_df)
        
        # Balance with SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        
        logger.info(f"Data preparation complete. SMOTE adjusted train size from {X_train_scaled.shape[0]} to {X_train_res.shape[0]}.")
        return X_train_res, X_test_scaled, y_train_res.values, y_test.values, scaler

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, tune: bool = True) -> RandomForestClassifier:
        """Trains a Random Forest model."""
        base_rf = RandomForestClassifier(random_state=self.random_state)
        
        if not tune:
            base_rf.fit(X_train, y_train)
            return base_rf
            
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        logger.info("Tuning Random Forest...")
        rf_random = RandomizedSearchCV(estimator=base_rf, param_distributions=param_grid, 
                                       n_iter=10, cv=5, verbose=1, random_state=self.random_state, n_jobs=-1)
        rf_random.fit(X_train, y_train)
        logger.info(f"Best RF params: {rf_random.best_params_}")
        return rf_random.best_estimator_

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, tune: bool = True) -> xgb.XGBClassifier:
        """Trains an XGBoost model."""
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        base_xgb = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, 
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        if not tune:
            base_xgb.fit(X_train, y_train)
            return base_xgb
            
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 8, 12],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        logger.info("Tuning XGBoost...")
        xgb_random = RandomizedSearchCV(estimator=base_xgb, param_distributions=param_grid, 
                                        n_iter=10, cv=5, verbose=1, random_state=self.random_state, n_jobs=-1)
        xgb_random.fit(X_train, y_train)
        logger.info(f"Best XGB params: {xgb_random.best_params_}")
        return xgb_random.best_estimator_

    def train_ensemble(self, rf_model: RandomForestClassifier, xgb_model: xgb.XGBClassifier, 
                       X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
        """Trains a soft voting ensemble of RF and XGBoost."""
        logger.info("Training Voting Classifier ensemble...")
        ensemble = VotingClassifier(
            estimators=[('rf', rf_model), ('xgb', xgb_model)],
            voting='soft'
        )
        ensemble.fit(X_train, y_train)
        return ensemble

    def train_all(self, feature_df: pd.DataFrame, target_col: str = 'label') -> Dict[str, Any]:
        """Runs the full training pipeline."""
        logger.info("Starting full model training pipeline.")
        X_train, X_test, y_train, y_test, scaler = self.prepare_data(feature_df, target_col)
        
        rf_model = self.train_random_forest(X_train, y_train, tune=True)
        xgb_model = self.train_xgboost(X_train, y_train, tune=True)
        ensemble_model = self.train_ensemble(rf_model, xgb_model, X_train, y_train)
        
        models = {
            'RandomForest': rf_model,
            'XGBoost': xgb_model,
            'Ensemble': ensemble_model
        }
        
        metrics = {}
        best_auc = 0
        best_model_name = ""
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            model_metrics = self.evaluator.evaluate_model(model, X_test, y_test, name)
            metrics[name] = model_metrics
            
            if model_metrics['roc_auc'] > best_auc:
                best_auc = model_metrics['roc_auc']
                best_model_name = name
                
        logger.info(f"Best model identified: {best_model_name} with AUC {best_auc:.4f}")
        self.save_model(models[best_model_name], scaler, 'best_landslide_model')
        
        return {
            'models': models,
            'metrics': metrics,
            'best_model_name': best_model_name,
            'scaler': scaler
        }

    def save_model(self, model: Any, scaler: StandardScaler, model_name: str) -> str:
        """Saves model and scaler to disk."""
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
        features_path = os.path.join(self.model_dir, f"{model_name}_features.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(self.feature_names, features_path)
        
        logger.info(f"Saved model to {model_path}")
        return model_path

    def load_model(self, model_name: str) -> Tuple[Any, StandardScaler, list]:
        """Loads model, scaler, and feature names from disk."""
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
        features_path = os.path.join(self.model_dir, f"{model_name}_features.joblib")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        
        return model, scaler, features
