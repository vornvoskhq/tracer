"""
Model interpretability module with Python 3.12 compatibility
Provides SHAP-like functionality using LIME and custom feature importance analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ModelInterpreter:
    """
    Model interpretability class compatible with Python 3.12
    Provides SHAP-like functionality using LIME and sklearn's inspection tools
    """
    
    def __init__(self):
        self.lime_explainer = None
        self.feature_names = None
        
    def initialize_lime_explainer(self, X_train: np.ndarray, feature_names: List[str], 
                                 mode: str = 'classification'):
        """
        Initialize LIME explainer for model interpretability
        
        Args:
            X_train: Training data for background
            feature_names: List of feature names
            mode: 'classification' or 'regression'
        """
        self.feature_names = feature_names
        
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                class_names=['Negative', 'Positive'] if mode == 'classification' else None,
                mode=mode,
                discretize_continuous=True,
                random_state=42
            )
            logger.info("LIME explainer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LIME explainer: {e}")
            return False
    
    def get_lime_explanation(self, model: BaseEstimator, X_instance: np.ndarray, 
                           num_features: int = 10) -> Dict[str, Any]:
        """
        Get LIME explanation for a single instance
        
        Args:
            model: Trained model
            X_instance: Single instance to explain
            num_features: Number of top features to show
            
        Returns:
            Dictionary with explanation data
        """
        if self.lime_explainer is None:
            logger.error("LIME explainer not initialized")
            return {}
        
        try:
            # Get explanation
            explanation = self.lime_explainer.explain_instance(
                X_instance.flatten(),
                model.predict_proba,
                num_features=num_features
            )
            
            # Extract feature importance
            feature_importance = {}
            explanation_list = explanation.as_list()
            
            # Debug: Check if explanation returns anything
            if not explanation_list:
                logger.warning("LIME explanation returned empty list")
                return {}
            
            for feature_idx, importance in explanation_list:
                if isinstance(feature_idx, str):
                    feature_name = feature_idx
                else:
                    feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                feature_importance[feature_name] = importance
            
            # Debug: Check if all importances are zero
            non_zero_count = sum(1 for v in feature_importance.values() if abs(v) > 1e-10)
            if non_zero_count == 0:
                logger.warning(f"LIME explanation returned all zero importances for {len(feature_importance)} features")
            
            return {
                'feature_importance': feature_importance,
                'explanation_object': explanation,
                'prediction_proba': model.predict_proba(X_instance.reshape(1, -1))[0]
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {}
    
    def get_permutation_importance(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str], n_repeats: int = 10) -> Dict[str, float]:
        """
        Calculate permutation importance for all features
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary with feature importance scores
        """
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
            )
            
            # Create importance dictionary
            importance_dict = {}
            for i, feature_name in enumerate(feature_names):
                importance_dict[feature_name] = perm_importance.importances_mean[i]
            
            logger.info("Permutation importance calculated successfully")
            return importance_dict
            
        except Exception as e:
            logger.error(f"Permutation importance calculation failed: {e}")
            return {}
    
    def get_feature_importance_summary(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str], sample_size: int = 100) -> Dict[str, Any]:
        """
        Get comprehensive feature importance summary
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            sample_size: Number of samples for LIME analysis
            
        Returns:
            Dictionary with comprehensive importance analysis
        """
        results = {
            'linear_coefficients': {},
            'permutation_importance': {},
            'lime_importance': {},
            'feature_statistics': {}
        }
        
        # 1. Linear model coefficients (if available)
        if hasattr(model, 'coef_'):
            results['linear_coefficients'] = dict(zip(feature_names, model.coef_[0]))
            logger.info("Linear coefficients extracted")
        
        # 2. Permutation importance
        results['permutation_importance'] = self.get_permutation_importance(
            model, X, y, feature_names
        )
        
        # 3. LIME importance (sample-based)
        if self.initialize_lime_explainer(X, feature_names):
            lime_importances = []
            sample_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
            
            for idx in sample_indices:
                lime_result = self.get_lime_explanation(model, X[idx])
                if lime_result and 'feature_importance' in lime_result:
                    lime_importances.append(lime_result['feature_importance'])
            
            # Average LIME importances
            if lime_importances:
                avg_lime_importance = {}
                for feature in feature_names:
                    importances = [imp.get(feature, 0) for imp in lime_importances]
                    avg_lime_importance[feature] = np.mean(importances)
                results['lime_importance'] = avg_lime_importance
                logger.info(f"LIME importance calculated from {len(lime_importances)} samples")
        
        # 4. Feature statistics
        for i, feature in enumerate(feature_names):
            feature_data = X[:, i]
            results['feature_statistics'][feature] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'correlation_with_target': float(np.corrcoef(feature_data, y)[0, 1]) if len(np.unique(y)) > 1 else 0.0
            }
        
        return results
    
    def plot_feature_importance_comparison(self, importance_dict: Dict[str, Dict[str, float]], 
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of different feature importance methods
        
        Args:
            importance_dict: Dictionary with different importance methods
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Prepare data for plotting
        methods = []
        features = []
        importances = []
        
        # Get all unique features
        all_features = set()
        for method_importances in importance_dict.values():
            all_features.update(method_importances.keys())
        all_features = sorted(list(all_features))
        
        # Prepare data
        for method, method_importances in importance_dict.items():
            for feature in all_features:
                methods.append(method)
                features.append(feature)
                importances.append(method_importances.get(feature, 0))
        
        # Create DataFrame for easier plotting
        df_plot = pd.DataFrame({
            'Method': methods,
            'Feature': features,
            'Importance': importances
        })
        
        # Create plot
        fig, axes = plt.subplots(len(importance_dict), 1, figsize=(12, 4 * len(importance_dict)))
        if len(importance_dict) == 1:
            axes = [axes]
        
        for i, (method, method_importances) in enumerate(importance_dict.items()):
            ax = axes[i]
            
            # Sort features by importance
            sorted_items = sorted(method_importances.items(), key=lambda x: abs(x[1]), reverse=True)
            features_sorted = [item[0] for item in sorted_items[:15]]  # Top 15 features
            importances_sorted = [item[1] for item in sorted_items[:15]]
            
            # Create horizontal bar plot
            colors = ['green' if imp > 0 else 'red' for imp in importances_sorted]
            bars = ax.barh(range(len(features_sorted)), importances_sorted, color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(features_sorted)))
            ax.set_yticklabels(features_sorted)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Feature Importance - {method.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels on bars
            for j, (bar, importance) in enumerate(zip(bars, importances_sorted)):
                ax.text(importance + (0.01 if importance > 0 else -0.01), j, 
                       f'{importance:.3f}', va='center', 
                       ha='left' if importance > 0 else 'right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance comparison plot saved to {save_path}")
        
        return fig
    
    def create_feature_importance_report(self, importance_summary: Dict[str, Any], 
                                       top_n: int = 10) -> str:
        """
        Create a text report of feature importance analysis
        
        Args:
            importance_summary: Output from get_feature_importance_summary
            top_n: Number of top features to include in report
            
        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 60)
        report.append("FEATURE IMPORTANCE ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Linear coefficients
        if importance_summary['linear_coefficients']:
            report.append("\n1. LINEAR MODEL COEFFICIENTS:")
            report.append("-" * 30)
            linear_sorted = sorted(importance_summary['linear_coefficients'].items(), 
                                 key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, coef) in enumerate(linear_sorted[:top_n]):
                direction = "Positive" if coef > 0 else "Negative"
                report.append(f"{i+1:2d}. {feature:<25} {coef:8.4f} ({direction})")
        
        # Permutation importance
        if importance_summary['permutation_importance']:
            report.append("\n2. PERMUTATION IMPORTANCE:")
            report.append("-" * 30)
            perm_sorted = sorted(importance_summary['permutation_importance'].items(), 
                               key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, importance) in enumerate(perm_sorted[:top_n]):
                report.append(f"{i+1:2d}. {feature:<25} {importance:8.4f}")
        
        # LIME importance
        if importance_summary['lime_importance']:
            report.append("\n3. LIME IMPORTANCE (Average):")
            report.append("-" * 30)
            lime_sorted = sorted(importance_summary['lime_importance'].items(), 
                               key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, importance) in enumerate(lime_sorted[:top_n]):
                direction = "Positive" if importance > 0 else "Negative"
                report.append(f"{i+1:2d}. {feature:<25} {importance:8.4f} ({direction})")
        
        # Feature statistics summary
        if importance_summary['feature_statistics']:
            report.append("\n4. TOP FEATURES CORRELATION WITH TARGET:")
            report.append("-" * 30)
            corr_sorted = sorted(importance_summary['feature_statistics'].items(), 
                               key=lambda x: abs(x[1]['correlation_with_target']), reverse=True)
            for i, (feature, stats) in enumerate(corr_sorted[:top_n]):
                corr = stats['correlation_with_target']
                direction = "Positive" if corr > 0 else "Negative"
                report.append(f"{i+1:2d}. {feature:<25} {corr:8.4f} ({direction})")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

# Compatibility wrapper for SHAP-like interface
class SHAPCompatibilityWrapper:
    """
    Wrapper to provide SHAP-like interface using LIME and permutation importance
    """
    
    def __init__(self, model: BaseEstimator, X_background: np.ndarray, feature_names: List[str]):
        self.model = model
        self.interpreter = ModelInterpreter()
        self.feature_names = feature_names
        
        # Initialize LIME explainer
        self.interpreter.initialize_lime_explainer(X_background, feature_names)
    
    def shap_values(self, X: np.ndarray, max_samples: int = 100) -> np.ndarray:
        """
        SHAP-like interface using LIME explanations
        
        Args:
            X: Data to explain
            max_samples: Maximum number of samples to process
            
        Returns:
            Array of feature importance values (LIME-based)
        """
        n_samples = min(len(X), max_samples)
        n_features = len(self.feature_names)
        
        shap_like_values = np.zeros((n_samples, n_features))
        
        # Track successful explanations for debugging
        successful_explanations = 0
        
        for i in range(n_samples):
            try:
                lime_result = self.interpreter.get_lime_explanation(self.model, X[i])
                if lime_result and 'feature_importance' in lime_result:
                    feature_importance = lime_result['feature_importance']
                    if any(abs(v) > 0 for v in feature_importance.values()):  # Check if any non-zero values
                        successful_explanations += 1
                        for j, feature_name in enumerate(self.feature_names):
                            shap_like_values[i, j] = feature_importance.get(feature_name, 0)
                    else:
                        # If LIME returns all zeros, use linear coefficients as fallback
                        if hasattr(self.model, 'coef_'):
                            for j, feature_name in enumerate(self.feature_names):
                                if j < len(self.model.coef_[0]):
                                    shap_like_values[i, j] = self.model.coef_[0][j]
                else:
                    # If LIME explanation fails, use linear coefficients as fallback
                    if hasattr(self.model, 'coef_'):
                        for j, feature_name in enumerate(self.feature_names):
                            if j < len(self.model.coef_[0]):
                                shap_like_values[i, j] = self.model.coef_[0][j]
            except Exception as e:
                # If any error occurs, use linear coefficients as fallback
                if hasattr(self.model, 'coef_'):
                    for j, feature_name in enumerate(self.feature_names):
                        if j < len(self.model.coef_[0]):
                            shap_like_values[i, j] = self.model.coef_[0][j]
        
        # Log the success rate for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"LIME explanations successful for {successful_explanations}/{n_samples} samples")
        
        return shap_like_values
    
    def summary_plot(self, shap_values: np.ndarray, X: np.ndarray, 
                    feature_names: List[str], plot_type: str = "bar", 
                    max_display: int = 10, show: bool = True) -> plt.Figure:
        """
        SHAP-like summary plot using LIME values
        """
        # Calculate mean absolute importance
        mean_importance = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        sorted_indices = np.argsort(mean_importance)[::-1][:max_display]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_importance = mean_importance[sorted_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max_display * 0.5))
        
        if plot_type == "bar":
            bars = ax.barh(range(len(sorted_features)), sorted_importance, alpha=0.7)
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel('Mean |LIME Importance|')
            ax.set_title('Feature Importance (LIME-based)')
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, sorted_importance)):
                ax.text(importance + 0.001, i, f'{importance:.3f}', 
                       va='center', ha='left', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig