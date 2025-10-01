import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class CropYieldPredictor:

    def __init__(self, model_dir: str = "models"):
        """
        Initialize the Crop Yield Predictor.
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.model_scores = {}
        self.best_model: Optional[str] = None
        self.feature_importance = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize various machine learning models."""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Train all models and evaluate their performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target
            
        Returns:
            Dictionary of model scores
        """
        print("Training machine learning models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store scores
            self.model_scores[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
            
            # Store feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  CV R²: {cv_mean:.4f} (±{cv_std:.4f})")
        
        # Select best model based on test R² score
        best_model_name = max(self.model_scores.keys(), 
                             key=lambda x: self.model_scores[x]['test_r2'])
        self.best_model = best_model_name
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   Test R²: {self.model_scores[best_model_name]['test_r2']:.4f}")
        
        return self.model_scores
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_name: str = 'random_forest') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model to tune
            
        Returns:
            Best parameters and score
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ridge_regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso_regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
        }
        
        if model_name not in param_grids:
            print(f"No hyperparameter grid defined for {model_name}")
            return {}
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Features for prediction
            model_name: Name of the model to use (default: best model)
            
        Returns:
            Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model available for prediction")
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)
    
    def predict_with_confidence(self, X: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals (for ensemble models).
        
        Args:
            X: Features for prediction
            model_name: Name of the model to use (default: best model)
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model available for prediction")
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # For ensemble models, get individual tree predictions
        if hasattr(model, 'estimators_'):
            predictions = []
            for estimator in model.estimators_:
                predictions.append(estimator.predict(X))
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            return {
                'predictions': mean_pred,
                'confidence_lower': mean_pred - 1.96 * std_pred,
                'confidence_upper': mean_pred + 1.96 * std_pred,
                'std': std_pred
            }
        else:
            # For non-ensemble models, just return predictions
            predictions = model.predict(X)
            return {
                'predictions': predictions,
                'confidence_lower': predictions,
                'confidence_upper': predictions,
                'std': np.zeros_like(predictions)
            }
    
    def get_feature_importance(self, model_name: Optional[str] = None, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name: Name of the model (default: best model)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            if self.best_model is None:
                print("No model available for feature importance")
                return pd.DataFrame()
            model_name = self.best_model
        
        if model_name not in self.feature_importance:
            print(f"No feature importance available for {model_name}")
            return pd.DataFrame()
        
        importance = self.feature_importance[model_name]
        feature_names = [f"Feature_{i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def plot_model_performance(self, save_path: Optional[str] = None):
        """
        Plot model performance comparison.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.model_scores:
            print("No model scores available. Train models first.")
            return
        
        # Extract scores
        models = list(self.model_scores.keys())
        test_r2 = [self.model_scores[model]['test_r2'] for model in models]
        test_rmse = [self.model_scores[model]['test_rmse'] for model in models]
        cv_mean = [self.model_scores[model]['cv_mean'] for model in models]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # R² Score comparison
        axes[0].bar(models, test_r2, color='skyblue', alpha=0.7)
        axes[0].set_title('Model Performance - R² Score')
        axes[0].set_ylabel('R² Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[1].bar(models, test_rmse, color='lightcoral', alpha=0.7)
        axes[1].set_title('Model Performance - RMSE')
        axes[1].set_ylabel('RMSE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Cross-validation scores
        axes[2].bar(models, cv_mean, color='lightgreen', alpha=0.7)
        axes[2].set_title('Cross-Validation R² Score')
        axes[2].set_ylabel('CV R² Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model_name: Optional[str] = None, top_n: int = 10, 
                              save_path: Optional[str] = None):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model_name: Name of the model (default: best model)
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        importance_df = self.get_feature_importance(model_name, top_n)
        
        if importance_df.empty:
            return
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Feature Importance - {model_name or self.best_model}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_models(self, prefix: str = "crop_yield"):
        """
        Save trained models to disk.
        
        Args:
            prefix: Prefix for model filenames
        """
        print("Saving trained models...")
        
        for name, model in self.models.items():
            filename = os.path.join(self.model_dir, f"{prefix}_{name}.joblib")
            joblib.dump(model, filename)
            print(f"  Saved {name} to {filename}")
        
        # Save model scores
        scores_file = os.path.join(self.model_dir, f"{prefix}_scores.json")
        import json
        with open(scores_file, 'w') as f:
            json.dump(self.model_scores, f, indent=2)
        print(f"  Saved model scores to {scores_file}")
    
    def load_models(self, prefix: str = "crop_yield"):
        """
        Load trained models from disk.
        
        Args:
            prefix: Prefix for model filenames
        """
        print("Loading trained models...")
        
        for name in self.models.keys():
            filename = os.path.join(self.model_dir, f"{prefix}_{name}.joblib")
            if os.path.exists(filename):
                self.models[name] = joblib.load(filename)
                print(f"  Loaded {name} from {filename}")
            else:
                print(f"  Model file not found: {filename}")
        
        # Load model scores
        scores_file = os.path.join(self.model_dir, f"{prefix}_scores.json")
        if os.path.exists(scores_file):
            import json
            with open(scores_file, 'r') as f:
                self.model_scores = json.load(f)
            print(f"  Loaded model scores from {scores_file}")
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all model performances.
        
        Returns:
            DataFrame with model performance summary
        """
        if not self.model_scores:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, scores in self.model_scores.items():
            summary_data.append({
                'Model': model_name,
                'Test R²': scores['test_r2'],
                'Test RMSE': scores['test_rmse'],
                'Test MAE': scores['test_mae'],
                'CV R² Mean': scores['cv_mean'],
                'CV R² Std': scores['cv_std']
            })
        
        return pd.DataFrame(summary_data).round(4)


def train_crop_yield_models(data_path: str = "Data/crop_yield_data.csv") -> CropYieldPredictor:
    """
    Train crop yield prediction models using the provided data.
    
    Args:
        data_path: Path to the crop yield data CSV file
        
    Returns:
        Trained CropYieldPredictor instance
    """
    # Import the preprocessing function
    import sys
    sys.path.append('.')
    from data_preprocessing import preprocess_crop_data
    
    print("Starting crop yield model training...")
    
    # Preprocess the data
    print("\n1. Preprocessing data...")
    preprocessed_data = preprocess_crop_data(data_path)
    
    # Extract training data
    X_train = preprocessed_data['X_train']
    X_test = preprocessed_data['X_test']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']
    
    print(f"\n2. Data shapes:")
    print(f"   Training: {X_train.shape}")
    print(f"   Testing: {X_test.shape}")
    
    # Initialize and train models
    print("\n3. Training models...")
    predictor = CropYieldPredictor()
    
    # Train all models
    scores = predictor.train_models(X_train, y_train, X_test, y_test)
    
    # Perform hyperparameter tuning on best model
    print(f"\n4. Hyperparameter tuning for best model...")
    if predictor.best_model is not None:
        tuning_results = predictor.hyperparameter_tuning(X_train, y_train, predictor.best_model)
    else:
        print("No best model available for tuning")
    
    # Create visualizations
    print(f"\n5. Creating visualizations...")
    predictor.plot_model_performance("Data/model_performance.png")
    
    if predictor.best_model in ['random_forest', 'gradient_boosting']:
        predictor.plot_feature_importance(predictor.best_model, save_path="Data/feature_importance.png")
    
    # Save models
    print(f"\n6. Saving models...")
    predictor.save_models()
    
    # Print final summary
    print(f"\nFinal Model Summary:")
    summary = predictor.get_model_summary()
    print(summary.to_string(index=False))
    
    print(f"\nModel training completed successfully!")
    if predictor.best_model is not None and predictor.best_model in scores:
        print(f"Best model: {predictor.best_model}")
        if isinstance(scores, dict) and predictor.best_model in scores:
            print(f"Test R² Score: {scores[predictor.best_model]['test_r2']:.4f}")
    else:
        print("No best model determined")
    
    return predictor


if __name__ == "__main__":
    # Train the models
    predictor = train_crop_yield_models()
    
    # Example prediction
    print(f"\nExample prediction using {predictor.best_model}:")
    
    # Create sample input with all required features
    sample_features = pd.DataFrame()
    
    # Add all required features with default values
    if predictor.best_model is not None and hasattr(predictor.models[predictor.best_model], 'feature_names_in_'):
        for feature in predictor.models[predictor.best_model].feature_names_in_:
            if 'Year_Numeric' in feature:
                sample_features[feature] = [2023]
            elif 'Country_Encoded' in feature:
                sample_features[feature] = [0]  # Ghana
            elif 'Decade' in feature:
                sample_features[feature] = [2020]
            else:
                sample_features[feature] = [0.0]  # Default value for other features
    
    if not sample_features.empty and predictor.best_model is not None:
        prediction = predictor.predict(sample_features, predictor.best_model)
        print(f"   Predicted crop yield: {prediction[0]:.2f} tons/hectare")
    else:
        print("   Unable to create sample prediction - no features available")
