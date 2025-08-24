import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import CLASSIFIER_CONFIG, TRAINING_CONFIG, EVALUATION_CONFIG, RANDOM_SEED

logger = logging.getLogger(__name__)

class SentimentClassifier:

    def __init__(self, config: dict = CLASSIFIER_CONFIG):
        self.config = config
        self.classifier = None
        self.feature_selector = None
        self.is_trained = False
        self.training_history = {}
        self.X_test = None
        self.y_test = None
        
        logger.info("SentimentClassifier initialized")
    
    def _get_classifier(self) -> object:
        model_type = self.config.get('model_type', 'logistic_regression').lower()
        params = self.config.get('params', {})
        
        if model_type == 'logistic_regression':
            return LogisticRegression(**params)
        elif model_type == 'svm':
            return LinearSVC(**params)
        elif model_type == 'random_forest':
            return RandomForestClassifier(**params)
        else:
            logger.warning(f"Unknown model type: {model_type}, using LogisticRegression")
            return LogisticRegression(**CLASSIFIER_CONFIG['params'])
    
    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        feature_config = self.config.get('feature_selection', {})
        
        if not feature_config or not feature_config.get('k'):
            return X
        
        k = min(feature_config['k'], X.shape[1])
        
        logger.info(f"Selecting top {k} features from {X.shape[1]} features")
        
        try:
            if feature_config.get('score_func') == 'f_classif':
                score_func = f_classif
            else:
                score_func = f_classif
            
            self.feature_selector = SelectKBest(score_func=score_func, k=k)
            
            X_selected = self.feature_selector.fit_transform(X, y)
            
            logger.info(f"Feature selection completed: {X_selected.shape[1]} features selected")
            
            return X_selected
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return X
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Optional[Dict]:
        logger.info(f"Training classifier on {X.shape[0]} samples with {X.shape[1]} features")
        
        try:
            X_selected = self._select_features(X, y)
            
            train_config = TRAINING_CONFIG
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y,
                test_size=train_config['test_size'],
                random_state=train_config['random_state'],
                stratify=y if train_config['stratify'] else None
            )
            
            self.X_test = X_test
            self.y_test = y_test
            
            logger.info(f"Train samples: {len(X_train)}")
            logger.info(f"Test samples: {len(X_test)}")
            
            self.classifier = self._get_classifier()
            
            logger.info("Training classifier...")
            self.classifier.fit(X_train, y_train)
            
            train_pred = self.classifier.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred)
            
            cv_folds = train_config.get('cv_folds', 3)
            cv_scores = cross_val_score(
                self.classifier, X_train, y_train, 
                cv=cv_folds, scoring='accuracy', n_jobs=-1
            )
            val_accuracy = cv_scores.mean()
            val_std = cv_scores.std()
            
            test_pred = self.classifier.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            self.training_history = {
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'val_std': val_std,
                'test_accuracy': test_accuracy,
                'cv_scores': cv_scores,
                'feature_shape': X_selected.shape,
                'train_shape': X_train.shape,
                'test_shape': X_test.shape
            }
            
            self.is_trained = True
            
            logger.info("Training completed successfully!")
            logger.info(f"Training accuracy: {train_accuracy:.3f}")
            logger.info(f"Validation accuracy: {val_accuracy:.3f} (±{val_std * 2:.3f})")
            logger.info(f"Test accuracy: {test_accuracy:.3f}")
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def predict(self, X: np.ndarray) -> Optional[Dict]:
        if not self.is_trained:
            logger.error("Classifier not trained")
            return None
        
        try:
            if self.feature_selector is not None:
                X_selected = self.feature_selector.transform(X)
            else:
                X_selected = X
            
            predictions = self.classifier.predict(X_selected)
            probabilities = self.classifier.predict_proba(X_selected)
            
            confidence = np.max(probabilities, axis=1)
            
            return {
                'prediction': predictions,
                'probabilities': probabilities,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def evaluate(self) -> Optional[Dict]:
        if not self.is_trained or self.X_test is None or self.y_test is None:
            logger.error("Model not trained or test data not available")
            return None
        
        logger.info("Evaluating model performance...")
        
        try:
            y_pred = self.classifier.predict(self.X_test)
            y_pred_proba = self.classifier.predict_proba(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            except:
                auc = None
            
            target_names = EVALUATION_CONFIG.get('target_names', ['Negative', 'Positive'])
            class_report = classification_report(
                self.y_test, y_pred, 
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
            
            cm = confusion_matrix(self.y_test, y_pred)
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'classification_report': class_report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info("Evaluation Results:")
            logger.info(f"   Accuracy: {accuracy:.3f}")
            logger.info(f"   Precision: {precision:.3f}")
            logger.info(f"   Recall: {recall:.3f}")
            logger.info(f"   F1-Score: {f1:.3f}")
            if auc:
                logger.info(f"   AUC: {auc:.3f}")
            
            if EVALUATION_CONFIG.get('plot_confusion_matrix', False):
                self._plot_confusion_matrix(cm, target_names)
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def _plot_confusion_matrix(self, cm: np.ndarray, target_names: List[str]) -> None:
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names,
                cbar_kws={'label': 'Count'}
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
            
            logger.info("Confusion matrix plotted")
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")
    
    def get_feature_importance(self, top_k: int = 20) -> Optional[Dict]:
        if not self.is_trained:
            logger.error("Classifier not trained")
            return None
        
        try:
            importance_info = {}
            
            if hasattr(self.classifier, 'coef_'):
                coefficients = self.classifier.coef_[0]
                
                if self.feature_selector is not None:
                    selected_features = self.feature_selector.get_support(indices=True)
                    feature_scores = self.feature_selector.scores_[selected_features]
                    
                    importance_scores = np.abs(coefficients) * feature_scores
                else:
                    importance_scores = np.abs(coefficients)
                
                top_indices = np.argsort(importance_scores)[-top_k:][::-1]
                
                importance_info = {
                    'feature_indices': top_indices,
                    'importance_scores': importance_scores[top_indices],
                    'coefficients': coefficients[top_indices] if hasattr(self.classifier, 'coef_') else None
                }
            
            elif hasattr(self.classifier, 'feature_importances_'):
                importances = self.classifier.feature_importances_
                top_indices = np.argsort(importances)[-top_k:][::-1]
                
                importance_info = {
                    'feature_indices': top_indices,
                    'importance_scores': importances[top_indices]
                }
            
            return importance_info
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return None
    
    def save_model(self, filepath: str) -> bool:
        if not self.is_trained:
            logger.error("No trained model to save")
            return False
        
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'classifier': self.classifier,
                'feature_selector': self.feature_selector,
                'config': self.config,
                'training_history': self.training_history,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            
            logger.info(f"Model saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        try:
            if not Path(filepath).exists():
                logger.error(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.classifier = model_data['classifier']
            self.feature_selector = model_data.get('feature_selector')
            self.config = model_data.get('config', self.config)
            self.training_history = model_data.get('training_history', {})
            self.is_trained = model_data.get('is_trained', True)
            
            logger.info(f"Model loaded from: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        info = {
            'is_trained': self.is_trained,
            'model_type': self.config.get('model_type', 'unknown'),
            'has_feature_selector': self.feature_selector is not None,
            'training_history': self.training_history
        }
        
        if self.is_trained and self.classifier:
            info['classifier_type'] = type(self.classifier).__name__
            
            if hasattr(self.classifier, 'coef_'):
                info['n_features'] = self.classifier.coef_.shape[1]
            if hasattr(self.classifier, 'classes_'):
                info['classes'] = self.classifier.classes_.tolist()
        
        return info

if __name__ == "__main__":
    # Create dummy data for testing
    np.random.seed(RANDOM_SEED)
    
    # Generate synthetic embeddings (768-dim like sentence transformers)
    n_samples = 1000
    n_features = 768
    X = np.random.randn(n_samples, n_features)
    
    # Generate binary labels
    y = np.random.randint(0, 2, n_samples)
    
    print("Testing sentiment classifier:")
    print("-" * 50)
    
    # Initialize classifier
    classifier = SentimentClassifier()
    
    # Train classifier
    results = classifier.train(X, y)
    
    if results:
        print("Training completed successfully")
        print(f"Training accuracy: {results['train_accuracy']:.3f}")
        print(f"Validation accuracy: {results['val_accuracy']:.3f}")
        print(f"Test accuracy: {results['test_accuracy']:.3f}")
        
        # Evaluate model
        eval_results = classifier.evaluate()
        if eval_results:
            print("Evaluation completed")
            print(f"Final accuracy: {eval_results['accuracy']:.3f}")
            print(f"F1-Score: {eval_results['f1_score']:.3f}")
        
        # Test prediction
        sample_X = X[:5]  # First 5 samples
        pred_results = classifier.predict(sample_X)
        if pred_results:
            print("Predictions made")
            print(f"Predictions: {pred_results['prediction']}")
            print(f"Confidence: {pred_results['confidence']}")
    else:
        print("Training failed")