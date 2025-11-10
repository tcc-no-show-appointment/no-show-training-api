import time
import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier

try:
    from category_encoders import TargetEncoder
except ImportError:
    raise RuntimeError("Missing 'category_encoders'. Install with: pip install category_encoders")

from imblearn.pipeline import Pipeline as ImbPipeline

from app.config import config
from app.constants import (
    NUMERICAL_FEATURES,
    CATEGORICAL_LOW_CARDINALITY,
    CATEGORICAL_HIGH_CARDINALITY,
    BINARY_FEATURES
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    def __init__(
        self,
        test_size: float = None,
        cv_splits: int = None,
        random_state: int = None
    ):
        self.test_size = test_size or config.TEST_SIZE
        self.cv_splits = cv_splits or config.CV_SPLITS
        self.random_state = random_state or config.MODEL_RANDOM_STATE
        
        self.preprocessor = None
        self.best_model = None
        self.training_metrics = {}
    
    def _create_preprocessor(self) -> ColumnTransformer:
        try:
            ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERICAL_FEATURES),
                ("gender_waitbucket", ohe, CATEGORICAL_LOW_CARDINALITY),
                ("neigh", TargetEncoder(), CATEGORICAL_HIGH_CARDINALITY),
                ("bin", "passthrough", BINARY_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )
        
        logger.info("Preprocessor pipeline created")
        return preprocessor
    
    def _to_float32(self, X):
        return X.astype(np.float32)
    
    def train(
        self,
        X: Any,
        y: Any,
        n_estimators_range: Optional[Tuple[int, int]] = None,
        max_depth_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        min_samples_leaf_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Starting model training. Data shape: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        logger.info(f"Train target distribution: {Counter(y_train)}")
        
        # Create preprocessor
        self.preprocessor = self._create_preprocessor()
        
        # Create pipeline with memory optimization
        astype32 = FunctionTransformer(self._to_float32)
        
        pipeline = ImbPipeline(steps=[
            ("preprocess", self.preprocessor),
            ("astype32", astype32),
            ("clf", RandomForestClassifier(
                n_estimators=120,
                max_depth=None,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=self.random_state
            ))
        ])
        
        # Define grid search parameters
        param_grid = {
            "clf__n_estimators": n_estimators_range or [100, 180],
            "clf__max_depth": max_depth_range or [None, 12],
            "clf__min_samples_leaf": min_samples_leaf_range or [10, 20],
        }
        
        logger.info(f"Grid search parameters: {param_grid}")
        
        # Cross-validation strategy
        cv = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            verbose=0
        )
        
        logger.info("Starting GridSearchCV...")
        grid_search.fit(X_train, y_train)
        
        # Store best model
        self.best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        
        pr_auc = average_precision_score(y_test, y_prob)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Store metrics
        self.training_metrics = {
            "best_params": grid_search.best_params_,
            "pr_auc": pr_auc,
            "best_cv_score": grid_search.best_score_,
            "training_time_seconds": training_time,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_positive_rate": float(y_train.sum() / len(y_train)),
            "test_positive_rate": float(y_test.sum() / len(y_test))
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"PR-AUC on test set: {pr_auc:.4f}")
        
        return {
            "model": self.best_model,
            "metrics": self.training_metrics,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob
        }
