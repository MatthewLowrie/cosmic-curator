
from typing import Dict, Any, List, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import PipelineConfig, logger


def make_forest(config: PipelineConfig) -> Pipeline:
    """Returns a fresh Scikit-learn Pipeline with an Imputer and Random Forest."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=config.random_state,
            n_jobs=-1,
        )),
    ])


def make_logistic() -> Pipeline:
    """Returns a fresh Scikit-learn Pipeline with standard scaling and Logistic Regression."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def prepare_labeled_splits(data: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, pd.Series, List[int], List[int]]:
    """Prepares targets and splits training/test sets while maintaining target ratios."""
    labeled = data[data["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    labeled["target"] = (labeled["koi_disposition"] == "CONFIRMED").astype(int)
    
    train_idx, test_idx = train_test_split(
        labeled.index,
        test_size=0.25,
        random_state=config.random_state,
        stratify=labeled["target"]
    )
    return labeled, labeled["target"], train_idx, test_idx


def run_model_training(
    labeled: pd.DataFrame,
    y: pd.Series,
    train_idx: List[int],
    test_idx: List[int],
    config: PipelineConfig
) -> Dict[str, Any]:
    """Handles training, auditing, and calculating metrics for all pipeline models."""
    logger.info("Starting model training and evaluation phase...")
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    # 1. Leaky Forest (Baseline with target leakage)
    leaky_features = config.physical_features + config.leakage_features
    leaky_forest = make_forest(config)
    leaky_forest.fit(labeled.loc[train_idx, leaky_features], y_train)
    leaky_probs = leaky_forest.predict_proba(labeled.loc[test_idx, leaky_features])[:, 1]
    leaky_auc = roc_auc_score(y_test, leaky_probs)
    
    # 2. Audited Forest (No leakage)
    audited_forest = make_forest(config)
    audited_forest.fit(labeled.loc[train_idx, config.physical_features], y_train)
    forest_probs = audited_forest.predict_proba(labeled.loc[test_idx, config.physical_features])[:, 1]
    forest_preds = audited_forest.predict(labeled.loc[test_idx, config.physical_features])
    forest_auc = roc_auc_score(y_test, forest_probs)
    confusion = confusion_matrix(y_test, forest_preds, labels=[0, 1])
    
    # 3. Logistic Baseline (For simple comparison)
    logistic = make_logistic()
    logistic.fit(labeled.loc[train_idx, config.physical_features], y_train)
    logistic_probs = logistic.predict_proba(labeled.loc[test_idx, config.physical_features])[:, 1]
    logistic_auc = roc_auc_score(y_test, logistic_probs)
    
    # 4. Final Production Fit on all Physical measurements
    final_forest = make_forest(config)
    final_forest.fit(labeled[config.physical_features], y)
    
    # Organize feature importances
    importance_df = pd.DataFrame({
        "feature": config.physical_features,
        "importance": final_forest.named_steps["model"].feature_importances_
    })
    importance_df["feature"] = importance_df["feature"].map(config.feature_labels)
    importance_df = importance_df.sort_values("importance", ascending=True)

    logger.info(f"Audited metrics computed. Forest Test AUC: {forest_auc:.3f}")
    return {
        "leaky_auc": leaky_auc,
        "forest_auc": forest_auc,
        "logistic_auc": logistic_auc,
        "confusion_matrix": confusion,
        "importances": importance_df,
        "production_model": final_forest
    }
