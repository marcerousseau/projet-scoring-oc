import gc
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from typing import Any, Callable, Dict, Optional
import itertools

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score, BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import KNNImputer

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline

from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score, confusion_matrix

import mlflow
from mlflow import log_metric, log_params

import matplotlib.pyplot as plt
import seaborn as sns

def run_classifier(model_str = 'logistic_regression', scorer = 'roc_auc', custom_scorer = None, sample_size = 1, experiment_name = "default-experiment"):
    """Run a classifier with the given model and scorer

    Args:
        model_str (str, optional): The model to use. Defaults to 'logistic_regression'. Options are 'random_forest', 'logistic_regression', 'ridge_classifier', 'xgb_classifier', 'decision_tree'
        scorer (str, optional): The scorer to use. Defaults to 'roc_auc'. Options are 'roc_auc', 'f1', 'precision', 'recall', 'custom'
        custom_scorer (function, optional): The custom scorer to use. Defaults to None. Required if scorer is 'custom'
        sample_size (float, optional): The sample size to use. Defaults to 1. If < 1, then a random sample of the data is used
    
    """
    # Load data from ./data/df.csv
    df = pd.read_csv("data/df.csv")

    df = df.replace([np.inf, -np.inf], np.nan)

    # imputer = KNNImputer(n_neighbors=5)
    # df_imputed = imputer.fit_transform(df)
    df.fillna(df.median(), inplace=True)

    X = df.drop(columns=["TARGET"])
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    y = df["TARGET"]

    if sample_size < 1:
        X = X.sample(frac=sample_size, random_state=42)
        y = y.sample(frac=sample_size, random_state=42)
    
    # Model selection
    if model_str == 'decision_tree':
        model = DecisionTreeClassifier()
        param_grid = {
            'model__max_depth': [None, 5, 10, 15], # Best is 5 over [None, 5, 10, 15]
        }
    elif model_str == 'logistic_regression':
        model = LogisticRegression(solver='liblinear')
        param_grid = {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100], # Best is 0.001 over [0.001, 0.01, 0.1, 1, 10, 100]
        }
    elif model_str == 'ridge_classifier':
        model = RidgeClassifier()
        param_grid = {
            'model__alpha': [0.1, 1, 10, 100],
        }
    elif model_str == 'xgb_classifier':
        model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        param_grid = {
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__n_estimators': [100, 200, 300],
        }
    elif model_str == 'random_forest':
        model = RandomForestClassifier()
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
        }
    else:
        raise ValueError(f"Unknown model string: {model_str}")

    param_grid = {f"over__k_neighbors": [1, 2, 3, 4, 5, 6, 7], **param_grid}
    
    print(f"Original class distribution: {Counter(y)}")
    # SMOTE and RandomUnderSampler
    over = SMOTE(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.5)

    # Apply oversampling
    X_over, y_over = over.fit_resample(X, y)
    print(f"Class distribution after oversampling: {Counter(y_over)}")

    # Apply undersampling
    X_under, y_under = under.fit_resample(X_over, y_over)
    print(f"Class distribution after undersampling: {Counter(y_under)}")
    
    # Pipeline
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 

    """Using GridSearchCV but not compatible with mlflow
    # GridSearchCV
    grid_search = GridSearchCV(pipeline, pipeline_params, scoring=scorer, cv=cv, n_jobs=-1)
    grid_search.fit(X, y)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best {scorer} score: {grid_search.best_score_:.3f}")
    """

    # Start mlflow experiment
    mlflow.set_experiment(experiment_name)

    # Create a parameter grid object
    param_grid_obj = ParameterGrid(param_grid)

    # Iterate over the parameter grid
    all_scores = []
    for params in param_grid_obj:
        pipeline.set_params(**params)
        if scorer == 'custom':
            scores = cross_val_score(pipeline, X, y, scoring=custom_scorer, cv=cv, n_jobs=-1)
        else:
            scores = cross_val_score(pipeline, X, y, scoring=scorer, cv=cv, n_jobs=-1)
        for secondary_score in ['roc_auc', 'f1', 'precision', 'recall']:
            if secondary_score != scorer:
                secondary_scores = cross_val_score(pipeline, X, y, scoring=secondary_score, cv=cv, n_jobs=-1)
                log_metric(secondary_score, np.mean(secondary_scores))
        mean_score = np.mean(scores)

        with mlflow.start_run(run_name=experiment_name, nested=True):
            # Log the model name as a tag
            mlflow.set_tag("mlflow.runName", model_str)
            log_params(params)
            log_params({'models': model_str, 'sample_size': sample_size, 'scorer': scorer})
            log_metric(scorer, mean_score)
            all_scores.append((params, mean_score))

    # Find the best parameters and score
    best_params, best_score = max(all_scores, key=lambda x: x[1])

    # Print results
    print(f"Best parameters: {best_params}")
    print(f"Best {scorer} score: {best_score:.3f}")

    # Train the best model with the best parameters
    best_pipeline = Pipeline(steps=steps)
    best_pipeline.set_params(**best_params)
    best_pipeline.fit(X, y)

    # Save the best model as a pickled file
    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    # Extract the feature importances
    if model_str in ["decision_tree", "random_forest", "xgb_classifier"]:
        importances = best_pipeline.named_steps["model"].feature_importances_
    elif model_str in ["logistic_regression", "ridge_classifier"]:
        importances = np.abs(best_pipeline.named_steps["model"].coef_[0])
    else:
        raise ValueError(f"Unknown model string: {model_str}")
    
    # Visualize the feature importances
    feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    feature_importances = feature_importances.sort_values("Importance", ascending=False)

    print("\nFeature Importances:")
    print(feature_importances)

    # Plot the feature importances
    plt.figure(figsize=(10, 5))
    sns.barplot(data=feature_importances.loc[:12], x="Importance", y="Feature", orient="h", color="b")
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    # Save the plot as a PNG file
    plt.savefig(f"feature_importances_best_{model_str}.png", bbox_inches="tight", dpi=300)

    # Log the pickled model as an artifact in mlflow
    with mlflow.start_run(run_name=experiment_name, nested=True):
        log_params(best_params)
        log_metric(scorer, best_score)
        mlflow.set_tag("mlflow.runName", model_str + '_best_model')
        mlflow.log_artifact("best_model.pkl")
        mlflow.log_artifact(f"feature_importances_best_{model_str}.png")
    return

def custom_loss(y_true, y_pred, false_positive_cost, false_negative_cost):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return (fp * false_positive_cost) + (fn * false_negative_cost)

def custom_score_func(y_true, y_pred):
    false_positive_cost = 1  # cost of missing a customer credit
    false_negative_cost = 10  # cost of a customer default
    return -custom_loss(y_true, y_pred, false_positive_cost, false_negative_cost)

if __name__ == "__main__":
    custom_scorer = make_scorer(custom_score_func, greater_is_better=True)
    # for model_str in ['logistic_regression', 'decision_tree', 'random_forest', 'xgb_classifier', 'ridge_classifier']:
    for model_str in ['decision_tree', 'random_forest', 'xgb_classifier', 'ridge_classifier']:
        run_classifier(model_str=model_str, sample_size=0.1, custom_scorer=custom_scorer, scorer='roc_auc', experiment_name=model_str + '-experiment')