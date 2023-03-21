import gc
import pandas as pd
import numpy as np
import pickle
from collections import Counter

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import KNNImputer

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline as make_imb_pipeline

from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score, confusion_matrix

import mlflow

mlflow.set_experiment("project-experiment")

def resample_data():
    # Load data from ./data/df.csv
    df = pd.read_csv("data/df.csv")

    df = df.replace([np.inf, -np.inf], np.nan)

    # imputer = KNNImputer(n_neighbors=5)
    # df_imputed = imputer.fit_transform(df)
    df.fillna(df.median(), inplace=True)

    # n_missing = df.isna().sum(axis=0) / df.shape[0]
    # missings = n_missing.sort_values(ascending=False)
    # print(missings)

    X = df.drop(columns=["TARGET"])
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Original training data shape: {X_train.shape} - {y_train.shape}; Resampled training data shape: {X_train_resampled.shape} - {y_train_resampled.shape}")
    # Create a sub-sample of the data to speed up the model exploration
    X_train_resampled = X_train_resampled.sample(10000, random_state=42)
    y_train_resampled = y_train_resampled.sample(10000, random_state=42)
    X_test = X_test.sample(2000, random_state=42)
    y_test = y_test.sample(2000, random_state=42)
    del X_train, y_train, smote, df, X, y
    gc.collect()
    print(f"Sub-sampled training data shape: {X_train_resampled.shape} - {y_train_resampled.shape}; Sub-sampled test data shape: {X_test.shape} - {y_test.shape}")
    pickle.dump(X_train_resampled, open("data/X_train_resampled.pkl", "wb"))
    pickle.dump(y_train_resampled, open("data/y_train_resampled.pkl", "wb"))
    pickle.dump(X_test, open("data/X_test.pkl", "wb"))
    pickle.dump(y_test, open("data/y_test.pkl", "wb"))
    return

def resample_data_over_under_method():
    # Load data from ./data/df.csv
    df = pd.read_csv("data/df.csv")

    df = df.replace([np.inf, -np.inf], np.nan)

    # imputer = KNNImputer(n_neighbors=5)
    # df_imputed = imputer.fit_transform(df)
    df.fillna(df.median(), inplace=True)

    # n_missing = df.isna().sum(axis=0) / df.shape[0]
    # missings = n_missing.sort_values(ascending=False)
    # print(missings)

    X = df.drop(columns=["TARGET"])
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    y = df["TARGET"]

    counter = Counter(y)
    print(counter)
    # transform the dataset
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    # summarize the new class distribution
    counter = Counter(y)
    print(counter)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)
    return

def run_logistic_regression():
    X_train_resampled = pickle.load(open("data/X_train_resampled.pkl", "rb"))
    y_train_resampled = pickle.load(open("data/y_train_resampled.pkl", "rb"))
    X_test = pickle.load(open("data/X_test.pkl", "rb"))
    y_test = pickle.load(open("data/y_test.pkl", "rb"))
    # Train a logistic regression classifier on the training set
    clf = LogisticRegression()
    clf.fit(X_train_resampled, y_train_resampled)

    # Use the classifier to predict the target variable for the test set
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Convert the predicted probabilities to class labels using a threshold of 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate the AUC score for the test set predictions
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"Logistic Classification AUC score: {auc_score}")
    mlflow.log_metric("auc_score", auc_score)

    beta = 3
    f_beta = fbeta_score(y_test, y_pred, beta=beta)
    print(f"F-{beta} score: {f_beta:.3f}")
    mlflow.log_metric(f"f_{beta}_score", f_beta)
    try:
        print(clf.feature_importances_)
    except:
        pass
    return

def run_classifier(model_str = 'random_forest', scorer = 'roc_auc', custom_scorer = None, sample_size = 1):
    """Run a classifier with the given model and scorer

    Args:
        model_str (str, optional): The model to use. Defaults to 'random_forest'. Options are 'random_forest', 'xgboost', 'lightgbm', 'catboost'
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
    
    if model_str == 'decision_tree':
        model = DecisionTreeClassifier()
        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'over__k_neighbors': [1, 2, 3, 4, 5, 6, 7],
            'model__max_depth': [None, 5, 10, 15],
        }
    
    # SMOTE and RandomUnderSampler
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)

    # Apply oversampling
    X_over, y_over = over.fit_resample(X, y)
    print(f"Class distribution after oversampling: {Counter(y_over)}")

    # Apply undersampling
    X_under, y_under = under.fit_resample(X_over, y_over)
    print(f"Class distribution after undersampling: {Counter(y_under)}")
    
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = make_imb_pipeline(steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
    grid_search.fit(X, y)
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best {scorer} score: {grid_search.best_score_:.3f}")

    return

def run_ridge_classifier():
    X_train_resampled = pickle.load(open("data/X_train_resampled.pkl", "rb"))
    y_train_resampled = pickle.load(open("data/y_train_resampled.pkl", "rb"))
    X_test = pickle.load(open("data/X_test.pkl", "rb"))
    y_test = pickle.load(open("data/y_test.pkl", "rb"))

    clf = RidgeClassifier()
    clf.fit(X_train_resampled, y_train_resampled)

    # Use the classifier to predict the target variable for the test set
    y_pred_proba = clf.predict(X_test)

    # Convert the predicted probabilities to class labels using a threshold of 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate the AUC score for the test set predictions
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"Ridge Classification AUC score: {auc_score}")
    mlflow.log_metric("auc_score", auc_score)

    beta = 3
    f_beta = fbeta_score(y_test, y_pred, beta=beta)
    print(f"F-{beta} score: {f_beta:.3f}")
    mlflow.log_metric(f"f_{beta}_score", f_beta)

    
    return

def run_xgb_classifier():
    X_train_resampled = pickle.load(open("data/X_train_resampled.pkl", "rb"))
    y_train_resampled = pickle.load(open("data/y_train_resampled.pkl", "rb"))
    X_test = pickle.load(open("data/X_test.pkl", "rb"))
    y_test = pickle.load(open("data/y_test.pkl", "rb"))

    xgb_model = XGBClassifier(objective='multi:softmax', num_class=3, n_jobs=-1)

    # Define the hyperparameters to search over
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [50, 100, 200],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Perform the grid search using cross-validation
    grid_search = GridSearchCV(xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_resampled, y_train_resampled)
    # Train the model on the training data
    # xgb_model.fit(X_train_resampled, y_train_resampled)
    # Make predictions on the test data
    # y_pred = xgb_model.predict(X_test)

    # Print the best hyperparameters
    print('Best hyperparameters: ', grid_search.best_params_)

    # Evaluate the model on the test data using the best hyperparameters
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate the accuracy of the model
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"XGBoost AUC score: {auc_score}")
    mlflow.log_metric("auc_score", auc_score)

    beta = 3
    f_beta = fbeta_score(y_test, y_pred, beta=beta)
    print(f"F-{beta} score: {f_beta:.3f}")
    mlflow.log_metric(f"f_{beta}_score", f_beta)
    return

def run_lazy_classifier():
    X_train_resampled = pickle.load(open("data/X_train_resampled.pkl", "rb"))
    y_train_resampled = pickle.load(open("data/y_train_resampled.pkl", "rb"))
    X_test = pickle.load(open("data/X_test.pkl", "rb"))
    y_test = pickle.load(open("data/y_test.pkl", "rb"))
    clf = LazyClassifier(verbose=2,ignore_warnings=True, custom_metric=None)
    models,predictions = clf.fit(X_train_resampled, X_test, y_train_resampled, y_test)
    print(models)

if __name__ == "__main__":
    # resample_data()
    # resample_data_over_under_method()
    run_classifier(model_str='decision_tree', sample_size=0.1)
    # run_logistic_regression()
    # run_ridge_classifier()
    # run_lazy_classifier()
    # run_xgb_classifier()

    # custom_scorer = make_scorer(custom_score_func, greater_is_better=True)
    # run_random_forest_classifier(custom_scorer)