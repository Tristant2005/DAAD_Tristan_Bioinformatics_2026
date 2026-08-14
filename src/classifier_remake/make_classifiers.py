# Mellanie Martin, Tristan Tucker
# July 2025, August 2026

### IMPORTS ###
import pandas as pd
import sklearn
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from itertools import combinations
from collections import Counter


### CLASSES ###
class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, xs, ys, **params):
        return self
    def transform(self, xs):
        return xs[self.columns]


### FUNCTIONS ###
# _eb = Classifier for embedded vs. bumps
def create_classifier_data(training_dataset, testing_dataset): 
    X_train = training_dataset.drop("class", axis=1)
    y_train = training_dataset["class"]
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, stratify = y, test_size = 0.2)

    X_test = testing_dataset.drop("class", axis=1)
    y_test = testing_dataset["class"]

    # smote = SMOTE(random_state = 42)
    # X_upsampled, y_upsampled = smote.fit_resample(X_train, y_train)
    # print(f"After Upsampling: {Counter(y_upsampled)}")

    return X_train, X_test, y_train, y_test


def train_classifier_eb(xs, ys):

    columns = list(xs)

    column_combinations = []
    for r in range(1, len(columns) + 1):
        column_combinations.extend(combinations(columns, r))
    column_combinations = [list(comb) for comb in column_combinations]

    steps = [
        ('column_select', SelectColumns(columns)),
        ("scale", MinMaxScaler()),
        ("classify", rfc(criterion = "gini")),
    ]
    pipe = Pipeline(steps)

    grid = { 
            "column_select__columns": column_combinations,
            "classify__max_depth": [8],
            "classify__min_samples_split": [2],
            "classify__n_estimators": [102],
        }

    search = GridSearchCV(pipe, grid, cv = 5, scoring = "f1_macro", n_jobs = -1)
    search.fit(xs, ys)
    print(f"f1_macro Score: {search.best_score_}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


def run_eb_classifier(train_data = "/home/mellanie/Desktop/Tristan_2026_code/src/classifier_remake/stats_in_csv/embedded_root_hair_bump_stats.csv",
                      test_data =  "/home/mellanie/Desktop/Tristan_2026_code/src/classifier_remake/stats_in_csv/test_embedded_root_hair_bump_stats.csv"):

    training_dataset = pd.read_csv(train_data)
    testing_dataset = pd.read_csv(test_data)
    X_train, X_test, y_train, y_test = create_classifier_data(training_dataset, testing_dataset)
    classifier = train_classifier_eb(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("mcc:", matthews_corrcoef(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    joblib.dump(classifier, "/home/mellanie/Desktop/container/model/classifiers/new_bumps_embedded_classifier.pkl")



# _bg = Classifier for background vs regular hairs
# def create_classifier_bg(dataset):
#     X = dataset.drop("class", axis=1)
#     y = dataset["class"]
#     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, stratify = y, test_size = 0.2)

#     smote = SMOTE(random_state = 7)
#     X_upsampled, y_upsampled = smote.fit_resample(X_train, y_train)
#     print(f"After Upsampling: {Counter(y_upsampled)}")

#     return X_upsampled, X_test, y_upsampled, y_test


def train_classifier_bg(xs, ys):

    columns = list(xs)
    print(columns)

    column_combinations = []
    for r in range(1, len(columns) + 1):
        column_combinations.extend(combinations(columns, r))
    column_combinations = [list(comb) for comb in column_combinations]

    steps = [
        ('column_select', SelectColumns(columns)),
        ("scale", MinMaxScaler()),
        ("classify", rfc(criterion = "gini")),
    ]
    pipe = Pipeline(steps)

    grid = { 
            "column_select__columns": column_combinations,
            "classify__max_depth": range(7, 9),
            "classify__min_samples_split": range(2, 4),
            "classify__n_estimators": range(100, 110),
        }

    search = GridSearchCV(pipe, grid, cv = 5, scoring = "f1_macro", n_jobs = -1)
    search.fit(xs, ys)
    print(f"f1_macro Score: {search.best_score_}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_



def run_bg_classifier(train_data = "/home/mellanie/Desktop/Tristan_2026_code/src/classifier_remake/stats_in_csv/root_hair_background_root_hair_stats.csv",
                      test_data = "/home/mellanie/Desktop/Tristan_2026_code/src/classifier_remake/stats_in_csv/test_root_hair_background_root_hair_stats.csv"):


    training_dataset = pd.read_csv(train_data)
    testing_dataset = pd.read_csv(test_data)
    X_train, X_test, y_train, y_test = create_classifier_data(training_dataset, testing_dataset)
    classifier = train_classifier_bg(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("mcc:", matthews_corrcoef(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    joblib.dump(classifier, "/home/mellanie/Desktop/annotations/helper_functions/new_bg_regular_classifier.pkl")


if __name__ == "__main__":
    # run_eb_classifier()
    run_bg_classifier()
    
