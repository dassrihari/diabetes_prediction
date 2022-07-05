import os
import math
import warnings
import sys
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, plot_roc_curve,accuracy_score
from sklearn.metrics import average_precision_score
from numpyencoder import NumpyEncoder
import yaml
import matplotlib.pyplot as plt
import argparse
import joblib
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

def train(config_path):
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    solver = config["base"]["solver"]
    
    target = config["base"]["target_col"]
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    train = pd.read_csv(train_data_path, sep=",", header=None, names=col_names)
    train_size = train.shape[0]
    print("Train Size", train_size)
    print(train.head())
    train_y = train[target]
    train_x = train.drop(target, axis=1)
    
    # Build logistic regression model
    model = LogisticRegression(solver=solver, random_state=random_state).fit(train_x, train_y)
    
    # Report training set score
    train_score = model.score(train_x, train_y) * 100
    #dvclive.log("Train Score", train_score)
    print(train_score)
    
    scores_file = config["reports"]["train_scores"]
    
    
    with open(scores_file, "w") as f:
        scores = {
            "Train Score": train_score,
            "Train Size": train_size,
            "Solver": solver,
       }
        json.dump(scores, f, indent=4)


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "final_model.joblib")
    joblib.dump(model, model_path)
    model_path = os.path.join(model_dir, "final_model_dvc.joblib")
    joblib.dump(model, model_path)
    print("Pima India dataset with solver as saga")


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train(config_path = parsed_args.config)
