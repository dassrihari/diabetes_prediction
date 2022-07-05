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

def evaluate(config_path):
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    test_data_path = config["split_data"]["test_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    solver = config["base"]["solver"]
    
    target = config["base"]["target_col"]
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    test = pd.read_csv(test_data_path, sep=",", names=col_names)
    test_size = test.shape[0]
    print("Test Size", test_size)
    test_y = test[target]
    print(test[target])
    test_x = test.drop(target, axis=1)

    model_path = os.path.join(model_dir, "final_model.joblib")
    model = joblib.load(model_path)    
    # Report test set score
    test_score = model.score(test_x, test_y) * 100
    print(test_score)

    predicted_val = model.predict(test_x)

           
    precision, recall, prc_thresholds = metrics.precision_recall_curve(test_y, predicted_val)
    fpr, tpr, roc_thresholds = metrics.roc_curve(test_y, predicted_val)

    avg_prec = metrics.average_precision_score(test_y, predicted_val)
    roc_auc = metrics.roc_auc_score(test_y, predicted_val)
    
    scores_file = config["reports"]["test_scores"]
    prc_file = config["reports"]["prc"]
    roc_file = config["reports"]["roc"]
    auc_file = config["reports"]["auc"]
    

        
    nth_point = math.ceil(len(prc_thresholds)/1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]    
    
    
    with open(prc_file, "w") as fd:
        prcs = {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in prc_points
                ]
            }
        json.dump(prcs, fd, indent=4, cls=NumpyEncoder)
        

    with open(roc_file, "w") as fd:
        rocs = {
                "roc": [
                    {"fpr": fp, "tpr": tp, "threshold": t}
                    for fp, tp, t in zip(fpr, tpr, roc_thresholds)
                ]
            }
        json.dump(rocs, fd, indent=4, cls=NumpyEncoder)
        

    

    # Print classification report
    print(classification_report(test_y, predicted_val))

    # Confusion Matrix and plot
    cm = confusion_matrix(test_y, predicted_val)
    print(cm)

        
    df1 = pd.DataFrame(predicted_val, columns = ['Predicted'])
    df_cm = pd.concat([test_y, df1], axis=1)
    print(df_cm)
    
          
    cm_file = config["reports"]["cm"]
    df_cm.to_csv(cm_file, index = False)



    roc_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    #dvclive.log("roc_auc", roc_auc)
    print('ROC_AUC:{0:0.2f}'.format(roc_auc))

    Logistic_Accuracy = accuracy_score(test_y, predicted_val)
    #dvclive.log("Accuracy",Logistic_Accuracy)
    print('Logistic Regression Model Accuracy:{0:0.2f}'.format(Logistic_Accuracy))

    # Average precision score
    average_precision = average_precision_score(test_y, predicted_val)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    
    with open(scores_file, "w") as f:
        scores = {
            "Test Score": test_score,
            "ROC_AUC": roc_auc,
            "Test Size": test_size,
            "Solver": solver,
            "Accuracy": Logistic_Accuracy
            
            
       }
        json.dump(scores, f, indent=4)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    evaluate(config_path = parsed_args.config)
