diabetes_prediction
==============================

Simple Classification using Pima Indians Diabetes Database

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── params.yaml        <- The configurations file.
    ├── dvc.yaml           <- The dvc configuration file.
    ├── requirements.txt   <- package installation requirements.
    ├── .github 
    │   ├── workflows
        │   ├── cml.yaml   <- cml file for git actions.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling - created locally on executing dvc repro make_data to keep test and train csv after split
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained models
    │   ├── final_model.joblib      <- The final model which can be used for deployment
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, or Json files etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    ├── dvc_plots          <- Generate by dvc plots show.
    │   ├── index.html     <- generated plot by dvc.
    
    
# Diabetes_prediction

The predicts the Whether the person is diabetic or not using PIMA INDIA DIABETES DATASET. 
ML Model used in Logistic Regression. Metrics are accuracy and Loss.
The repository is configured with DVC and CML. 
The github actions will be carried out upon code push.


Following is the file structure

## src -- Source code direcory
      data/make_dataset.py   -- Preparing the PIMA INDIA DIABETES DATASET. Splitting into Train and Test
      models/train_model.py          -- Performing the training with baseline Linear Regression model
      models/evaluate.py       -- Performing validation using the test split (20%)

## report -- Generated Reports
      features/roc_auc.png    -- ROC AUC Plot
      train_scores.json       -- Report containing the Train Size, Train Score and Solver
      test_scores.json        -- Report containing the Test Size, Test Score, ROC AUC Score and Solver
      cm.csv                  -- Predicted Labels and Test Labels
      prc.json                -- Precision values
      roc.json                -- roc values
      prediction.png    -- Scatter plot showing the Prediction vs. Ground Truth

![ROC_AUC](https://user-images.githubusercontent.com/103778538/177507393-b4c23d1c-e339-44d0-baf1-d0c97cd550ce.png
   
## models -- 
      final_model.joblib  -- Model saved using the joblib utility

## Makefile --
      make requirements -- To install dependencies

## DVC Stages
![DVC Stages](https://user-images.githubusercontent.com/103778538/177500733-fb8b1f99-3f76-4a35-b45f-3e93069eb8d5.png)

DVC has 3 stages
- make_data : preparing the data, performing train test split
- train : Training the model with train split and saving the model
- test : Validating the model with test split, generating the report and graph

![DVC Metrics DIff](https://user-images.githubusercontent.com/103778538/177500957-fb73a58d-d625-4351-9e71-287e831002ef.png)

## dvc plots

- Plots generated by dvc plots show

## Continuous ML
- Github actions are configured for each push
- Everytime a docker instance is created
- It performs the setup using requirement installation
- Then performs the evaluate step of DVC using ''dvc repro''

## AWS Configuration
- The project has been configured in AWS Cloud9 IDE
- It successfully ran the code
- Able to perform DVC operations and Git operations from cloud.
- The same can be used in future for web app deployment.


![AWS_Cloud9](https://user-images.githubusercontent.com/103778538/177501013-4904d298-f3f2-4efa-a254-248d751589ab.png)



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
