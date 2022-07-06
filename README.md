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
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
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
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    
    
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
      train_scores.json        -- Report containing the Train Size, Train Score and Solver
      test_scores.json        -- Report containing the Test Size, Test Score, ROC AUC Score and Solver
      cm.csv                  -- Predicted Labels and Test Labels
      prc.json                -- Precision values
      roc.json                -- roc values
      prediction.png    -- Scatter plot showing the Prediction vs. Ground Truth

## models -- 
      final_model.joblib  -- Model saved using the joblib utility


## DVC Stages

DVC has 3 stages
- make_data : preparing the data, performing train test split
- train : Training the model with train split and saving the model
- test : Validating the model with test split, generating the report and graph

![image](https://user-images.githubusercontent.com/47142192/177393533-df06f843-0309-48f0-9592-2b01b2cfbbb7.png)


## Continuous ML
- Github actions are configured for each push
- Everytime a docker instance is created
- It performs the setup using requirement installation
- Then performs the evaluate step of DVC using ''dvc repro test''

## AWS Configuration
- The project has been configured in AWS Cloud9 IDE
- It successfully ran the code
- Able to perform DVC operations and Git operations from cloud.
- The same can be used in future for web app deployment.


![image](https://user-images.githubusercontent.com/47142192/177395538-a507c2d3-f672-43fe-bdb3-920df5801b38.png)


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
