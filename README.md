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
    
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
