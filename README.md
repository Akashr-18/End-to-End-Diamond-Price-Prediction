# End-to-End-Diamond-Price-Prediction

# Forest Cover Prediction
![my badge](https://img.shields.io/badge/Python-3-blue)
![my badge](https://img.shields.io/badge/Machine-Learning-brightgreen)
![my badge](https://img.shields.io/badge/Flask-App-green)
![my badge](https://img.shields.io/badge/ML-Flow-yellowgreen)
![my badge](https://img.shields.io/badge/AI-OPS-orange)
![my badge](https://img.shields.io/badge/-Heroku-purple)
![my badge](https://img.shields.io/badge/-GIT-green)
![my badge](https://img.shields.io/badge/-DVC-darkblue)

# About The Project

This project has been developed to predict what type of trees grow in an area based on it's surrounding characteristics. The dataset consists of observations of various catagrophic features (No sensor data). The dataset used for the project mostly contain observations from Roosevelt National Forest in Colorado.

# Project Description 

This project has been developed in semi-supervised learning way. Initally, the data without the target column is passed through a clustering model which predicts the cluster. Then for each cluster, a ML supervised learning model has been used to predict the type of forest cover. A web app has been developed for this project which takes a CSV file as an input and returns the predictions as a result. The app is deployed in Heroku.

# Dataset Used

This dataset is part of the UCI Machine Learning Repository and more information about the dataset can be found below.

Dataset : [Link](https://archive.ics.uci.edu/ml/datasets/Covertype)

## Commands to connect DagsHub

```bash 
set MLFLOW_TRACKING_URI=https://dagshub.com/Akashr-18/End-to-End-Diamond-Price-Prediction.mlflow
```

```bash 
set MLFLOW_TRACKING_USERNAME=Akashr-18 
```

```bash 
set MLFLOW_TRACKING_PASSWORD=605c3e58682b85cc8ec99a16fc3a850355af9be5
```
