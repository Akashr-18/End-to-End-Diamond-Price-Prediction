# Diamond-Price-Prediction
![my badge](https://img.shields.io/badge/Python-3-blue)
![my badge](https://img.shields.io/badge/Machine-Learning-brightgreen)
![my badge](https://img.shields.io/badge/Flask-App-green)
![my badge](https://img.shields.io/badge/ML-Flow-yellowgreen)
![my badge](https://img.shields.io/badge/AI-OPS-orange)
![my badge](https://img.shields.io/badge/-GIT-green)
![my badge](https://img.shields.io/badge/-DVC-darkblue)

## Features
- **Predictive Models Compared**:
  - Linear Regression
  - Lasso
  - Ridge
  - ElasticNet
  - Decision Tree
  - XGBoost
  - Random Forest
- **Logging**: Implementation of logging for tracking events and debugging.
- **Custom Exceptions**: Custom exceptions for handling errors.
- **Modular Codebase**: Organized and modularized code structure for easy maintenance and scalability.
- **Integration with DVC**: Data Version Control (DVC) is integrated for code reproducibility.
- **Integration with MLflow**: MLflow is integrated for experiment tracking, model management, and reproducibility.
- **Integration with DagsHub**: DagsHub is used for collaboration, sharing, and versioning of machine learning projects.
- **Integration with GitHub Actions**: GitHub Actions is employed for continuous integration and deployment.
- **Integration with Airflow**: Airflow is used for workflow orchestration and automation (Continous Training).

## About The Project

The objective of this project is to forecast the price of diamonds by considering their diverse characteristics, including Color, Clarity, Cut, Table, Depths, and Dimensions. <br><br>

| Attribute | Description |
| --------- | ----------- |
| carat     | Weight of the diamond |
| cut       | Quality of the cut (Fair, Good, Very Good, Premium, Ideal) |
| color     | Color of the diamond, from J (Worst) to D (Best) |
| clarity   | How clear the diamond is, from I1 (Worst) to IF (Best) |
| x         | Length in mm |
| y         | Width in mm |
| z         | Depth in mm |
| depth     | Total depth percentage. Formula: z / mean(x,y) |
| table     | Width of top of diamond relative to the widest point |
| price     | Price in US dollars |


## Dataset Used

This dataset used for this project is Diamond dataset taken from Kaggle.

## Usage

1. Run the Streamlit app:

    ```bash
    python app.py
    ```

2. Open your web browser and go to [http://localhost:5000](http://localhost:5000).

3. Preview of the Web App

#### Home page :

<img width="960" alt="image" src="https://github.com/Akashr-18/Data_Store/blob/main/Screenshot%20(14).png?raw=true">
<br>

#### Enter the values :

<img width="960" alt="image" src="https://github.com/Akashr-18/Data_Store/blob/main/Screenshot%20(15).png?raw=true">
<br>

#### Prediction Output :

<img width="960" alt="image" src="https://github.com/Akashr-18/Data_Store/blob/main/Screenshot%20(16).png?raw=true">
<br>
