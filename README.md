# Diabetes Prediction Model

## Introduction
This project aims to predict diabetes using various health metrics. The model is trained on a dataset containing several health-related features.

## Dataset Description
The dataset `diabetes.csv` includes the following features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1) indicating the presence or absence of diabetes

## Model Training
The model training script `modeltrain.py` performs the following steps:
1. Loads the dataset and preprocesses it by replacing zero values in specific columns with their median.
2. Splits the data into features and the target variable.
3. Normalizes the features using `StandardScaler`.
4. Splits the data into training (80%) and testing (20%) sets.
5. Trains an XGBoost classifier and evaluates its performance, printing accuracy and a classification report.

## Installation Instructions
To run this project, you need to install the following libraries:
```bash
pip install pandas numpy scikit-learn xgboost
```

## Usage Instructions
To run the model training script, execute the following command:
```bash
python modeltrain.py
```
The output will include the model's accuracy and a classification report.

## Conclusion
This model can be used to predict diabetes based on health metrics. Future improvements could include hyperparameter tuning, exploring different algorithms, and enhancing data preprocessing techniques.
