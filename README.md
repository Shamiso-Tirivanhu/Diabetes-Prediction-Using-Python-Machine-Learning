# Diabetes-Prediction-Using-Python-Machine-Learning

## Table of Contents

1. [Overview](#overview)

2. [Features](#feature)

3. [Dataset](#dataset)

4. [Requirements](#requirements)

    - [Required Libraries](#required-libraries)

5. [Implementation Steps](#implementation-status)

6. [Model Training](#model-training)

7. [Model Evaluation](#model-evaluation)

8. [How to Use](#how-to-use)

9. [Results](#results)

10. [Contributing](#contribution)

## 1. Overview

Diabetes is a chronic disease that affects millions of people worldwide. Early detection can help in managing the condition effectively and improving quality of life. This project utilizes Python and machine learning to build a predictive model for diabetes diagnosis based on medical input features.


A diagram preview of the dataset that l was working on this project.

![image_alt](https://github.com/Shamiso-Tirivanhu/Diabetes-Prediction-Using-Python-Machine-Learning/blob/3ebb13a556487331fe914a2af71ab12eaf58336a/A%20data%20preview%20of%20the%20Diabetes%20dataset%20-first%20five%20rows%20of%20the%20dataset.png)

## 2. Features

- Data preprocessing and cleaning

- Exploratory Data Analysis (EDA) to understand feature correlations

- Implementation of machine learning models

- Model evaluation using accuracy metrics

- Deployment-ready code for real-world applications

## 3. Dataset

The dataset used in this project is sourced from the Pima Indians Diabetes Database, which contains the following features:

- Pregnancies

- Glucose

- Blood Pressure

- Skin Thickness

- Insulin

- BMI

- Diabetes Pedigree Function

- Age

Outcome (0 - No Diabetes, 1 - Diabetes)

## 4. Requirements

To run this project, install the necessary dependencies using the following command:

| pip install -r requirements.txt |
|---------------------------------|

Required Libraries

- Python 3.x

- Pandas

- NumPy

- Scikit-learn

- Google colab Notebook (optional for analysis)

## 5. Implementation Steps

1. Data Preprocessing: Handle missing values, normalize numerical features, and prepare the dataset for modeling. A Standard Scaler was used to process the data into a standardized format, making it easier for the model to train efficiently and improve accuracy.

2. Exploratory Data Analysis (EDA): Visualize correlations, distributions, and feature importance.

## 6. Model Training

After preprocessing, the dataset was split into training and test sets using an 80-20 ratio. The model was trained using Support Vector Machine (SVM), which is effective for classification tasks.

- Training Accuracy: Achieved an accuracy score of 77% on the training data.

- Test Accuracy: Achieved an accuracy score of 78% on the test data


Training & Evaluation for Diabetes Machine Learning model for the Diabetes data set

![image_alt]()


## 7. Model Evaluation

The model’s performance was evaluated using the Accuracy Score, which measures the percentage of correctly predicted instances.

## 8. How to Use

1. Clone the repository:

| git clone https://github.com/Shamiso-Tirivanhu/diabetes-prediction.git cd diabetes-prediction                                             |
|-----------------------------------------------------------------------------------------|

2. Run the Jupyter Notebook:

| jupyter notebook Diabetes_Prediction_Using_Python_&_Machine_Learning.ipynb |
|----------------------------------------------------------------------------|

3.Train and evaluate the model as per the notebook instructions.

## 9. Results

The model achieves an accuracy of 78% on test data.


## 10. Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request with enhancements.

