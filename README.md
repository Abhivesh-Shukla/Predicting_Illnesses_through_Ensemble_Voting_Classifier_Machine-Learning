# Predicting Illnesses through Symptomatic Patterns using Ensemble Voting Classifier

## Introduction

This repository contains the code and data for predicting illnesses based on symptomatic patterns using various machine-learning models. The core objective of this initiative is to leverage machine learning to enhance the precision and efficiency of disease diagnostics. Utilizing the GitHub Collaborative Disease-Symptom Dataset, this project aims to provide personalized diagnostic insights to improve patient care.

## Features

- **Machine Learning Models**: Logistic Regression, Support Vector Machine (SVM), Decision Tree Classifier, Random Forest Classifier, K-Nearest Neighbors (KNN), and an Ensemble model.
- **Confusion Matrices**: Essential for evaluating the performance of each model, these matrices help identify true positives and discern false positives and negatives.
- **Weighted Symptom Analysis**: Symptoms are weighted based on their severity and prevalence to improve prediction accuracy.
- **Performance Metrics**: Precision, Recall, F1 Score, and Accuracy are used to evaluate model performance.

## Dataset

The dataset used in this project is the GitHub Collaborative Disease-Symptom Dataset. It includes a rich array of symptom-disease pairs, allowing for detailed analysis and effective model training. The dataset features:
- **Disease Categories**: 41 distinct diseases such as Influenza, Diabetes, Chronic Kidney Disease, Hypertension, and Tuberculosis.
- **Symptoms**: 132 specific symptoms like Fever, Headache, Cough, and Muscle Aches.

## Methodology

1. **Data Preprocessing**: Cleaning and preparing the data for training.
2. **Model Training**: Training various machine learning models on the dataset.
3. **Model Evaluation**: Using confusion matrices and performance metrics to evaluate each model's effectiveness.
4. **Ensemble Approach**: Combining multiple models to enhance predictive accuracy and mitigate overfitting.

## Workflow of the Project

<img width="834" alt="Workflow of the Program" src="https://github.com/PrantikGhosh/Predicting_Illnesses_through_Ensemble_Voting_Classifier_Machine-Learning/assets/84172492/849980d4-70af-4d5a-941e-8324f1464e95">

## Experimental Results

- **Random Forest Classifier**: Demonstrated high predictive accuracy by leveraging multiple decision trees.
- **K-Nearest Neighbors**: Highlighted the importance of similarity in symptomatic patterns for effective classification.
- **Ensemble Model**: Showcased the benefits of integrating outputs from multiple models, achieving a record-breaking 98.96% accuracy.

## Visualizations

- **Confusion Plots**: Visually represent the accuracy of predictive models.
- **Line Graphs**: Compare the accuracy of different models.
- **Bar Charts**: Analyze precision, F1 score, and recall across all models.

## Software Requirements

- **Software**: Python 3.10, Jupyter Notebook, Anaconda Distribution.
- **Libraries**: scikit-learn, pandas, NumPy, Matplotlib, Seaborn.

## Conclusion

This project demonstrates the transformative impact of machine learning on healthcare diagnostics. By meticulously analyzing symptomatic data and employing sophisticated machine-learning techniques, this initiative aims to pave the way for more accurate and personalized diagnostic tools, ultimately enhancing patient care.
