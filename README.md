# Vaccine-Usage-Prediction
## Project Overview
The aim of this project is to predict how likely individuals are to receive their H1N1 flu vaccine. I believe the prediction outputs (model and analysis) of this study will give public health professionals and policy makers, as an end user, a clear understanding of factors associated with low vaccination rates. This in turn, enables end users to systematically act on those features hindering people to get vaccinated.

In this project, I used vacccine dataset to predict whether or not respondents got the H1N1 vaccine. It is important to understand past vaccination patterns in order to understand those of more recent pandemics, such as COVID-19. The most influential factors determining vaccination status are found to be Doctor Recommendation of H1N1 vaccine, Health Insurance, Opinion of H1N1 Vaccine Effectiveness, and Opinion of H1N1 Risk by our final model. I used four machine learning models to make the predictions such as Decision Tree Classifier, Logistic Regression, Random Forest and  K-Nearest Neighborhood Classifier. The Logistic Regression yielded the best accuracy and precision score.

So as to classify exactly those who got H1N1 flu shot from those that did not, I need a higher accuracy of the model outputs, as well as a high precision score, which is related to a low false positive rate (those who predicted to be vaccinated but did actually not get H1N1 flu shot). 

## Data Overview

These data inquires about whether or not people received the seasonal flu and/or the H1N1 flu vaccination, as well as their demographic, behavioral, and health factors. There are 26,707 respondents to this survey. The dataset consists of 34 predictors including target variable i.e. h1n1 vaccine and filled missing values using mode.

## Problem statement

This project aims to predict the likelihood of people taking an H1N1 flu vaccine
using Logistic Regression. It involves analyzing a dataset containing various features
related to individuals' behaviors, perceptions, and demographics, and building a
predictive model to determine vaccine acceptance.
Predict the probability of individuals taking an H1N1 flu vaccine based on their
characteristics and attitudes. This can help healthcare professionals and policymakers
target vaccination campaigns more effectively.

## Methodology

I wanted to use a variety of different models so as to find the most accurate model. Because there are many different hyperparameters for each model and I did not know the optimal combinations, I used GridSearrchCV to find the best combinations for each model. I specified class weight to be balanced in order to address the class imbalance issue for our models, whenever possible. I also analyzed the accuracy score, precision score, fl score, and roc-auc curve for each model. I also compared the different roc-auc curves of each model, to choose the final model. Additionally, I looked closely at the confusion matrix to see whether or not I  minimize false positives.Logistic Regression gave me the best accuracy and precision scores, so I chose it to be our final model.

## Project Results

The Logistic Regression model has the highest overall scores of all the models done so far, and also does the best job of minimizing the false positives. This  model did not overfit to the training set, I got similar AUC, precision and accuracy scores for the holdout set. Because the model does a good job of minimizing the false positive rate on the hold out data, I fairly confident that it will generalize well to unseen data and will accurately help public health offials determine the people who didn't get the vaccine. 

I also looked into feature importances to understand the relationship between the features and vaccination behavior. I saw that the demographic and behavioral features are not that important compared to health related factors and opinions. The doctor recommendation, health insurance, opinion of vaccine efficiency, and opinion of H1N1 risk are the top important factors in determining people's vaccination status.
