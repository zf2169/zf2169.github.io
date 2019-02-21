---
layout: post
title: "Machine Learning Algorithms Summary"
date: 2019-01-30
---

I summarize the top 15 algorithms and the 5 most commonly asked questions (The Big Five).

1. What are the basic concepts? What problem does it solve? 
2. What are the assumptions?
3. What are the steps of the algorithm?
4. What is the cost function?
5. What are the advantages/disadvantages?

Top 15 Machine Learning Algorithms Summary
Linear Regression
Regression with Lasso
Regression with Ridge
Stepwise Regression
Logistic Regression
Naïve Bayes
K-Nearest Neighbors
K-means Clustering
Decision Tree
Random Forest
Ada-Boost
Gradient Boosting
SVM (Support Vector Machine)
PCA (Principal Component Analysis)
Neural Networks
1. Linear Regression
1.1 What are the basic concepts? What problem does it solve?
Basic Concept: linear regression is used to model relationship between continuous variables. Xs are the predictor, explanatory, or independent variable. Y is the response, outcome, or dependent variable.

1.2 What are the assumptions?
The mean function is linear
The variance function is constant
Residuals are statistically independent, have uniform variance, are normally distributed
The epsilon ϵ term is assumed to be a random variable that has a mean of 0 and normally distributed
The ϵ term has constant variance σ2 at every value of X (Homoscedastic)
The error terms are also assumed to be independent. ϵ ~ N(0,σ2)
1.3 What are the steps of the algorithm?
We want to find the best fit line:

