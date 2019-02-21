---
layout: post
title: "Machine Learning Algorithms Summary I"
date: 2019-01-30
---

# Part 0
## 1.What is Machine Learning?
Tom Mitchell in his book Machine Learning provides a definition in the opening line of the preface:
> A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in T, as measured by P, improves with experience E.\\ 
where,\\
T := The goal of the algorithm(prediction/classification/cluster).\\
P := The cost function of the algorithm.\\
E := The process of the algorithm.

Therefore, based on the definition above, I summarize five questions that we should ask to know about an algorithm and top 15 most popular machine learning algorithms. (**BIG 5 $$\times$$ TOP 15**)
### BIG 5 QUESTIONS
#### 1. What are the basic concepts/ What problem does it solve? 
#### 2. What are the assumptions?
#### 3. What is the process of the algorithm?
#### 4. What is the cost function?
#### 5. What are the advantages and disadvantages?


### TOP 15 ALGORITHMS
#### 1. Linear Regression
#### 2. Regression with Lasso
#### 3. Regression with Ridge
#### 4. Stepwise Regression
#### 5. Logistic Regression
#### 6. Naive Bayes
#### 7. K-Nearest Neighbors
#### 8. SVM (Support Vector Machine)
#### 9. K-means Clustering
#### 10. Decision Tree
#### 11. Gradient Boosting
#### 12. Ada-Boost
#### 13. Random Forest
#### 14. Neural Network
#### 15. PCA (Principal Component Analysis)


# Part I: BIG 5 $$\times$$ TOP 15
## 1. Linear Regression
### 1.1 What are the basic concepts/ What problem does it solve? 
**Basic concepts**: A **linear** approach to model the relationship between continuous variables. '$$X$$'s are the predictors( or explanatory, or independent variables). '$$Y$$' is the outcome( or response, or dependent variable). \\ \\
**Model**: $$\hat{Y}= \theta X+ \epsilon $$

### 1.2 What are the assumptions?
- The mean function is linear
- The variance function is constant
- Residuals are statistically independent, normally distributed with uniform variance
- The error term $$\epsilon$$ is assumed to be normally distributed with a mean of 0 and constant variance $$\sigma^2$$ at every value of $$X$$ (Homoscedastic)
- The error terms are assumed to be independent, that is $$\epsilon \sim N(0,\sigma^2)$$

### 1.3 What is the process of the algorithm?
We want to find the best fit line:

\$$ \hat{y_i}y= \theta_0 + \theta_1 x_i $$

Usually, we achieve this by using **least squares estimation**, which finds the values of $$\theta_0$$ and $$\theta_1$$ that minimize the sum of the squared predicted errors:

\$$ \sum_{i=1}^n (y_i- \hat{y_i})^2 $$

thus we have,

\$$\theta_1= \frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^n (x_i−\bar{x})^2$$

and

\$$\theta_0= \bar{y} - \theta_1 \bar{x}$$

### 1.4 What is the cost function?
\$$J(\theta)= \frac{1}{n} \sum_{i=1}^n (y_i− \hat{y_i})^2$$

### 1.5 What are the Advantages and Disadvantages?

**Advantages:**
- Linear regression is an extremely simple method, very easy to use, understand, and explain.
- The best fit line acquires the minimum error from all the points, which has high efficiency.

**Disadvantages:**
- It assumes a straight-line relationship between dependent and independent variables, which sometimes is incorrect.
- Very sensitive to the outliers(anomalies) in the data.
- In n<p (the number of parameters larger than samples) cases, linear regression tends to model noise rather than relationship.


