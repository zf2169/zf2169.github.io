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
We want to find the best fit line: \$$ \hat{y_i}= \theta_0 + \theta_1 x_i $$

Usually, we achieve this by using **least squares estimation**, which finds the values of $$\theta_0$$ and $$\theta_1$$ that minimize the sum of the squared predicted errors: \$$ \sum_{i=1}^n (y_i- \hat{y_i})^2 $$

thus we have, \$$\theta_1= \frac {\sum_{i=1}^n (x_i-\bar{x}) (y_i-\bar{y})} {\sum_{i=1}^n (x_i−\bar{x})^2} $$

and \$$\theta_0= \bar{y} - \theta_1 \bar{x}$$

### 1.4 What is the cost function?
\$$J(\theta)= \frac{1}{n} \sum_{i=1}^n (y_i− \hat{y_i})^2$$

### 1.5 What are the Advantages and Disadvantages?

**Advantages:**
- Linear regression is an extremely simple method, very easy to use, understand, and explain.
- The best fit line acquires the minimum error from all the points, which has high efficiency.

**Disadvantages:**
- It assumes a straight-line relationship between dependent and independent variables, which sometimes is incorrect.
- Very sensitive to the outliers(anomalies) in the data.
- In n<p cases (the number of parameters larger than samples), linear regression tends to model noise rather than relationship.


## 2. Regression with Lasso
### 2.1 What are the basic concepts/ What problem does it solve? 
- **Lasso** is a regularization method, usually used in **linear regression**, performing both variable selection and regularization to reduce overfitting.

- Lasso uses L1 penalty when fitting the model, L1 penalty := $$\sum_{j=1}^n |\theta_j|$$, is the sum of the absolute values of the coefficients.

- Lasso can force regression coefficients to be exactly 0.

### 2.2 What are the assumptions?
The same as linear regression

### 2.3 What is the process of the algorithm?
Similar to linear regression, we want to find the best fit line: $$ \hat{y_i} = \theta_0 + \theta_1 x_i $$. However, we add an L1 penalty to the previous cost function, that is, we want to find \$$ \hat{\theta} = \arg\min_{\theta} \frac{1}{n} \sum_{i=1}^n (y_i-\theta_0-x_i^T\theta)^2 + \lambda \sum_{j=1}^n |\theta_j| $$

`note: the penalty only penalize` $$\theta_1, \dots, \theta_n$$, `not` $$\theta_0$$

When $$\lambda=0$$, Lasso gives the least squares fit, same as linear regression.

When $$\lambda \to \infty$$, Lasso makes all estimated coefficients nearly equal 0, which gives a null model.

### 2.4 What is the cost function?
\$$J(\theta)= \frac{1}{n} \sum_{i=1}^n (y_i-\theta_0-x_i^T\theta)^2)+\lambda \sum_{j=1}^n |\theta_j|$$


### 2.5 What are the advantages and disadvantages?
**Advantages:**
- Lasso can act as feature selection since it can force coefficients to be 0.
- Produces simple models which are easier to interpret.
- Lasso to perform better in a setting where a relatively small number of predictors have substantial coefficients, and the remaining predictors have coefficients that are very small or that equal zero
- Lasso tends to outperform ridge regression in terms of bias, variance, and MSE

**Disadvantages:**
- For n<p case (high dimensional case), LASSO can at most select n features.
- For usual case, where we have correlated features (usually for real word datasets, such as gene data), LASSO selects only one feature from a group of correlated features, that is, LASSO doesn't help in grouped selection.
- For n>p case, it is often seen that Ridge outperforms LASSO for correlated features.


## 3. Regression with Ridge
### 3.1 What are the basic concepts/ What problem does it solve? 
- **Ridge** is also a regularization method, which uses **L2 penalty** in linear regression to reduce overfitting.

- Ridge can only regression coefficients to approach 0, but not exactly 0.

### 3.2 What are the assumptions?
Same as linear regression

### 3.3 What are the steps of the algorithm?
Similar to Lasso, Ridge uses L2 penalty (the sum of the squares of the coefficients) instead, that is, we try to find \$$ \hat{\theta} = \arg\min_{\theta} \frac{1}{n} \sum_{i=1}^n (y_i-\theta_0-x_i^T\theta)^2 + \lambda \sum_{j=1}^n \theta_j^2 $$

### 3.4 What is the cost function?
\$$J(\theta)= \frac{1}{n} \sum_{i=1}^n (y_i-\theta_0-x_i^T\theta)^2 + \lambda \sum_{j=1}^n \theta_j^2$$

### 3.5 What are the Advantages/Disadvantages?
**Advantages:**
- As $$\lambda$$ increases, the shrinkage of the ridge coefficient estimates leads to a substantial reduction in the variance of the predictions, at the expense of a slight increase in bias
- Ridge regression works best in situations where the least squares estimates have high variance. Meaning that a small change in the training data can cause a large change in the least squares coefficient estimates
- Ridge will perform better when the response is a function of many predictors, all with coefficients of roughly equal size
- Ridge also has substantial computational advantages over best subset selection.

**Disadvantages:**
- Ridge is not able to shrink coefficients to exactly 0, thus it cannot perform variable selection.
