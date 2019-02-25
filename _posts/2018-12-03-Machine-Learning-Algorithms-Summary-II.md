---
layout: post
title: "Machine Learning Algorithms Summary II"
date: 2018-12-03
---

## 9. KNN (K-Nearest Neighbors)
### 9.1 What are the basic concepts/ What problem does it solve?
**KNN (K-nearest neighbors)** attempts to estimate the conditional distribution of Y given X, and then classify agiven observation to the class with highest estimated probability. 

### 9.2 What are the assumptions? What is the process of the algorithm? What is the cost function?
Given a positive integer $$K$$ and a test observation $$x_0$$, the KNN classifier first identifies the neighbors $$K$$ points in the training data that are closest to $$x_0$$, represented by $$N_0$$.

It then estimates the conditional probability for class $$j$$ as the fraction of points in $$N_0$$ whose response values equal $$j$$: \$$ Pr(Y = j\vert X = x_0) = \frac{1}{K}\sum_{i \in N_0} I(y_i = j)$$.

Finally, KNN applies Bayes rule and classifies the test observation $$x_0$$ to the class with the largest probability.

### 9.5 What are the advantages and disadvantages?
**Advantages:**
- Easy to interpret output
- Naturally handles multi-class cases
- Predictive power, can do well in practice with enough representative data

**Disadvantages:**
- Large search problem to find nearest neighbors
- The choice of K has a drastic effect on the KNN classifier


## 10. Decision Tree
### 10.1 What are the basic concepts/ What problem does it solve?
Decision trees can be applied to both regression and classification problems.

These involve stratifying or segmenting the predictor space into a number of simple regions, we typically use the mean or the mode of the training observations in the region to which it belongs to make a prediction. 

**Decision Tree** gets the name "tree" since the set of splitting rules used to segment the predictor space can be summarized in a tree.

### 10.2 What is the process of the algorithm?
#### 10.2.1 Regression Tree (Recursive binary splitting for example)
1. Select the predictor $$X_j$$ and the cutpoint $$s$$ such that splitting the predictor space into the regions $$\{X \vert X_j < s\}$$ and $$\{X \vert X_j \ge s \}$$ leads to the greatest possible reduction in RSS. 
2. For any $$j$$ and $$s$$, we define the pair of half-planes \$$R_1(j, s) = \{X \vert X_j < s\}$$ and $$R_2(j, s) = \{X \vert X_j \ge s \}$$, and we seek the value of $$j$$ and $$s$$ that minimize the equation: \$$ \sum_{i:x_i\in R_1(j,s)} (y_i-\hat{y}_{R_1})^2$$


where ˆyR1 is the mean response for the training observations in R1(j, s),
and ˆyR2 is the mean response for the training observations in R2(j, s).
Finding the values of j and s that minimize can be done quite quickly,
especially when the number of features p is not too large.

3. Repeat this process, looking for the best predictor and best cutpoint in order to split the data further so as to minimize the RSS within each of the resulting regions.

4. The process contincues until a stopping criterion is reached: e.g. no region contains more than 5 observations.



### 10.3 What is the cost function?


### 10.4 What are the advantages and disadvantages?











