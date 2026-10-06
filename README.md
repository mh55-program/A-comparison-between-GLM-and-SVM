::: center
**Nonlinear Programming**\
Gamma Regression and Support Vector Machines\
**NLPCA2**\
Faculty of Mathematics, Statistics, and Computer Science\
**Mohammad Shahinfar**
:::

# Overview {#overview .unnumbered}

This repository contains the implementation and analysis of a project
for the **Nonlinear Programming** course.

The project studies two important optimization-based statistical and
machine-learning problems:

1.  **Gamma Regression**

2.  **Support Vector Machines (SVMs)**

The main objective is to connect the mathematical formulation of
statistical learning problems with practical optimization algorithms.
The project includes mathematical derivations, convexity analysis,
manual implementations of numerical optimization methods, and
comparisons with established Python libraries.

# Part I: Gamma Regression

## Generalized Linear Models

Generalized Linear Models (GLMs) extend ordinary linear regression to
response variables that follow non-normal probability distributions.

A GLM relates the expected response $\mu$ to a linear combination of the
explanatory variables through an appropriate link function.

Different distributions and link functions can be used depending on the
structure of the response variable.

For example, logistic regression models a binary response:

$$Y \sim \operatorname{Bernoulli}(p).$$

The probability $p$ is modeled using the logistic (sigmoid) function:

$$p = \sigma(x^\top \beta)
=
\frac{1}{1+e^{-x^\top\beta}}.$$

Equivalently, the log-odds are modeled linearly:

$$\log\left(\frac{p}{1-p}\right)
=
x^\top\beta.$$

The project specification uses logistic regression as a motivating
example before introducing Gamma Regression.

## Gamma Regression

Gamma Regression is appropriate for modeling a response variable that
is:

- strictly positive,

- continuous,

- right-skewed,

- and potentially characterized by variance increasing with the mean.

The model assumes

$$Y \sim \operatorname{Gamma}(\nu,\lambda),$$

where the Gamma density is given by

$$f_Y(y)
=
\frac{\lambda^\nu}{\Gamma(\nu)}
y^{\nu-1}e^{-\lambda y},
\qquad y\geq 0.$$

The expected value and variance are

$$E[Y]
=
\frac{\nu}{\lambda}
=
\mu,$$

and

$$\operatorname{Var}(Y)
=
\frac{\nu}{\lambda^2}.$$

Using

$$\mu=\frac{\nu}{\lambda},$$

the variance can be written as

$$\operatorname{Var}(Y)
=
\mu^2\frac{1}{\nu}.$$

Thus,

$$\operatorname{Var}(Y)=\sigma^2\mu^2,$$

where

$$\sigma^2=\frac{1}{\nu}.$$

## Log Link

The Gamma regression model uses the logarithmic link function:

$$\log(\mu_i)
=
x_i^\top\beta.$$

Therefore,

$$\mu_i
=
\exp(x_i^\top\beta).$$

This guarantees that the predicted mean remains positive.

## Likelihood

Suppose we have $n$ independent observations

$$(x_1,y_1),\ldots,(x_n,y_n),$$

where

$$Y_i\sim\operatorname{Gamma}(\nu,\lambda_i).$$

The mean is

$$\mu_i
=
E[Y_i]
=
\frac{\nu}{\lambda_i}.$$

Using the log-link,

$$\log(\mu_i)=x_i^\top\beta.$$

The likelihood of the observed dataset is

$$L(\beta)
=
\prod_{i=1}^{n}f_Y(y_i).$$

The log-likelihood is

$$\ell(\beta)
=
\log L(\beta).$$

After removing terms that do not depend on $\beta$, the corresponding
Negative Log-Likelihood (NLL) becomes the objective function to be
minimized.

The optimization problem can therefore be written as

$$\hat{\beta}
=
\arg\min_{\beta}
\operatorname{NLL}(\beta).$$

The project also requires showing that the resulting objective is convex
in $\beta$.

## Exploratory Data Analysis

Before fitting the Gamma regression model, exploratory data analysis is
performed.

The analysis includes:

- Histogram or KDE of the response variable $Y$.

- Scatter plots of $Y$ against each predictor.

- Investigation of positivity and right-skewness.

- Investigation of whether the variance increases with the mean.

- Investigation of nonlinear or multiplicative relationships.

These observations help explain why Ordinary Least Squares (OLS) may not
be appropriate for the dataset.

## Optimization Methods

Three optimization approaches are implemented.

### 1. Convex Optimization with CVXPY

The Gamma regression NLL is formulated as a convex optimization problem
and solved using `CVXPY`.

### 2. Gradient Descent

Gradient Descent is implemented manually.

The generic update rule is

$$\beta^{(k+1)}
=
\beta^{(k)}
-
\eta_k\nabla f(\beta^{(k)}),$$

where

- $\beta^{(k)}$ is the parameter vector at iteration $k$,

- $\eta_k$ is the learning rate,

- $\nabla f(\beta^{(k)})$ is the gradient of the objective.

The implementation includes an appropriate learning-rate strategy and
stopping criterion.

### 3. Newton--Raphson Method

Newton--Raphson optimization is also implemented manually.

The update rule is

$$\beta^{(k+1)}
=
\beta^{(k)}
-
\left[
\nabla^2 f(\beta^{(k)})
\right]^{-1}
\nabla f(\beta^{(k)}),$$

where

$$\nabla f(\beta)$$

is the gradient and

$$\nabla^2 f(\beta)$$

is the Hessian matrix of the objective function.

The project requires both Gradient Descent and Newton--Raphson to be
implemented without using pre-built optimization solvers. NumPy may be
used for matrix operations.

## Statsmodels Comparison

The manually implemented methods are compared with the Gamma GLM
implementation provided by `statsmodels`.

The reference model is specified as

``` {.python language="Python"}
import statsmodels.api as sm

model = sm.GLM(
    y,
    X,
    family=sm.families.Gamma(
        link=sm.families.links.log()
    )
)

result = model.fit()
```

The estimated coefficients are compared across the different approaches.

## Gamma Regression Evaluation

A 10-fold cross-validation procedure is used to evaluate and compare the
different methods.

The following criteria are considered:

- Adjusted $R^2$

- Root Mean Squared Error (RMSE)

- Computation time

The Root Mean Squared Error is defined as

$$\operatorname{RMSE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}.$$

Adjusted $R^2$ is custom-computed for the project since it is not
directly provided by the GLM implementation in the required form.

# Part II: Support Vector Machines

## Overview

Support Vector Machines (SVMs) are optimization-based classification
methods that seek a separating hyperplane with a large margin.

For a linear classifier,

$$w^\top x+b=0$$

defines the decision boundary.

For linearly separable data, the Hard-Margin SVM solves

$$\min_{w,b}
\frac{1}{2}\|w\|^2$$

subject to

$$y_i(w^\top x_i+b)\geq 1,
\qquad
\forall i.$$

The objective is convex and the constraints are linear, making this a
convex optimization problem.

## Hard-Margin SVM

The Hard-Margin SVM assumes that the training data is perfectly linearly
separable.

The optimization problem is

$$\boxed{
\begin{aligned}
\min_{w,b}\quad
&\frac{1}{2}\|w\|^2\\
\text{subject to}\quad
&y_i(w^\top x_i+b)\geq1,
\qquad i=1,\ldots,n.
\end{aligned}
}$$

The margin is related to the norm of the weight vector by

$$\text{Margin}
=
\frac{2}{\|w\|}.$$

Therefore, minimizing $\frac12\|w\|^2$ maximizes the separation margin.

## Soft-Margin SVM

Perfect linear separability is not always possible in practical
datasets.

The Soft-Margin formulation introduces slack variables

$$\xi_i\geq0$$

to allow some observations to violate the margin constraints.

The optimization problem becomes

$$\boxed{
\begin{aligned}
\min_{w,b,\xi}\quad
&
\frac{1}{2}\|w\|^2
+
C\sum_{i=1}^{n}\xi_i
\\
\text{subject to}\quad
&
y_i(w^\top x_i+b)
\geq
1-\xi_i,
\\
&
\xi_i\geq0,
\qquad i=1,\ldots,n.
\end{aligned}
}$$

The parameter $C$ controls the trade-off between maximizing the margin
and penalizing classification/margin violations.

A larger $C$ places more emphasis on correctly classifying training
observations, whereas a smaller $C$ allows more violations in exchange
for a potentially simpler decision boundary.

## Kernel SVM

A linear hyperplane may fail when the classes are not linearly separable
in the original feature space.

The **kernel trick** allows the SVM to implicitly operate in a
higher-dimensional feature space without explicitly computing the
transformed feature vectors.

A kernel function can be written as

$$K(x_i,x_j)
=
\phi(x_i)^\top\phi(x_j),$$

where $\phi(\cdot)$ is a feature mapping.

Common kernel functions include:

- Linear kernel

- Polynomial kernel

- Radial Basis Function (RBF) kernel

- Sigmoid kernel

In the dual formulation, the optimization problem can be expressed in
terms of the kernel matrix.

For example, the Hard-Margin SVM dual has the form

$$\boxed{
\begin{aligned}
\max_{\alpha}\quad
&
\sum_{i=1}^{n}\alpha_i
-
\frac{1}{2}
\sum_{i=1}^{n}
\sum_{j=1}^{n}
\alpha_i\alpha_j
y_i y_j
K(x_i,x_j)
\\
\text{subject to}\quad
&
\alpha_i\geq0,
\\
&
\sum_{i=1}^{n}\alpha_i y_i=0.
\end{aligned}
}$$

This formulation demonstrates how the kernel function replaces the
explicit inner products between transformed feature vectors.

## Regularization

Regularization is used to reduce overfitting and control model
complexity.

For Soft-Margin SVM, the regularization parameter $C$ determines the
relative importance of margin violations.

The project investigates how changing $C$ affects:

- The learned decision boundary.

- Model complexity.

- Training performance.

- Generalization performance.

The project also includes tuning $C$ as an optional bonus task.

# SVM Implementation

Each of the following models is implemented:

1.  Hard-Margin SVM

2.  Soft-Margin SVM

3.  Kernel SVM

For each model, two implementations are considered.

## CVXPY Implementation

The corresponding convex optimization problem is formulated directly
using `CVXPY`.

## Scikit-Learn Implementation

The same models are implemented using established machine-learning tools
from `scikit-learn`.

The two approaches are compared to investigate the relationship between
the mathematical optimization formulation and standard machine-learning
implementations.

# SVM Evaluation

The SVM models are evaluated using the following classification metrics.

## Accuracy

$$\operatorname{Accuracy}
=
\frac{TP+TN}
{TP+TN+FP+FN}.$$

## Precision

$$\operatorname{Precision}
=
\frac{TP}
{TP+FP}.$$

## Recall

$$\operatorname{Recall}
=
\frac{TP}
{TP+FN}.$$

## F1-Score

$$F_1
=
2
\frac{
\operatorname{Precision}
\cdot
\operatorname{Recall}
}{
\operatorname{Precision}
+
\operatorname{Recall}
}.$$

A Logistic Regression model is also trained on the same dataset and used
as a baseline for comparison.

# Technologies

The project is implemented in Python using the following tools and
libraries:

- Python

- NumPy

- Pandas

- Matplotlib

- CVXPY

- Statsmodels

- Scikit-Learn

# Project Objectives

The main objectives of this project are:

1.  Formulate statistical learning problems as optimization problems.

2.  Derive likelihood-based objective functions.

3.  Analyze convexity of optimization objectives.

4.  Implement numerical optimization algorithms manually.

5.  Solve convex optimization problems using CVXPY.

6.  Compare custom implementations with established libraries.

7.  Evaluate models using appropriate statistical and machine learning
    metrics.

8.  Investigate the effect of regularization on SVM performance.

# Repository Structure

A possible repository organization is:

    Nonlinear-Programming/
    |
    |-- README.md
    |
    |-- data/
    |   |-- gamma.csv
    |   `-- svm.csv
    |
    |-- notebooks/
    |   |-- gamma_regression.ipynb
    |   `-- svm.ipynb
    |
    |-- src/
    |   |-- gamma_regression.py
    |   |-- gradient_descent.py
    |   |-- newton_raphson.py
    |   |-- hard_margin_svm.py
    |   |-- soft_margin_svm.py
    |   `-- kernel_svm.py
    |
    |-- results/
    |   |-- figures/
    |   `-- tables/
    |
    `-- requirements.txt

# Results

The final experimental results compare the different optimization
approaches according to their predictive performance and computational
efficiency.

For Gamma Regression, the main comparison is between:

$$\text{CVXPY}
\quad\text{vs.}\quad
\text{Gradient Descent}
\quad\text{vs.}\quad
\text{Newton--Raphson}
\quad\text{vs.}\quad
\text{Statsmodels}.$$

The methods are evaluated using:

$$\text{Adjusted }R^2,
\qquad
\text{RMSE},
\qquad
\text{Computation Time}.$$

For SVM classification, the main comparison is between:

$$\text{Hard-Margin SVM},
\quad
\text{Soft-Margin SVM},
\quad
\text{Kernel SVM},
\quad
\text{Logistic Regression}.$$

These models are evaluated using:

$$\text{Accuracy},
\qquad
\text{Precision},
\qquad
\text{Recall},
\qquad
F_1\text{-Score}.$$

The actual numerical results, plots, convergence behavior, and
comparative analysis are provided in the corresponding notebooks and
result files.

# Academic Context

- **Course:** Nonlinear Programming

- **Project:** NLPCA2

- **Faculty:** Faculty of Mathematics, Statistics, and Computer Science

- **Author:** Mohammad Shahinfar

# Author {#author .unnumbered}

**Mohammad Shahinfar**

Statistics -- Data Science

Areas of interest:

- Statistical Modeling

- Data Science

- Machine Learning

- Mathematical Optimization

- Numerical Methods

# Note {#note .unnumbered}

This repository contains an academic implementation of statistical and
machine-learning optimization methods. The primary emphasis is on
understanding the mathematical formulation, numerical optimization, and
empirical comparison of the methods.
