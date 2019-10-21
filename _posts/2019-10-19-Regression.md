---
layout: post
title: Regression
---

Let's do a quick review of basic regression, to lay the framework for future posts. The goal of regression, one of the principal problems in supervised learning, is to predict the value(s) of one or more _continuous_ target variables $y$, given a corresponding vector of input variables $x$.

## 1. Setup

Suppose we have a training dataset consisting of $n$ observations. The $i$-th observation is a ${x_i, y_i}$ pair, where $x_i$ is a $D$-dimensional vector of input variables and $y_i$ the target variable (we shall assume one target, going forward). We would like to be able to predict $y$ when given some new value of $x$. 

The simple linear model involves a linear combination of the input variables $x$. Consider a single observation $x$; its vector components denoted $x_1, x_2, ..., x_D$. Then we can imagine a function:

$$y(x, w) = w_0 + w_1x_1 + ... + w_Dx_D$$

It takes in our inputs $x$ and a set of parameters $w$, and spits out a predicted target value. This is the simple linear regression model. Linear, in the sense that it is a linear combination of the _parameters_ $w$! While it also happens to be linear in the data $x$, do not be deceived. We can also consider linear combinations of nonlinear functions of the data like so:

$$y(x, w) = w_0 + \sum_{j=1}^{M-1} w_j\phi_j(x)$$

where the $\phi_j$ _basis functions_. Notice the indices - there are $M$ parameters in total, one for each $w$ (including $w_0$), and most importantly $M$ does not necessarily equal $D$. In practical terms, we can think of these basis functions as feature transformations on our original data $x$. The key is that by using nonlinear basis functions, we can turn $y(x, w)$ into a nonlinear function w.r.t the data $x$. A simple example is polynomial regression - consider an input $x$ and the basis functions as successive powers, s.t. $\phi_j(x) = x^j$. For our discussion, however, the choice of basis function has no effect on our analysis. We will choose the trivial identity function $\phi(x) = x$ and omit $\phi$ to simplify the notation.

## 2. Least Squares

Let us assume that our target data is of the form:

$$y_i = w^Tx_i + \epsilon$$

where $\epsilon \sim N(0, \sigma^2)$. That is, the target has mean given by $y(x,w) = w^Tx$ but is normally distributed around that value with a given variance $\sigma^2$:

$$p(y_i|x, w, \sigma^2) = N(y_i|w^Tx_i, \sigma^2)$$

Then, across all observations, the likelihood function $P(D\|W)$ becomes:

$$p(y|x, w, \sigma^2) = \prod_{i=1}^{N} N(y_i| w^Tx_i, \sigma^2)$$

Now we can do maximum likelihood estimation to obtain the parameters $w$ and $\sigma^2$:

$$ln(p(y|x, w, \sigma^2)) = \sum_{i=1}^{N} ln(N(y_i| w^Tx_i, \sigma^2))$$

$$ln(p(y|x, w, \sigma^2)) = \frac{N}{2}ln\frac{1}{\sigma^2} - \frac{N}{2}ln(2\pi) - \frac{1}{2\sigma^2}E(w)$$

where:

$$E(w) = \sum_{i=1}^{N} (y-w^Tx)^2$$

which is the sum-of-squares error function.

### 2.1 Solving for $w$

Consider the maximization w.r.t $w$. Notice the first two terms in the log-likelihood are constant. So the MLE solution for $w$ reduces to minimizing the residual squared error.

Taking the gradient of the log-likelihood w.r.t $w$ and setting it equal to 0, we can solve for $w_{MLE}$:

$$w_{MLE} = (x^Tx)^{-1}x^Ty$$

This should be a familiar result! Wrack your brain for a moment - we will return to this shortly.

Notice that we must compute $(x^Tx)^{-1}$. This can be very tricky in practice when the matrix is singular or close to singular. 

Here we can gain a greater understanding of the bias parameter $w_0$. We can make the bias parameter $w_0$ explicit and do MLE to see that:

$$w_{0, MLE} = \bar{y} - \sum w_i\bar{x_i}$$ 

That is, it addresses the difference between the sample mean of the target, and the weighted sum of the sample means of the data. In other words - it is compensating for the bias, hence the name. More on bias later.

### 2.2 Solving for $\sigma^2$

Consider the maximization w.r.t $\sigma^2$. Taking the gradient of the log-likelihood w.r.t $\sigma^2$ and setting it equal to 0, we can solve for $\sigma^2_{MLE}$:

$$\sigma^2_{MLE} = \frac{1}{N}\sum_{i=1}^N (y_i-w_{MLE}^Tx_i)^2$$

This is actually the variance of the true target values around the regression values from the regression model. 

### 2.3 Geometric interpretation

Have you seen it? $w_{MLE}$ is actually the projection of $y$ onto $x$! 

Let us take a perspective informed by linear algebra. Consider the classic linear algebra problem - solving $Ax=b$. In this case, $Xw=y$.

Sometimes, this cannot be solved. When $X$ has more rows than columns ($m > n$ where $m$ is the number of rows, $n$ the number of columns) there are more equations than unknowns. Make the connection between $X$ and the real world meaning - our dataset. This translates to: more observations than variables. Then there are infinitely many solutions, as the $n$ columns span a small part of $m$-space, and hence, $y$ is probably outside of the column space of $X$. 

Framed differently, $e = b - Ax = y - Xw$, the error, is not always 0. If it were, we can obtain an exact solution for $w$. But in the case of $m > n$, $e$ cannot be taken to 0. But we can still find an approximate solution for $w$. One way to determine the optimal $\hat{w}$ that minimizes $e$ is the least squares solution, which minimizes Euclidean distance. Multiply both sides by $X^T$:

$$X^TX\hat{w} = X^Ty$$

The closest point to $y$ in $Xw$ is the projection of $y$ onto $X$. 

So we have shown that this is equivalent to the MLE result when assuming Gaussian noise!

## 3. Regularized least-squares

We can control overfitting by introducing a regularization term to the error function. Recall our data-dependent error for $w$:

$$E_D(w) = \sum_{i=1}^{N} (y-w^Tx)^2$$

We can add a regularization term and minimize this total error function instead:

$$E(w) = E_D(w) + \lambda E_W(w)$$

From a probabilistic Bayesian point of view, we can interpret this regularization as assuming a prior on the parameters $w$. For example:

A simple choice is a Gaussian. We don't know much a priori - let's say $w_j \sim N(0, \lambda^2)$.

$$\text{arg max}_w log P(w|D,\sigma,\lambda) = \text{arg max}_w (log P(D|w, \sigma) + log P(w|\lambda))$$

The regularization term will be $\frac{1}{2\lambda^2}w^Tw$ - the sum of squares of the weights. This regularizer is known as weight decay as it heavily penalizes high weights due to the squaring. It is also known as ridge regression.

The exact solution can be found in closed form:

$$w_{MAP} = (X^TX + \frac{\sigma^2}{\lambda^2}I)^{-1}X^Ty$$

As $\lambda \rightarrow \infty$, our prior becomes broader, and MAP reduces to MLE! As $n$ increases, the entries of $X^TX$ grow linearly but the prior's effect is fixed, so the effect of the prior also vanishes. Also notice that this avoids the problem of $X^TX$ being noninvertible.

A general regularizer can be used, for which the error looks like:

$$\frac{1}{2}\sum (y-w^Tx)^2 + \frac{\lambda}{2}\sum |w|^q$$

When $q=1$ we have lasso regression, which corresponds to a Laplace prior. It has the property of feature selection, as it can drive the parameters to 0. 

## 4. Bias-variance decomposition of loss

Let's take a step back and engage in a frequentist thought experiment related to model complexity. 

We have a model that takes a dataset $D$ and predicts $y$ given $x$. Call it $h(x, D)$. A linear regression model, for example, would be $h(x, D)=w^T_Dx$.

Using a typical squared error loss function, we can calculate the expected loss on a new observation $(x, y)$:

$$E(L) = E((h(x, D)-y)^2) = \int_x\int_y(h(x, D)-y)^2p(x, y)dxdy$$

Of course, we can't compute this without knowing $p(x, y)$, and we indeed do not know it. However, suppose we had a large number of datasets drawn i.i.d from $p(x, y)$. For any given dataset we can run our algorithm to obtain a prediction function $h(x, D)$. Clearly, different $D$ produce different predictors. The performance of the model as a whole can be evaluated by averaging over this ensemble of all possible datasets.

$$\bar{h}(x) = E_D(h(x, D))$$

We want to know the expected loss (squared error) over all datasets, which will be the best way to evaluate the performance of the algorithm.

$$E(L) = E_{x, y, D}((h(x, D)-y)^2)$$

$$E(L) = \int_D\int_x\int_y(h(x, D)-y)^2p(x, y)p(D)dxdydD$$

Add and subtract $\bar{h}(x)$ in the first term:

$$E_{x, y, D}((h(x, D)-y)^2) = E_{x, y, D}((h(x, D) - \bar{h}(x) + \bar{h}(x) - y)^2)$$

$$E_{x, y, D}((h(x, D)-y)^2) = E_{x, y, D}((h(x, D) - \bar{h}(x))^2 + (\bar{h}(x) - y)^2 + 2(h(x, D) - \bar{h}(x))(\bar{h}(x) - y))$$

But the cross term vanishes since $E_D(h(x, D) - \bar{h}(x))=0$, and the other part doesn't depend on $D$. So ultimately:

$$E_{x, y, D}((h(x, D)-y)^2) = E_{x, y, D}((h(x, D) - \bar{h}(x))^2) + E_{x, y}((\bar{h}(x) - y)^2)$$

The first term is the variance of the model. The second term can be further decomposed. Following the same trick and defining $\bar{y}(x) = E_{y\|x}(y)$ - the average $y$ at every $x$:

$$E_{x, y}((\bar{h}(x) - y)^2) = E_{x, y}((\bar{h}(x) - \bar{y}(x) + \bar{y}(x) - y)^2$$

$$E_{x, y}((\bar{h}(x) - y)^2) = E_{x, y}((\bar{h}(x) - \bar{y}(x))^2 + (\bar{y}(x) - y)^2 + 2(\bar{h}(x) - \bar{y}(x))(\bar{y}(x) - y))$$

Again the cross term vanishes as $E(\bar{y}(x) - y) = 0$.

$$E_{x, y}((\bar{h}(x) - y)^2) = E_{x}((\bar{h}(x) - \bar{y}(x))^2) + E_{x, y}((\bar{y}(x) - y)^2)$$

Which are bias squared and noise terms, respectively.

So the expected loss is variance + bias squared + noise.

$$E_{x, y, D}((h(x, D) - \bar{h}(x))^2) = E_{x, y, D}((h(x, D) - \bar{h}(x))^2) + E_{x}((\bar{h}(x) - \bar{y}(x))^2) + E_{x, y}((\bar{y}(x) - y)^2)$$