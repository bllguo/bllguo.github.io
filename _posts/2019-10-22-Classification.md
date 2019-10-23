---
layout: post
title: Classification
---

I realize that the order of posts here seems without rhyme or reason. I have no justification to offer. But these posts are better late than never! Here we proceed to lay another block of the foundation by discussing classification, logistic regression, and finally, generalized linear models.

## 1. The Classification Problem

In classification, we wish to assign one of several classes $C_k$ to input data $x$. 

The classification problem can be broken down into two stages, inference and decision. 

1. Inference stage: use the training data to learn the conditional posterior probabilities $p(C_k\|x)$. Here we are considering the priors as $p(C_k)$. Thus, by Bayes' theorem:
    

    $$p(C_k|x) = \frac{p(x|C_k)p(C_k)}{p(x)}$$
2. Decision stage: use these posterior probabilities to assign classes.

There are three general approaches to classification, based on how they approach these stages.

1. Generative


    Determine the class-conditional probability densities $p(x\|C_k)$ for each class, as well as the priors $p(C_k)$. From there, use Bayes' theorem to compute $p(C_k\|x)$:

    $$p(C_k|x) = \frac{p(x|C_k)p(C_k)}{p(x)} = \frac{p(x|C_k)p(C_k)}{\sum_k p(x|C_k)p(C_k)}$$

    It is generative in the sense that we are also modeling the distribution of the inputs $x$, which allows us to generate synthetic $x$. This is demanding since we need to find the joint distribution over $x, C_k$ (the class priors are easy since we can estimate simply from the class proportions in the training set). When $x$ is of high dimensionality, we may need a large dataset to do this. Having the marginal density $p(x)$ can also be useful for outlier detection, where we identify points that are low probability under the model and thus have poor predictions from the model.

2. Discriminative



    Determine the posteriors $p(C_k\|x)$ directly, then use decision theory to assign classes. Discriminative approaches require less computational power, less parameters, and less data, and we still get to have the posteriors $p(C_k\|x)$.

3. Discriminant


    Determine a discriminat function $f(x)$ which directly maps $x$ to a class label. Discriminant approaches lose the probabilistic outputs entirely.

We will be skipping over discriminants in the following discussion.

## 2. The Generative Approach

Let us start with the simplest case, binary classification. Recall we want to model the class-conditional densities $p(x\|C_k)$ as well as the priors $p(C_k)$ in order to compute the posteriors $p(C_k\|x)$ via Bayes' theorem.

$$p(C_1|x) = \frac{p(x|C_1)p(C_1)}{p(x|C_1)p(C_1) + p(x|C_2)p(C_2)}$$

Now generalizing to multiclass:

$$p(C_k|x) = \frac{p(x|C_k)p(C_k)}{\sum_j p(x|C_j)p(C_j)}$$

$$p(C_k|x) = \frac{\exp(a_k)}{\sum_j\exp(a_j)}$$

where $a_k = \ln{p(x\|C_k)p(C_k)}$. The form of the posterior is the softmax function applied to $a_k$.

We can now throw in assumptions about the class-conditional densities $p(x\|C_k)$ and solve for the posteriors. 

For example, suppose all of our features are binary and discrete. If there are $m$ features, and $K$ classes, we have $K \times 2^m$ possible values of $(x, y)$ to enumerate in order to describe the discrete distribution. This is clearly infeasible at high dimensionality. This only gets worse if we relax the binary condition. 

A simplified representation can be attained by assuming conditional independence of the features given the class $C_k$:

$$p(x|C_k) = \prod_j P(x_j|C_k)$$

Then we can compute the posterior class probabilities like so:

$$\text{arg max}_c\text{ }p(C_k=c|x) = P(C_k=c)\prod_j P(x_j|C_k=c)$$

This is called the naive Bayes assumption.

## 3. Discriminative Models and Logistic Regression

In the generative approach, we can use maximum likelihood to estimate the parameters of the class-conditional densities as well as the class priors, under specific assumptions over the class-conditional densities. From there we can apply Bayes' theorem to find the posterior probabilities.

We can also exploit the functional form of our model for the posterior probabilities, and use maximum likelihood to determine its parameters directly, without needing the class-conditionals or priors. Logistic regression is one example. 

### 3.1 Binary

Let's start with the binary case once more.

Recall the general formulation of the posterior:

$$p(C_1|x) = \frac{p(x|C_1)p(C_1)}{p(x|C_1)p(C_1) + p(x|C_2)p(C_2)}$$

Notice that this can be expressed as:

$$p(C_1|x) = \sigma(a)$$

$$a = \ln{\frac{p(x|C_1)p(C_1)}{p(x|C_2)p(C_2)}}$$

$\sigma$ is the sigmoid function:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

So instead of looking at the class-conditional density or the prior density, we roll that all up into the sigmoid. Let:

$$a = w^Tx$$

$$p(C_1|x) = \sigma(a) = \sigma(w^Tx)$$

Then, the likelihood function can be expressed in terms of $p(C_1\|x)$ (let $y \in [0, 1]$):

$$p(y|w) = \prod_n \sigma(w^Tx_n)^{y_n}(1-\sigma(w^Tx_n))^{1-y_n}$$

Why can we assume this linear form for $a$? This is in fact a key point - this assumption implies a deeper assumption, that the target $y$ has a distribution in the exponential family.

Taking the negative logarithm, we get the cross-entropy error function:

$$E(w) = -\log{(p(y|w))} = -\sum_n y_n\log{(\sigma(w^Tx_n))} + (1-y_n)\log{(1-\sigma(w^Tx_n))}$$

Taking the gradient w.r.t $w$:

$$\nabla E(w) = \sum_n (\sigma(w^Tx_n) - y_n)x_n$$

There is no closed-form solution here, but we can simply use gradient descent.

### 3.2 Multiclass

To generalize to multiclass, let:

$$p(C_k|x) = f_k(x) = \frac{\exp(w^T_kx_k)}{\sum_j \exp(w^T_jx_j)}$$

This is the softmax transformation.

We use a 1-of-k coding scheme where $y_n$ for a feature vector $x_n$ is a vector with all elements equal to 0, except for element $k$.

The likelihood takes the form:

$$p(Y|w_1, ..., w_K) = \prod_n\prod_kp(C_k|x_n)^{y_{n, k}} = \prod_n\prod_kf_k(x_n)^{y_{n, k}}$$

$Y$ is a $N\times K$ matrix of target variables.

The negative log gives us the cross-entropy error function:

$$E(w_1, ..., w_K) = -\log{p(Y|w_1, ..., w_K)} = \sum_n\sum_ky_{n, k}\log{f_k(x_n)}$$

Keep in mind the constraint $\sum_ky_{n, k} = 1$ and take the gradient with respect to each $w_k$ to obtain the cross-entropy error function:

$$\nabla_{w_k} E(w_1, ..., w_K) = \sum_{n=1}^N(f_{n,k} - y_{n, k})x_n$$


## 4. Generalized Linear Models

A generalized linear model is a model of the form:

$$y = f(w^T\phi)$$

$f(.)$ is the activation function, and $f^{-1}(.)$ is the link function. In the GLM, $y$ is a nonlinear function of a linear combination of the inputs. Notice that linear and logistic regression satisfy this definition.

The link function describes the association between the mean of the target, $y$, and the linear term $w^T\phi$. It _links_ them in such a way that the range of the transformed mean, $f^{-1}(y)$, is $(-\infty, \infty)$, which allows us to form the linear equation $y = f(w^T\phi)$ and solve using MLE. Take classification as an example - the target is $\in [0, 1]$ and so the untransformed $y$ is in the range $[0, 1]$.

### 4.1 Linear Regression Recap

Now, let's take a quick review of linear regression:

Recall that for a linear regression model with Gaussian noise, the likelihood function is:

$$p(y|x, w, \sigma^2) = \prod_{i=1}^{N} N(y_i| w^Tx_i, \sigma^2)$$

And the log likelihood is:

$$ln(p(y|x, w, \sigma^2)) = \sum_{i=1}^{N} ln(N(y_i| w^Tx_i, \sigma^2))$$

$$ln(p(y|x, w, \sigma^2)) = \frac{N}{2}ln\frac{1}{\sigma^2} - \frac{N}{2}ln(2\pi) - \frac{1}{2\sigma^2}E(w)$$

where:

$$E(w) = \sum_{i=1}^{N} (y-w^Tx)^2$$

which is the sum-of-squares error function.

Take the partial derivative w.r.t $w$:

$$\nabla ln(p(y|x, w, \sigma^2)) = \frac{1}{\sigma^2}\sum_{i=1}^N (y_i - w^Tx_i)x_i^T$$

Have you noticed something? In linear regression, binary logistic regression, _and_ multiclass logistic regression, the partial w.r.t $w$ of the error function takes the form $(y_i - \hat{y}_i)x_i$.

This is no coincidence - this is a consequence of 1. assuming the target has a distribution from the exponential family, and 2. our choice of activation function.

### 4.2 The Exponential Family

The exponential family of distributions over $x$ given parameters $\eta$ is defined as:

$$p(x|\eta) = h(x)g(\eta)exp(\eta^Tu(x))$$

Where $\eta$ are the _natural parameters_ and $u(x)$ is some function of $x$. $g(\eta)$ is a normalizing coefficient that ensures:

$$g(\eta)\int h(x)exp(\eta^Tu(x))dx = 1$$

For simplicity, consider a restricted subclass of exponential family distributions, where $u(x)$ is the identity, and where we have introduced a scale parameter $s$ such that:

$$p(x|\eta, s) = \frac{1}{s}h(\frac{x}{s})g(\eta)exp(\frac{\eta x}{s})$$

### 4.3 Canonical Link Functions

Now, let us assume that our target variable $t$ is in the exponential family. Then,

$$p(t|\eta, s) = \frac{1}{s}h(\frac{t}{s})g(\eta)exp(\frac{\eta t}{s})$$

Consider taking the gradient of both sides with respect to $\eta$ and simplifying. We will arrive at:

$$E(t|\eta) = y = -s\frac{d}{d\eta}\ln g(\eta)$$

Which shows us that $y$ and $\eta$ are related. Let this relationship be $\eta = \psi(y)$.

Now go back and consider the log-likelihood for the model on $t$, assuming all observations have the same scale parameter.

$$\ln p(t|\eta, s) = \sum_{n=1}^N \ln p(t_n|\eta, s) = \sum_{n=1}^N (\ln g(\eta_n) _ \frac{\eta_nt_n}{s}) + \text{const}$$

Take the derivative with respect to $w$:

$$\nabla_w \ln p(t|\eta, s) = \sum_{n=1}{N}(\frac{d}{d\eta_n}\ln g(\eta_n) + \frac{t_n}{s})\frac{d\eta_n}{dy_n}\frac{dy_n}{da_n}\nabla a_n$$

where $a_n = w^T\phi_n$. Now using $y = E(t|\eta)$, $\eta = \psi(y)$, and our earlier GLM definition $y_n = f(a_n)$:

$$\nabla_w \ln p(t|\eta, s) = \sum_{n=1}{N}\frac{1}{s}(t_n-y_n)\psi'(y_n)f'(a_n)\phi_n)$$

If we choose the link function $f^{-1}(y) = \psi(y)$, then $f(\psi(y)) = y$. Consequently:

$$f'(\psi)\psi'(y) = \frac{df}{d\psi}\frac{d\psi}{dy} = \frac{dy}{d\psi}\frac{d\psi}{dy} = 1$$

Notice $a = f^{-1}(y)$, hence $a = \psi$, and therefore from the above:

$$\psi'(y_n)f'(a_n) = 1$$

And the error function has reduced to a familiar form:

$$\nabla E(w) = \frac{1}{s}\sum_{n=1}^N(y_n-t_n)\phi_n$$

The proper choice of link function - the canonical link function - takes our error function to this simple form, and also enforces $\eta = E(t)$. There are a number of other desirable statistical properties associated with the canonical link function.