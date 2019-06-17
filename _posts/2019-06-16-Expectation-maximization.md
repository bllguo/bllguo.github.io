---
layout: post
title: Expectation Maximization
---

In the last two posts we have seen two examples of the expectation maximization (EM) algorithm at work, finding maximum likelihood solutions for models with latent variables. Now we derive EM for general models, and demonstrate how it maximizes the log-likelihood.

## 1. Decomposing the log-likelihood

Suppose we have a model with variables $X$ and latent variables $Z$, with a joint distribution described by parameters $\theta$. The likelihood function is then:

$$p(X|\theta) = \sum_Z p(X, Z|\theta)$$

The key assumption we make is that optimizing $p(X\|\theta)$ is complex relative to optimizing $p(X, Z\|\theta)$.

To optimize over the latent variables we will need a distribution $q(Z)$ over the latent variables. Now we can represent the log likelihood $\ln p(X\|\theta)$ as follows:

$$\ln p(X|\theta) = \sum_Z q(Z)\ln p(X|\theta)$$

By the rules of probability we know $p(X,Z\|\theta) = p(Z\|X, \theta)p(X\|\theta)$, hence:

$$\ln p(X|\theta) = \sum_Z q(Z)\ln \frac{p(X,Z|\theta)}{p(Z|X, \theta)}$$

$$\ln p(X|\theta) = \sum_Z q(Z)(\ln \frac{p(X,Z|\theta)}{q(Z)} - \ln\frac{p(Z|X, \theta)}{q(Z)})$$

$$\ln p(X|\theta) = \sum_Z q(Z)\ln \frac{p(X,Z|\theta)}{q(Z)} - \sum_Z q(Z)\ln\frac{p(Z|X, \theta)}{q(Z)}$$

$$\ln p(X|\theta) = \mathcal{L}(q,\theta) + KL(q\|p)$$

Notice the second term is the KL-divergence between the prior $q(Z)$ and the posterior $p(Z\|X, \theta)$! Furthermore, as KL-divergence is $\geq 0$, it follows that $\mathcal{L}(q,\theta) \leq \ln p(X\|\theta)$. It is a lower bound on the log-likelihood.

## 2. EM Algorithm

Now, let our current parameters be $\theta_0$. 

### 2.1 E-step

Consider taking the partial of the log likelihood $\ln p(X\|\theta_0)$ with respect to $q(Z)$, keeping $\theta_0$ fixed. Notice that $\ln p(X\|\theta_0)$ does not depend on $q(Z)$ as it is marginalized out! However, when $q(Z) = p(Z\|X,\theta_0)$ - setting the $q$ distribution to the posterior for the current $\theta_0$ - the KL-divergence term goes to 0. Then we have $\mathcal{L}(q,\theta_0) = \ln p(X\|\theta_0)$.

Now fix $q(Z)$ and take the partial w.r.t $\theta$ to maximize $\mathcal{L}(q,\theta)$ and find $\theta_1$. $\mathcal{L}$, the lower bound on the log-likelihood, will necessarily increase. 

### 2.2 M-step

Remember that we fixed $q(Z) = p(Z\|X, \theta_0)$. Substituting in $\mathcal{L}$:

$$\mathcal{L}(q,\theta) = \sum_Z p(Z|X, \theta_0)\ln p(X, Z|\theta) - \sum_z p(Z|X, \theta_0)\ln p(Z|X, \theta_0)$$

$$\mathcal{L}(q,\theta) = \sum_Z p(Z|X, \theta_0)\ln p(X, Z|\theta) + H(q(Z))$$

Where $H(q(Z))$ is the entropy of the $q$ distribution. The key is that this quantity is independent of $\theta$. So really what we are maximizing is $\sum_Z p(Z\|X, \theta_0)\ln p(X, Z\|\theta) = Q(\theta, \theta_0)$ - the expected value of the complete-data ($X$ and $Z$) log-likelihood. Another way to think about this is a weighted MLE with the weights given by $q(Z)$.

Also note that we fixed $q(Z) = p(Z\|X, \theta_0)$ instead of $p(Z\|x, \theta_1)$, so the KL-divergence term is positive (unless $\theta_0 = \theta_1$). Hence the log-likelihood actually increases by more than the increase in $\mathcal{L}$. 

### 2.3 Summary

The algorithm can be expressed beautifully as two simple steps - iteratively computing the posterior over the latent variables $p(Z\|X, \theta)$, and then using the posterior to update the parameters $\theta$ by maximizing the expected full-data log-likelihood.

Through the decomposition into $\ln p(X\|\theta) = \mathcal{L}(q,\theta) + KL(q\|\|p)$ we also have shown that the EM algorithm continually improves the lower-bound on the log-likelihood as well as the log-likelihood itself.