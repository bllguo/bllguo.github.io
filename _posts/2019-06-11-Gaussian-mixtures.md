---
layout: post
title: Gaussian Mixtures and EM
---

Continuing from last time, I will discuss Gaussian mixtures as another clustering method. We assume that the data is generated from several Gaussian components with separate parameters, and we would like to assign each observation to its most likely Gaussian parent. It is a more flexible and probabilistic approach to clustering, and will provide another opportunity to discuss expectation-maximization (EM). Lastly, we will see how K-means is a special case of Gaussian mixtures!

## 1. Gaussian Mixtures

Here we will take a probabilistic approach to clustering, where we assume the data is generated from $K$ underlying Gaussian distributions:

$$p(x) = \sum_{k=1}^K \pi_kN(x|\mu_k,\Sigma_k)$$

where $\pi_k$, the mixing coefficient, is the probability that the sample was drawn from the $k$th Gaussian distribution. We can reformulate this model in terms of an explicit latent variable $z$ with a 1-of-K representation. That is, $z$ is $K$-dimensional and $z_k = 1$ when the sample belongs to the $k$th Gaussian. Let us proceed with this to arrive at the joint distribution $p(x, z) = p(z)p(x\|z)$.

The marginal distribution over $z$ can be specified:

$$p(z) = \prod_k \pi_k^{z_k}$$

since

$$p(z_k=1) = \pi_k$$

and $0 \leq \pi_k \leq 1$ and $\sum_k \pi_k=1$.

The conditional for a particular $z$ is:

$$p(x|z_k=1) = N(x|\mu_k, \Sigma_k)$$

Hence using the 1-of-K representation,

$$p(x|z) = \prod_k N(x|\mu_k, \Sigma_k)^{z_k}$$

So the joint distribution is simply:

$$p(x, z) = \prod_k (\pi_kN(x|\mu_k, \Sigma_k))^{z_k}$$

Now the marginal distribution of $x$ can be obtained from the joint:

$$p(x) = \sum_z p(z)p(x|z)$$

$$p(x) = \sum_z \prod_k^K (\pi_kN(x|\mu_k, \Sigma_k))^{z_k}$$

Consider what the possible values of $z$ are. They are $K$-dimensional one-hot vectors. So the summation over $z$ is just over $K$ possibilities. Furthermore, if $z_k=1$, notice that the product simplifies to $\pi_kN(x\|\mu_k, \Sigma_k)$ as $z_{i\neq k} = 0$.

$$p(x) = \sum_j^K \pi_jN(x|\mu_j, \Sigma_j)$$

So we have recovered the initial Gaussian mixture formulation, but now with the latent $z$.

Lastly we can define the conditional $p(z|x)$:

$$p(z_k=1|x) = \frac{p(x|z_k=1)p(z_k=1)}{p(x)}$$

$$p(z_k=1|x) = \frac{\pi_kN(x|\mu_k, \Sigma_k)}{\sum_j^K \pi_jN(x|\mu_j, \Sigma_j)} = \gamma(z_k)$$

So $p(z_k=1\|x) = \gamma(z_k)$ is the posterior probability of $z_k=1$, and $\pi_k$ the prior.

## 2. Maximum likelihood

We have a dataset of observations $x^{(1)}, ..., x^{(N)}$ which we assume are i.i.d. Their corresponding latent variables are $z^{(1)}, ..., z^{(N)}$. We can stack them into matrices $X, Z$ which are $N\times D$ and $N\times K$, respectively.

We know:

$$p(x) = \sum_j^K \pi_jN(x|\mu_j, \Sigma_j)$$

So the likelihood:

$$p(X|\pi, \mu, \Sigma) = \prod_n\sum_k^K \pi_kN(x^{(n)}|\mu_k, \Sigma_k)$$

And the log-likelihood:

$$\ln p(X|\pi, \mu, \Sigma) = \sum_n \ln(\sum_k^K \pi_kN(x^{(n)}|\mu_k, \Sigma_k))$$

### 2.1 $\mu_k$

Taking the partial with respect to $\mu_k$ and carefully stepping through the math, we obtain:

$$\mu_k = \frac{1}{N_k}\sum_n \gamma(z_{nk})x^{(n)}$$

$$N_k = \sum_n \gamma(z_{nk})$$

Notice that $\mu_k$ is the weighted mean of $x$ where the weights are the posterior probabilities that $x^{(n)}$ belongs to Gaussian component $k$. $N_k$ thus can be interpreted as the effective number of points assigned to $k$.

### 2.2 $\Sigma_k$

Taking the partial with respect to $\Sigma_k$ we will arrive at a similar weighted result:

$$\Sigma_k = \frac{1}{N_k}\sum_n \gamma(z_{nk})(x^{(n)} - \mu_k)(x^{(n)} - \mu_k)^T$$

### 2.3 $\pi_k$

Lastly, we take the partial with respect to $\pi_k$. Recall $\sum_k \pi_k = 1$, so we will need a Lagrange multiplier to maximize

$$\ln p(X|\pi, \mu, \Sigma) + \lambda(\sum_k \pi_k - 1)$$

We will arrive at 

$$\pi_k = \frac{N_k}{N}$$

### 2.4 Expectation-maximization for Gaussian mixtures

This is not a closed-form solution thanks to the complicated relationship to $\gamma(z_{nk})$. However we can solve for our parameters using the iterative method of EM!

0. Initialize the parameters and evaluate the initial log-likelihood
1. E-step: Evaluate the posteriors under the current parameters:

    $$\gamma(z_{nk}) = frac{\pi_kN(x^{(n)}|\mu_k, \Sigma_k)}{\sum_j^K \pi_jN(x^{(n)}|\mu_j, \Sigma_j)}$$
2. M-step: Re-evaluate the parameters using the new posteriors:

    $$\mu_k = \frac{1}{N_k}\sum_n \gamma(z_{nk})x^{(n)}$$

    $$\Sigma_k = \frac{1}{N_k}\sum_n \gamma(z_{nk})(x^{(n)} - \mu_k)(x^{(n)} - \mu_k)^T$$

    $$\pi_k = \frac{N_k}{N}$$

    $$N_k = \sum_n \gamma(z_{nk})$$
3. Calculate the log-likelihood and check stopping criteria:

    $$\ln p(X|\pi, \mu, \Sigma) = \sum_n \ln(\sum_k^K \pi_kN(x^{(n)}|\mu_k, \Sigma_k))$$

## 3. Expectation-maximization

In general, EM is used to find MLE solutions when we have latent variables. Consider the general log-likelihood:

$$\ln p(X|\theta) = \ln(\sum_Z p(X, Z|\theta))$$

where we have introduced the latent variable matrix $Z$. 

We cannot compute this in practice, because we do not observe the latent variables $Z$. What we know about $Z$ is captured by our posterior $p(Z\|X, \theta)$. We can use it to compute the expected value of the log-likelihood under this posterior.

Consider our current parameters $\theta_0$ and the new parameters $\theta_1$.

In the E-step, we use $\theta_0$ to find the posterior over the latent variables, $p(Z\|X, \theta_0)$. We can use it to calculate:

$$Q(\theta, \theta_0) = \sum_Z p(Z|X, \theta_0)\ln p(X, Z|\theta)$$

which is the expectation of the log-likelihood for a general $\theta$.

In the M-step, we find $\theta_1$ by maximizing the expectation:

$$\theta_1 = \text{arg max}_{\theta} Q(\theta, \theta_0)$$

## 4. Comparison with K-means

Notice that K-means performs a hard, binary assignment of observations to clusters - a point is either in a cluster or it isn't. In Gaussian mixtures, we make a soft assignment by way of the posterior probabilities. It turns out that K-means is a special case of Gaussian mixtures! 

Let the covariance matrices $\Sigma_k$ all be equal to $\epsilon I$. Then:

$$p(x|\mu_k, \Sigma_k) = \frac{1}{(2\pi\epsilon)^{D/2}}\exp(-\frac{||x-\mu_k||^2}{2\epsilon})$$

by the definition of the multivariate Gaussian.

Treating the variances $\epsilon$ as fixed, we arrive at:

$$\gamma(z_{nk}) = \frac{\pi_k\exp(-\frac{||x^{(n)}-\mu_k||^2}{2\epsilon})}{\sum_j \pi_j\exp(-\frac{||x^{(n)}-\mu_j||^2}{2\epsilon})}$$

As $\epsilon \rightarrow 0$, the term with the smallest value of $\\|x^{(n)}-\mu_l\\|^2$ will approach $0$ slowest. So in the limit, $\gamma(z_{nl}) \rightarrow 1 \rightarrow r_{nl}$ while the other posteriors converge to 0. We have obtained the hard assignment of K-means. Recall:

$$r_{nk} = \begin{cases}
1,  & \text{if }k=\text{arg min}_j ||x_n-\mu_j||^2 \\\
0, & \text{otherwise}
\end{cases}$$

The update equation for $\mu_k$ will simplify to the K-means result thanks to these weights. We can also show that as $\epsilon \rightarrow 0$, the log-likelihood converges to the distortion loss in K-means.

Notice the geometric interpretation here. Because of the way we have specified the $\Sigma_k$ in K-means (as diagonal matrices), the underlying assumption is spherical clusters. Whereas in Gaussian mixtures, we can have elliptical clusters depending on our $\Sigma_k$.