---
layout: post
title: Latent Dirichlet Allocation
---

Latent Dirichlet allocation (LDA) is a generative probabilistic model for discrete data. The objective is to find a lower dimensionality representation of the data while preserving the salient statistical structure - a complex way to describe clustering. It is commonly used in NLP applications as a topic model, where we are interested in discovering common topics in a set of documents. 

## 1. Probability Distributions in LDA

First let us review some relevant probability distributions.

### 1.1 Poisson

Recall the Poisson distribution. It is a discrete distribution that models the probability of an event occuring $k$ times in an interval, assuming that they occur at a constant rate and are independent (the occurrence of one event does not effect the probability of another occurring).

The PMF is given by:

$$p(k) = e^{-\lambda}\frac{\lambda^k}{k!}$$

where $\lambda$ is the average number of events per interval. Notice that the length of the interval is baked into $\lambda$. We can explicitly define the interval length $t$ if we know the rate of events per unit interval $r$, and thus $\lambda=rt$.$\lambda$ is the expected value and the variance of the Poisson distribution.

### 1.2 Multinomial

Next we have the multinomial distribution, a generalization of the binomial distribution. Whereas the binomial distribution can be viewed as a sequence of Bernoulli trials, i.e. coin flips, the multinomial can be viewed as a sequence of categorical trials, i.e. $k$-sided dice rolls.

Given $k$ categories and $n$ trials, with the probabilities $p_i$ of a trial resulting in category $i$ of $k$, and $X_i$ the number of trials resulting in category $i$, the PMF is given by:

$$p(X_1=x_1, ..., X_k=x_k) = \frac{n!}{\prod_i^k x_i!}\prod_i^k p_i^{x_i}$$

where $\sum_i^k x_i = n$. This can also be expressed with gamma functions:

$$p(X_1=x_1, ..., X_k=x_k) = \frac{\Gamma(n+1)}{\prod_i^k \Gamma(x_i+1)}\prod_i^k p_i^{x_i}$$

Recall the gamma function is defined for positive integers $n$ as: 

$$\Gamma(n) = (n-1)!$$

and otherwise as:

$$\Gamma(z) = \int_0^{\infty} x^{z-1}e^{-x}dx$$

The expected number of trials with outcome $i$ and the variance are:

$$E(X_i) = np_i$$

$$Var(X_i) = np_i(1-p_i)$$

$$Cov(X_i, X_j) = -np_ip_j$$

### 1.3 Dirichlet

Finally, the Dirichlet distribution is a multivariate generalization of the beta distribution. A $k$-dimensional Dirichlet random variable $\theta$ takes values in the $(k-1)$-simplex ($\theta$ lies in the $(k-1)$-simplex if $\theta_i\geq 0, \sum_{i=1}^k\theta_i = 1$) with the following pdf:

$$p(\theta|\alpha) = \frac{1}{B(\alpha)}\prod_{i=1}^k\theta_i^{\alpha_i-1}$$

$$p(\theta|\alpha) = \frac{\Gamma(\sum_{i=1}^k\alpha_i)}{\prod_{i=1}^k\Gamma(\alpha_i)}\prod_{i=1}^k\theta_i^{\alpha_i-1}$$

where $\alpha$ is a $k$-dimensional parameter with positive elements. Note this looks very similar to the form of the multinomial! Recall the beta is the conjugate prior to the binomial; similarly the Dirichlet is the conjugate prior to the multinomial. So if our likelihood is multinomial with a Dirichlet prior, the posterior is also Dirichlet! This gives us some intuition for $\alpha_i$ as the prior proportion of the $i$th class.

The expected value and variance:

$$E(X_j) = \frac{\alpha_j}{\sum_{i=1}^k \alpha_i} = \frac{\alpha_j}{\alpha_0}$$

$$Var(X_j) = \frac{\alpha_j(\alpha_0 - \alpha_j)}{\alpha_0^2(\alpha_0+1)}$$

## 2. LDA

LDA is a three-level Bayesian model where every item is modeled as a finite mixture over a set of latent topics, and the topics are in turn modeled as infinite mixtures over underlying topic probabilities.

It is helpful to discuss LDA in the context of NLP, in terms of guiding intuition. Do note that LDA can absolutely be applied to other problem spaces. We define:
1. Word: basic unit of discrete data. Given a vocabulary of size $V$, the $v$th word can be represented by a one-hot vector with a 1 in the $v$th index and 0 in all others.
2. Document: sequence of $N$ words $w = (w_1, ..., w_N)$
3. Corpus: collection of $M$ documents $D = {w_1, ... w_M}$

### 2.1 Formulating LDA

We want to find a probabilistic model for our corpus that assigns high probability to the members $d_1, ..., d_M$ but also similar documents outside of our training set.

We will represent a document as a random mixture over latent topic variables, and the topics will themselves be described using distributions over words. That is, every document can consist of many topics, and words within the document belong to topics according to the distribution of words conditional on topics.

LDA models the generative process for each document in the corpus as:

1. $N \sim \text{Poisson}(\xi)$
2. $\theta \sim \text{Dir}(\alpha)$
3. For each of the $N$ words $w_n$:
    1. Sample a topic $z_n \sim \text{Multinomial}(\theta)$
    2. Sample a word $w_n$ from a multinomial conditioned on $z_n$, $w_n \sim \text{Multinomial(\beta_{z_n})}$, where each topic corresponds to its own $\beta_{z_n}$. This can also be written as $p(w_n \| z_n, \beta)$ where $\beta$ is a $k\times V$ matrix with $\beta_{ij} = p(w^j=1\|z^i=1)$.

Note that we have predetermined and fixed $k$. Secondly, note that the Poisson assumption is not relevant to the rest of the process and can be discarded for more realistic document length assumptions.

Given the parameters $\alpha, \beta$, the joint distribution of the topic mixture $\theta$, the latent topic variables $z$, and the set of $N$ words $w$:

$$p(\theta, z, w|\alpha, \beta) = p(\theta|\alpha)\prod_{n=1}^Np(z_n|\theta)p(w_n|z_n,\beta)$$

We can marginalize out the latent variables $\theta, z$ to get the marginal distribution of a document:

$$p(w|\alpha, \beta) = \int p(\theta|\alpha)(\prod_{n=1}^N\sum_{z_n} p(z_n|\theta)p(w_n|z_n, \beta))d\theta$$

Then the probability of a corpus - $p(D\|\alpha, \beta)$, the likelihood - is the product of these marginal probabilities over all documents.

### 2.2 Comparisons

Take careful note of the hierarchy. LDA is not simply Dirichlet-multinomial clustering! There, we would sample a Dirichlet once for the corpus to describe the topic mixture, then sample a multinomial once for each document to describe a topic, then select words conditional on the topic (clustering)variable. A document would be associated with a single topic. In LDA, topics are repeatedly sampled within each document.

Indeed, what we have just described is the mixture of unigrams model. Each document is generated by choosing a topic $z$, then generating $N$ words from the multinomial $p(w\|z)$:

$$p(w) = \sum_z p(z)\prod_{n=1}^Np(w_n|z)$$

A unigram model would be even more simplistic, with the words of every document being drawn independently from one multinomial:

$$p(w) = \prod_{n=1}^Np(w_n)$$

### 2.3 Estimation, EM, and Variational Inference

We need to find the parameters $\alpha, \beta$ that maximize the likelihood of the data, marginalizing over latent $\theta, z$. Of course, whenever we have to find maximum-likelihood solutions for models with latent variables, we turn to EM. At a high level:

1. E-step: compute the posterior of the latent variables $p(\theta, z\|w, \alpha, \beta)$ given the document $w$ and the parameters

    $$p(\theta, z|w, \alpha, \beta) = \frac{p(\theta, z, w|\alpha, \beta)}{p(w|\alpha, \beta)}$$


2. M-step: estimate $\alpha, \beta$ given the revised latent variable estimates

In practice, the posterior of the latent variables is analytically intractable in general. In such situations we turn to approximations to the posterior distribution. The most common method, and the one discussed in the original LDA paper, is variational inference. We will leave the details of variational EM for a future time.


