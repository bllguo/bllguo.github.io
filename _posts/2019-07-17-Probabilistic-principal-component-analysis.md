---
layout: post
title: Probabilistic PCA
---

Today let's return to [principal component analysis](https://bllguo.github.io/Principal-component-analysis/) (PCA). Previously we had seen how PCA can be expressed as a variance-maximizing projection of the data onto a lower-dimension space, and how that was equivalent to a reconstruction-error-minimizing projection. Now we will show how PCA is _also_ a maximum likelihood solution to a continuous latent variable model, which provides us several useful benefits, some of which include:
1. Solvable using [expectation-maximization](https://bllguo.github.io/Expectation-maximization/) (EM) in an efficient manner, where we avoid having to explicit construct the covariance matrix. Of course, we can also alleviate this issue in regular PCA by using singular value decomposition (SVD).
2. Handles missing values in the dataset
3. Permits a Bayesian treatment of PCA
4. Allows comparisons with probabilistic density modeling techniques

## 1. Setup

Suppose we have $n$ observations of data $x$, with dimensionality $D$. Recall that our goal in PCA is to reduce this data by projecting $X$ onto a $M$-dimensional subspace where $M < D$.

First let's introduce a latent variable $z$ for the lower-dimensional principal component subspace. We assume a Gaussian prior distribution for $z$, $p(z)$ with zero mean and unit covariance.

$$p(z) = N(z|0, I)$$

Let us also choose the conditional distribution $p(x\|z)$ to be Gaussian:

$$p(x|z) = N(x|Wz + \mu, \sigma^2I)$$

Here the mean is described as a linear function of $z$ with a $D\times M$ matrix $W$ and $D$-dimensional $\mu$. 

From a generative viewpoint, we can describe our $D$-dimensional data as a linear transformation of the $M$-dimensional latent $z$ with some Gaussian noise:

$$x = Wz + \mu + \epsilon$$

where $\epsilon$ is $D$-dimensional Gaussian noise with zero mean and covariance $\sigma^2I$. So generating $x$ can be done by fixing some $z$ and then sampling from the distribution conditioned on $z$. 

## 2. Specifying the likelihood and prior

Now we want to determine the parameters $W, \mu, \sigma^2$ using maximum likelihood. We need to specify the likelihood function $p(x\|W, \mu, \sigma^2)$ first.

$$p(x) = \int p(x|z)p(z)dz$$

By the properties of the Gaussian and of linear-Gaussian models, this marginal distribution $p(x)$ is also Gaussian:

$$p(x) = N(x|\mu, C)$$

$$C = WW^T + \sigma^2I$$

where $C$ is the $D\times D$ covariance matrix.

This can be noticed from $x = Wz + \mu + \epsilon$:

$$E(x) = E(Wz+\mu+\epsilon) = \mu$$

$$cov(x) = E((Wz+\epsilon)(Wz+\epsilon)^T)$$

$$cov(x) = E(Wzz^TW^T) + E(\epsilon\epsilon^T) = WW^T + \sigma^2I$$

Remember $z$ was defined to have zero mean and unit variance, and that $\epsilon$ was defined to have covariance $\sigma^2I$. The cross terms vanish because $z, \epsilon$ are uncorrelated.

Next we compute the posterior $p(z\|x)$. We will be handwavy about the mathematical details here, as we have not yet covered linear-Gaussian models, but nevertheless the final result is presented for completeness. (In any case the mathematics here is not the critical point.)

It can be shown that 

$$C^{-1} = \sigma^{-2}I - \sigma^{-2}WM^{-1}W^T$$

where

$$M = W^TW + \sigma^2I$$

which is $M\times M$.

From here we can use linear-Gaussian results:

$$p(z|x) = N(z|M^{-1}W^T(x-\mu), \sigma^2M^{-1})$$

## 3. Maximum likelihood solution

Now we can use maximum likelihood to obtain the parameters. We will skip over the derivations, but the parameters all have closed form solutions:

$$\mu_{\text{MLE}} = \bar{x}$$

$$W_{\text{MLE}} = U_M(L_M - \sigma^2I)^{1/2}R$$

$$\sigma^2_{\text{MLE}} = \frac{1}{D-M}\sigma_{i=M+1}^D \lambda_i$$

### 3.1 $W_{\text{MLE}}$

Let $S$ be the covariance matrix of the data $x$:

$$S = \frac{1}{n}\sum_n(x_i-\bar{x}(x_i-\bar{x})^T)$$

Then $U_M$ is a $D\times M$ matrix whose columns are the eigenvectors of $S$. 

$L_M$ is $M\times M$ and contains the corresponding eigenvalues. 

$R$ is an arbitrary $M\times M$ orthogonal (orthonormal) matrix. This can be confusing - just substitute in $I$ if so, which is perfectly valid! $R$ can be thought of as a rotation matrix in latent $M$-space. If we substitute $W_{\text{MLE}}$ into $C$, it can be shown that $C$ is independent of $R$. The point is that the predictive density is unaffected by rotations in latent space.

### 3.2 $\sigma^2_{\text{MLE}}$

We have assumed that the eigenvectors were arranged in decreasing order of the eigenvalues. Then $W$ is the principal subspace, and $\sigma^2_{\text{MLE}}$ is just the average variance of the remaining components.

## 4. $M$-space and $D$-space

Probabilistic PCA can be thought of as mapping points in $D$-space to $M$-space, and vice-versa. We can show that:

$$E(z|x) = M^{-1}W^T_{\text{MLE}}(x-\bar{x})$$

Which is $x$'s posterior mean in latent space. Of course, in data space this is:

$$WE(z|x) + \mu$$

The posterior covariance in latent space is: 

$$\sigma^2M^{-1}$$

which is actually independent of $x$.

Notice that as $\sigma^2 \rightarrow 0$, the posterior mean simplifies to a familiar result:

$$(W^T_{\text{MLE}}W_{\text{MLE}})^{-1}W^T_{\text{MLE}}(x-\bar x)$$

which is the orthogonal projection of $x$ onto $W$ - the standard PCA result!

## 5. EM for PCA

We have just shown that probabilistic PCA can be solved using maximum likelihood with closed-form solutions. Why, then, would we want to use EM? 

Typical PCA requires us to evaluate the covariance matrix, which is $O(ND^2)$. In EM, we do not need the explicit covariance matrix. The costliest operations are sums over the data, which are $O(NDM)$. If $M \ll D$ then EM can be much more efficient despite having to go through multiple EM cycles.

Second, if memory is a concern, or if the data is being streamed, the iterative nature of EM is useful as it can be used in an online learning fashion.

Finally, thanks to our probabilistic model, we can deal with missing data in a principled way. We can marginalize over the distribution of the missing data in a generalized $E$-step ([Chen et al., 2009](https://core.ac.uk/download/pdf/397806.pdf)).

## 6. Bayesian PCA

Probabilistic PCA also allows us to take a Bayesian approach for selecting $M$. One example is by choosing a prior over $W$ that performs selection on relevant dimensions. This is an example of automatic relevance determination (ARD). We define a Gaussian prior for each component, or column of $W$, with respective precision hyperparameters $\alpha_i$:

$$p(W|\alpha) = \prod_{i=1}^M (\frac{\alpha_i}{2\pi})^{D/2}\exp{(-\frac{1}{2}\alpha_iw_i^Tw_i)}$$

During the EM process, by maximizing the marginal likelihood after marginalizing out $W$, we can find the optimal $\alpha_i$. Some $\alpha_i$ will be driven to infinity, consequently pushing $w_i$ to 0. Then the finite $\alpha_i$ determine the relevant components $w_i$!

We can also do a fully Bayesian treatment with priors over all parameters $\mu, \sigma^2, W$.