---
layout: post
title: Principal Component Analysis
---

Principal component analysis (PCA) is a widely used technique for dimensionality reduction. This in turn leads to practical applications such as compression and data visualization, problems that can be reduced to dimensionality reduction at their core. 

Consider a simple example. I have several books on my desk. We can collect all kinds of data for each book - their topic, their content, how many pages there are, their physical thickness, their appearance, etc. But many of these are correlated properties, such as thickness and page number. In some sense, these two features are telling us very similar things, which is redundant. Theoretically we can construct a set of meta-properties that will describe these books in a more efficient manner, known as the principal components. For instance, one principal component might capture book length and synthesize information from the thickness, page number, word count, etc. into one measure.

How can we formalize this intuition? Below I will discuss two equivalent formulations of PCA:
1. maximizing variance of the projected data
2. minimizing mean-squared error of the projected data

## 1. Maximum variance

Suppose we have $n$ observations of data $x$, with dimensionality $D$. Our goal in PCA is to reduce this data by projecting $X$ onto a $M$-dimensional subspace where $M < D$. In this perspective, we will do this by maximizing the variance of the projected data. This makes sense intuitively; the most descriptive features for the data will have high variance. Think about the opposite - having a feature for our set of books that is low variance. Then it really does not help us distinguish one book from another. 

Consider a one-dimensional space with direction given by $D$-dimensional vector $u_1$. Since we only care about direction, we can define $u_1$ as a unit vector ($u_1^Tu_1 = 0$). The projection of observation $x_i$ onto $u_1$ is given as $\frac{u_1^Tx_i}{u_1^Tu_1}u_1$ - the coordinate in $D$-space. The distance from the origin is simply the scalar value $\frac{u_1^Tx_i}{u_1^Tu_1}$. The denominator simplifies to 1 since $u_1$ is a unit vector. Hence each point $x_i$ gets projected to $u_1^Tx_i$.

The mean of the projected data is

$$\bar{x} = \frac{1}{n}\sum_n x_i$$

Hence the variance of the projected data is

$$\sigma^2 = \frac{1}{n}\sum_n(u_1^Tx_i-u_1^T\bar{x})^2$$

$$\sigma^2 = u_1^TSu_1$$

$$S = \frac{1}{n}\sum_n(x_i-\bar{x}(x_i-\bar{x})^T)$$

Where $S$ is the covariance matrix. Maximize the projected variance $u_1^TSu_1$ with respect to $u_1$, subject to $u_1^Tu_1=1$:

$$L = u_1^TSu_1 + \lambda_1(1-u_1^Tu_1)$$

Where $L$ is the Lagrangian. Taking $\frac{\partial L}{\partial u_1}$, we will arrive at a familiar equation:

$$Su_1 = \lambda_1u_1$$

That is, $u_1$ is an eigenvector of the covariance matrix $S$! Furthermore:

$$u_1^TSu_1 = \lambda_1u_1^Tu_1$$

$$u_1^TSu_1 = \lambda_1$$

The variance is given by the eigenvalue, so the eigenvector with the largest eigenvalue will maximize the variance. So $u_1$ is the first principal component. 

Generalizing to $M$-space, the variance-maximizing projection will be defined by the $M$ largest eigenvectors of the covariance matrix $S$.

### 1.1 Matrix version

This can be expressed in a simpler way by starting with matrices. Express our data as a $n\times D$ matrix $X$, but after centering by subtracting the mean from each row such that $\bar{X} = 0$. Recalling our projection result above, all the projections will be given from $Xu$, which will be $n\times M$. Apply the variance definition:

$$\sigma^2_u = E((Xu - \bar{Xu})^2)$$

$$\sigma^2_u = E((Xu)^2)$$

$$\sigma^2_u = \frac{1}{n}(Xu)^T(Xu)$$ 

$$\sigma^2_u = \frac{1}{n}u^TX^TXu$$ 

$$\sigma^2_u = u^T\frac{X^TX}{n}u$$ 

$$\sigma^2_u = u^TSu$$

Since after mean-centering, $S = \frac{X^TX}{n}$! So we have arrived at the same result. Now we can again use the Lagrangian and so on.

## 2. Minimizing error

Here the idea is to minimize reconstruction error. We look for a projection onto $M$-space that minimizes the average distance between the original data vectors and their projections. We shall see that this is equivalent to maximizing the variance!

Consider again an individual observation $x_i$ and its projection onto a 1D subspace $u_1$. To simplify the math, we will center the data by subtracting the mean. 

Recall the projection of observation $x_i$ onto $u_1$ is given as $\frac{u_1^Tx_i}{u_1^Tu_1}u_1$. Keeping in mind $u_1^Tu_1=1$, we will find that the squared distance between $x_i$ and the projection is:

$$||x_i-(u_1^Tx_i)u_1||^2 = x_i^Tx_i - (u_1^Tx_i)^2$$

Summing to get the mean-squared error across all $x_i$:

$$MSE(u_1) = \frac{1}{n}\sum_n x_i^Tx_i - (u_1^Tx_i)^2$$

The first term does not depend on $u_1$. Hence to minimize the MSE we simply need to maximize

$$\frac{1}{n}\sum_n (u_1^Tx_i)^2$$

This is the sample mean of $u_1^Tx_i$. Remember that $\text{var}(x) = E(x^2) - E(x)^2$:

$$\frac{1}{n}\sum_n (u_1^Tx_i)^2 = E(u_1^Tx_i)^2 + \text{var}(u_1^Tx_i)$$

Thanks to centering, the first term is 0. So we have shown that we are simply maximizing the variance again!