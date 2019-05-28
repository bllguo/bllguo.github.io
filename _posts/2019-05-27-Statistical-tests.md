---
layout: post
title: Statistical Tests as Linear Models
---

These are some quick and incomplete notes on a [fantastic post from Dr. Jonas Lindeløv](https://lindeloev.github.io/tests-as-linear/)

## Correlation as linear models

Let's begin with the simple linear regression model:

$$y(x) = \alpha + \beta x$$

The parameter estimates are:

$$\hat{\alpha} = \bar{y} - \hat{\beta}\bar{x}$$

$$\hat{\beta} = \frac{cov(x, y)}{var(x)} = r(x,y)\frac{\sigma_y}{\sigma_x}$$

Recall $r(x, y)$ is the Pearson correlation coefficient:

$$r(x, y) = \frac{cov(x, y)}{\sigma_x\sigma_y}$$

So the slope $\hat{\beta}$ clearly can be expressed in terms of $r$:

$$\hat{\beta} = r(x,y)\frac{\sigma_y}{\sigma_x}$$

And if we standardized our data to standard deviations of 1, the slope is exactly the Pearson correlation.

### Ranks

Rank transformation is just replacing a vector of numbers with the integer of their rank. The smallest number becomes 1, the second smallest becomes 2, and so on. 

Spearman rank correlation is simply Pearson correlation but on rank-transformed data:

$$r_{rank} = \frac{cov(rank(x), rank(y))}{\sigma_{rank(x)}\sigma_{rank(y)}}$$

Rank correlation coefficients don't require a linear relationship. Suppose we had points (0, 1), (10, 1000), (1000, 1200). An increase in $x$ is always paired with an increase in $y$ - this is rank correlation of 1. The increase is not linear, though - the Pearson correlation is $0.64$. Rank correlation can be useful for evaluating relationships with ordinal variables.

We can recover Spearman correlation from a linear model as well:

$$rank(y) = \alpha + \beta rank(x)$$

$$\hat{\beta} = r_{rank}$$

## One-sample t-tests

The t-test refers to statistical hypothesis tests where the test statistic is distributed according to Student's t, under the null hypothesis. In the one-sample t-test, we are comparing a sample mean to a known population mean. 

If we can assume the population of sample means $\bar{X}$ is normally distributed, the test statistic $t$ is distributed according to Student's t with $v=N-1$ degrees of freedom. 

$$t = \frac{\bar{X}-\mu}{\sigma/\sqrt{N}}$$

Notice that this assumption explains the robustness of the t-test, via the central limit theorem. The distribution of $\bar{X}$ converges towards normality regardless of the distribution of $x$. If $x$ were i.i.d normal to begin with, then we would satisfy our assumption immediately - but this is not strictly necessary thanks to the CLT.

Armed with this $t$ value, we can look up the critical $t$ value from the Student's t-distribution table accordingly. If $t >$ the critical value at our chosen confidence level, we reject the null hypothesis that there is no difference between the sample mean and $\mu$.

At significance level $a$, we reject the null hypothesis if $|t| > t_{1-a/2,v}$

It turns out that the one-sample t-test can be expressed as simply as:

$$y = \alpha$$

## Two-sample t-tests

In a two-sample t-test, we have two i.i.d samples from two respective populations, and we want to compare their means. 

The test statistic is:

$$t = \frac{\bar{X_1} - \bar{X_2}}{\sqrt{\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}}}$$

with degrees of freedom

$$v = \frac{(\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2})^2}{\frac{(s_1^2/N_1)^2}{N_1-1} + \frac{(s_1^2/N_2)^2}{N_2-1}}$$

This is the Welch's t-test, which does not assume identical variances in the two populations. Student's two-sample t-test does, and is easier to express as a linear model (Welch's requires some different handling of the variances). But the idea is the same:

$$y = \alpha + \beta x$$

where $x$ is an indicator variable indicating which population the observation came from. 

The Wilcoxon rank-sum test can be expressed as:

$$rank(y) = \alpha + \beta x$$

