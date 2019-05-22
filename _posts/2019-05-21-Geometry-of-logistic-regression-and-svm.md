---
layout: post
title: Geometry of Logistic Regression vs. SVM
---

## Geometric intuition behind logistic regression and support vector machines


It always helps to have an intuition for the geometric meaning of a model. This is typically emphasized in the case of other common models such as linear regression and support vector machines, but there are relatively few discussions of this for logistic regression.

Recall in logistic regression we are modeling $$P(c_1|x, w) = \sigma(w^Tx)$$