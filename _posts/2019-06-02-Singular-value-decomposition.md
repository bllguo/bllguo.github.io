---
layout: post
title: Singular Value Decomposition
---

Singular value decomposition is a crucially important method underpinning the mathematics behind all kinds of applications in machine learning. At its core, it is a linear algebra technique for decomposing a matrix. It separates any matrix $A$ into simple pieces.

A simple example of the utility of the SVD is in compression. Imagine we have a matrix $A$:

$$A = \begin{bmatrix}1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\end{bmatrix}$$  

We could send all $m \times n \rightarrow 5 \times 5 = 25$ numbers. Or, we could realize that:

$$A = \begin{bmatrix}1 \\ 1 \\ 1 \\ 1 \\ 1\end{bmatrix}\begin{bmatrix}1 & 1 & 1 & 1 & 1\end{bmatrix}$$

We now only need to send $m + n \rightarrow 5 + 5 = 10$ numbers! Of course this is a contrived example, but the intuition is clear - if we can define "special" vectors like these vectors of ones, we can represent $A$ in a much more efficient manner.

SVD will find these pieces for us. It will choose orthonormal vectors $u_iv_i^T$ in order of "importance."

## 1. SVD Formulation

$$A = U\Sigma V^T$$

$$A = u_1\sigma_1v_1^T + ... + u_r\sigma_rv_r^T$$

Where:

$$U = \begin{bmatrix}{} & {} & {} & {} & {}\\u_1 & ... & u_r & ... & u_m\\{} & {} & {} & {} & {}\\\end{bmatrix}$$

$$V = \begin{bmatrix}{} & {} & {} & {} & {}\\v_1 & ... & v_r & ... & v_n\\{} & {} & {} & {} & {}\\\end{bmatrix}$$

$$\sigma = \begin{bmatrix}\sigma_1 & {} & {} & {}\\{} & ... & {} & {}\\{} & {} & \sigma_r & {}\\{} & {} & {} & {}\end{bmatrix}$$

$u_1, ..., u_m$ are the eigenvectors of $AA^T$, otherwise known as the left singular vectors of $A$. 

$v_1, ..., v_n$ are the eigenvectors of $A^TA$, otherwise known as the right singular vectors of $A$.

$\sigma_1, ..., \sigma_r$ are the eponymous singular values. They are the square roots of the equal eigenvalues of $AA^T$ and $A^TA$. Proof in appendix A1 at the bottom of the post.

$$AA^Tu_i = \sigma_i^2u_i$$

$$A^TAv_i = \sigma_i^2v_i$$

$r$ is the rank of $A$, which is also the rank of $A^TA$ and $AA^T$ (appendix A2). We will discuss the importance of this in the next section.

## 2. Proof of SVD

We start with the matrix $A^TA$. Notice that it is not only symmetric, but also positive semi-definite (appendix A3). By the spectral theorem the eigenvectors $V$ can be chosen to be orthonormal. Hence $V^{-1} = V^T$, so we can diagonalize the matrix: 

$$A^TA = V\Delta V^T$$

And because $A^TA$ is psd, all the eigenvalues are $\geq 0$ by definition. 

Let $v_1, ..., v_r$ be an orthonormal set of eigenvectors for the positive eigenvalues, sorted in decreasing order. The remainder, $v_{r+1}, ..., v_n$ is a basis for the zero-eigenspace $A^TAx = \lambda x = 0$, which is also the nullspace of $A^TA$. Thus they are a basis for the nullspace of $A$.

And as the nullspace is orthogonal to the row space, clearly $v_1, ..., v_r$ is a basis for the row space.

If $v$ is a unit eigenvector of $A^TA$ with eigenvalue $\sigma^2$:

$$A^TAv = \sigma^2v$$

$$AA^T(Av) = \sigma^2(Av)$$

$$AA^T(\frac{1}{\sigma}Av) = \sigma^2(\frac{1}{\sigma}Av)$$

Let $u = (\frac{1}{\sigma}Av)$ - it is an eigenvector of $AA^T$ with eigenvalue $\sigma^2$! And we can choose it to be a unit eigenvector, thanks to the spectral theorem. This is key!

Now if we take $V_r$ as the first $r$ columns of $V$ ($n \times r$), $\Sigma_r$ the diagonal matrix with $i$th entry $\sigma_i$ ($r \times r$), and $U_r$ as $m \times r$ matrix with $i$th column $u_i = \frac{1}{\sigma_i}AV_i$:

$$U_r = AV_r\Sigma_r^{-1} \rightarrow U\Sigma_r = AV_r$$

$$U_r\Sigma_rV_r^T = AV_rV_r^T = A$$

The vectors $u_1, ..., u_r$ are $r$ orthonormal vectors in the column space of $A$. We can see this because the columns of $A$ are combinations of the columns of $U\Sigma_r$ and $\Sigma_r$ is just a diagonal matrix! Thus  $u_1, ..., u_r$ spans the column space of $A$ - it is an orthonormal basis. Then the remaining orthonormal vectors $u_{r+1}, ..., u_m$ must span the left nullspace.

Notice that the remainder $n-r\text{ }v$'s and $m-r\text{ }u$'s are orthogonal to the earlier $v$'s and $u$'s, and must have eigenvalues of 0. So if we include them in $U, V$, we still have $A = U\sigma V^T$. We finally arrive at:

$$AV = U\Sigma$$

$$A = U\Sigma V^T$$

$$A = u_1\sigma_1v_1^T + ... + u_r\sigma_rv_r^T$$

Not only that, but SVD has provided us with bases for all the four fundamental subspaces of $A$!

The singular values $\sigma$ give us an idea of the "importance" of each piece $u\sigma v^T$. In an compression context, for instance, we can discard the pieces with small singular values, with minimal data loss. This idea will be clearer when we discuss PCA.

## Appendix - Misc. Linear Algebra Proofs

### A1. Proof that $AA^T, A^TA$ share nonzero eigenvalues:
Suppose $AA^Tx = \lambda x$. Multiply both sides by $A^T$:

$$A^TA(A^Tx) = \lambda(A^Tx)$$

QED

### A2. Proof that $\text{rank}(A) = \text{rank}(A^TA)$

First off, $A$ and $A^TA$ share the same nullspace.

Case 1. 

$$Ax = 0$$

$$A^TAx = 0$$

$$x \in N(A^TA)$$

Thus $N(A) \subseteq N(A^TA)$

Case 2. 

$$A^TAx = 0$$

$$x^TA^TAx = 0$$

$$(Ax)^T(Ax) = 0$$

$$Ax = 0$$

$$x \in N(A)$$

Thus $N(A^TA) \subseteq N(A)$

So they share the same nullspace.

Now, recall the rank-nullity theorem:

$$\text{rank}(A) + \text{dim}(N(A)) = n$$

where $n$ is the number of columns.

We just showed that $\text{dim}(N(A)) = \text{dim}(N(A^TA))$.

If $A$ is $m\times n$ then $A^TA$ is $n\times n$.

Putting it all together, $\text{rank}(A) = \text{rank}(A^TA)$. And we are done!

By a similar argument we can show that $\text{rank}(A) = \text{rank}(AA^T)$ - remember that $\text{rank}(A) = \text{rank}(A^T)$.

### A3. Proof that $A^TA$ and $AA^T$ are positive semi-definite

By definition, $A^TA$ is psd if for any $x$

$$x^TA^TAx \geq 0$$

The LHS is:

$$(Ax)^TAx = ||Ax||^2 \geq 0$$

By definition $AA^T$ is psd if for any $x$\

$$xAA^Tx^T \geq 0$$

The LHS is:

$$(xA)(xA)^T = ||xA||^2$$ 

Remember that the inner product of a vector with itself - its squared magnitude - is always $\geq 0$. QED