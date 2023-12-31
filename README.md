# bayes-linreg

This package fits the linear regression model

$$
\begin{aligned}
P(\mathbf{w} \mid \alpha) &= \text{Normal}(\mathbf{w} \mid 0,\ \alpha^{-1}\mathbf{I}) \\
P(\mathbf{y} \mid \mathbf{w},\ \beta) &= \text{Normal}(\mathbf{y}\mid \mathbf{Xw},\ \beta^{-1}\mathbf{I})
\end{aligned}
$$

Rather than finding a point estimate for $\mathbf{w}$, we take a Bayesian approach and find the posterior distribution, given by

$$
P(\mathbf{w},\ \mathbf{y}) = \text{Normal}(\mathbf{w}\mid \mu,\ \mathbf{H})\ ,
$$

where

$$
\begin{aligned}
\mathbf{\mu} &= \beta \mathbf{H}^{-1}\mathbf{X}^T \mathbf{y} \\
\mathbf{H} &= \alpha \mathbf{I} + \beta \mathbf{X}^T\mathbf{X}\ .
\end{aligned}
$$

The algorithm works by maximizing the marginal likelihood

$$
\begin{aligned}
P(\mathbf{y} \mid \alpha,\ \beta) &= \int P(\mathbf{y}\mid \mathbf{w},\ \alpha,\ \beta)\ d\mathbf{w} \\
&= \frac{1}{2}\left(   n \log \alpha  + k \log \beta  - \beta \lVert \mathbf{y} - \mathbf{X\mu} \rVert^2  - \alpha \lVert \mathbf{\mu} \rVert ^ 2 -\log \left| \alpha \mathbf{I} +\beta \mathbf{X}^T \mathbf{X}\right| - n \log 2\pi
      \right)\ .
\end{aligned}
$$

using the approach described in Bishop (2006), which cycles through


1. Update α and β

2. Update H

3. Update μ

For details, see

Bishop, C. M. (2006). [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) (M. Jordan, J. Kleinberg, & B. Schölkopf (eds.); Vol. 53, Issue 9). Springer. Available online: https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf
