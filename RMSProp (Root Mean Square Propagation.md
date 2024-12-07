

**The algorithm**
$$
\begin{align}
g_{i} & =\frac{\partial}{\partial \theta_{i}}\text{Loss}(y^{(k)},F(\vec{x}^{(k)};\vec{\theta})) \\
G_{i} & = \alpha G_{i}+(1-\alpha)g_{i}^{2} \\
\theta_{i} & =\theta_{i}-\frac{\eta}{\sqrt{ G_{i}+ \epsilon }}g_{i}
\end{align}
$$

