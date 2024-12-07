The optimization process can oscillate, resulting in slow convergence. To alleviate this problem, momentum is introduced into the optimization process. Momentum helps smooth the updates by accumulating a running average of past gradients, which allows the optimization process to maintain a consistent direction in the parameter space, thus speeding up convergence and reducing oscillations.

$$
\begin{align}
g_{i} & =\frac{\partial}{\partial \theta_{i}}\text{Loss}(y^{(k)},F(\vec{x}^{k},\vec{\theta})) \\
m_{i} & =\gamma m_{i}+\eta g_{i} \\
\theta_{i} & =\theta_{i}-m_{i}
\end{align}
$$

- $\gamma$ is the momentum coefficient (typically set to a value like 0.9), which controls how much of the previous velocity is retained.
