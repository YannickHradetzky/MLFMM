The AdaGrad algorithm adjusts the learning rate for each parameter based on the magnitudes of the gradient. This means that parameters with large gradients have their effective learning rate reduced, while parameters with smaller gradients have their learning rates increased, relatively speaking. AdaGrad is particularly well-suited for dealing with sparse data, as it helps make larger updates to infrequent features. One limitation of AdaGrad is that the accumulated squared gradients in Gi can lead to the effective learning rate becoming infinitesimally small, which may cause the model to stop learning too early.
$$
g_{i} = \frac{\partial}{\partial \theta_{i}}\text{Loss}(y^{(k)},F(\vec{x}^{(k)};\vec{\theta}))
$$
**Cummulative squared gradient**
$$
G_{i} = G_{i}+g_{i}^{2}
$$
**adaptive gradient update**
$$
\theta_{i}^{t+1} = \theta_{i}^{t}-\frac{\eta}{\sqrt{ G_{i}+\epsilon }}g_{i}
$$

