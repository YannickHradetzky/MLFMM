The Adam optimizer combines the advantages of both AdaGrad and RMSProp. It computes individual adaptive learning rates for each parameter and also incorporates momentum, which helps, as we just discussed, to smooth out parameter updates. Momentum in optimization refers to the use of the moving average of past gradients to stabilize the direction of updates. Adam maintains two moving averages: one for the gradients (first moment) and one for the squared gradients (second moment).

$$
\begin{align}
\vec{g_{t}}  & = \frac{\partial}{\vec{\partial \theta}}\text{Loss}(y^{(k)};\vec{\theta}) \text{  (Gradient at time t)} \\
\vec{m_{t}} & =\beta_{1}m_{t-1}+(1-\beta_{1})\vec{g_{t}} \text{  (Updated biased first momentum estimate)} \\
G_{t}  & = \beta_{2}G_{t-1}+(1-\beta_{2})\vec{g_{t}}^{2}\text{   (Updated biased second raw moment estimate)} \\
\hat{\vec{m}}_{t}  & = \frac{\vec{m_{t}}}{1-\beta_{1}}, \hat{G_{t}}=\frac{G_{t}}{1-\beta_{2}} \text{.    (Compute bias-correction)} \\
\vec{\theta_{t}} & =\vec{\theta}_{t-1}-\frac{\eta}{\sqrt{ \hat{G_{t}} }+\epsilon}\hat{\vec{m_{t}}}
\end{align}
$$
