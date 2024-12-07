- We are interested in Optimization Methods that only use the first derivative to avoid big computational expenses
- The step size we take is sometimes dependent on the value of the gradient. 

In [[AdaGrad (Adaptive Gradient Algorithm)]] the problem of overshooting in regimes with a high gradient is solved. However the learning rate will become very small towards the end of the training. Because $G$ always grows and our $\eta_{eff}\to 0$

[[RMSProp (Root Mean Square Propagation]] modifies AdaGrad by keeping an exponentially decaying average of past squared gradients, which helps to avoid the drastic decay in learning rates. This allows RMSProp to perform better in non-convex settings, such as training deep neural networks.

In [[Stochastic Gradient Descent with Momentum]] we modify the direction of the update (no square!) by using an exponential average of prior gradients

Combining the benefits of [[Stochastic Gradient Descent with Momentum]] and [[Stochastic Gradient Descent with Momentum]] yields: 

[[ADAM (Adaptive Moment Estimation)]] probably the most popular and effective optimization algorithms in machine learning today due to its adaptive learning rate mechanism and momentum incorporation. It typically works well across a wide range of problems and requires little tuning.
