Is a technique to automatically compute derivatives accurately and efficiently.
It works by splitting the relevant function into a combination of **base functions** with known derivatives. Then The **chain rule** is utilized to obtain the derivative of the function in question.

**Forward Mode**
Efficient if $n_{target}\gg n_{features}$


**Backward Mode**
Efficient if $n_{features}\gg n_{targets}$


# Jax
Precompute derivative of functions which can then be applied many times

# PyTorch
Generates computational graph (History of computational Steps)



