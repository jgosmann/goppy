import numpy as np

np.random.seed(3)  # Make the examples reproducible.

n = 15
x = np.atleast_2d(2 * np.random.rand(n)).T
y = np.cos(x) + 0.1 * np.random.randn(n, 1)


from goppy import OnlineGP, SquaredExponentialKernel

gp = OnlineGP(SquaredExponentialKernel(0.5), noise_var=0.1)
gp.fit(x, y)
