from common import *

import matplotlib.pyplot as plt

test_x = np.linspace(0, 2 * np.pi)
pred = gp.predict(np.atleast_2d(test_x).T, what=("mean", "mse"))
mean = np.squeeze(pred["mean"])  # There is only one output dimension.
mse = pred["mse"]

plt.fill_between(test_x, mean - mse, mean + mse, color=(0.8, 0.8, 1.0))
plt.plot(test_x, pred["mean"])
plt.scatter(x, y)
