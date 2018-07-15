#implement pca
import numpy as np
import matplotlib.pyplot as plt


#generate fake data:
mean = [0, 0]
cov = [[1, 5], [8, 10]]  # diagonal covariance


x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()


