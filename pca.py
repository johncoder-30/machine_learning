from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# x, y = datasets.make_classification()
iris = datasets.load_iris()
x, y = iris.data, iris.target
pca = decomposition.PCA(n_components=2)
principalComponents = pca.fit_transform(x)


col = np.array(['b', 'g', 'r'])
plt.subplot(1, 2, 1)
plt.scatter(x[:, 0], x[:, 1], c=col[y])
plt.title('Actual')
plt.subplot(1, 2, 2)
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=col[y])
plt.title('Predicted')
plt.show()
