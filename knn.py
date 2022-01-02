from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
print(knn.score(x_test, y_test))
predict = knn.predict(x_test)
col = np.array(['b', 'g','y','r'])
plt.subplot(1, 2, 1)
plt.scatter(x_test[:, 0], x_test[:, 1], c=col[y_test])
plt.title('Actual')
plt.subplot(1, 2, 2)
plt.scatter(x_test[:, 0], x_test[:, 1], c=col[predict])
plt.title('Predicted')
plt.show()
