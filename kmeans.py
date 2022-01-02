from sklearn import datasets
from sklearn import cluster
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

x, y = datasets.make_blobs()
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = cluster.KMeans(n_clusters=3)
clf.fit(x, y)

acc = clf.score(x, y)
predict = clf.predict(x)
print(acc)

col = np.array(['b', 'g', 'r', 'y'])
plt.subplot(1, 2, 1)
plt.scatter(x[:, 0], x[:, 1], c=col[y])
plt.title('Actual')
plt.subplot(1, 2, 2)
plt.scatter(x[:, 0], x[:, 1], c=col[predict])
plt.title('Predicted')
plt.show()
