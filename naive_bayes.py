from sklearn import datasets
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# x, y = datasets.make_classification()
iris = datasets.load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = naive_bayes.GaussianNB()
clf.fit(x_train, y_train)

acc = clf.score(x_test, y_test)
predict = clf.predict(x_test)
print(acc)

col = np.array(['b', 'g', 'r'])
plt.subplot(1, 2, 1)
plt.scatter(x_test[:, 0], x_test[:, 1], c=col[y_test])
plt.title('Actual')
plt.subplot(1, 2, 2)
plt.scatter(x_test[:, 0], x_test[:, 1], c=col[predict])
plt.title('Predicted')
plt.show()
