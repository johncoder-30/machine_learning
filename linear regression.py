from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

x, y = datasets.make_regression(n_samples=100, n_features=5)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
predicted = linear.predict(x_test)

col = np.array(['b', 'g', 'c', 'y', 'r'])
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.scatter(x[:, i], y, c=col[i])
    plt.plot(x[:, i], linear.coef_[i] * x[:, i] + linear.intercept_)
plt.show()
