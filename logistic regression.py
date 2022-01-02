from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

c = np.array(['b', 'g'])
x, y = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

logistic = linear_model.LogisticRegression()
logistic.fit(x_train, y_train)
acc = logistic.score(x_test,y_test)
plt.subplot(1, 2, 1)
plt.scatter(x_train[:,0],y_train,c=c[y_train])
plt.subplot(1, 2, 2)
plt.scatter(x_train[:,1],y_train,c=c[y_train])
plt.show()
print(acc)


