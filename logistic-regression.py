import matplotlib.pyplot as plt
import numpy as np

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

target = []
data = []

with open('target.txt', 'r') as txt_file:
    txt = txt_file.read()
    target_buff = [int(x) for x in txt.split(" ")]
    for i in target_buff:
        if i == 3: target.append(2)
        else: target.append(i)

with open('data.txt', 'r') as txt_file:
    txt = txt_file.read()
    array = txt.split("\n")
    for item in array:
        xd = []
        digits = item.split(" ")
        if(len(digits) < 2):
            break
        for i in range(len(digits) - 1):
            xd.append(float(digits[i]))
        data.append(xd)

numpy_data = np.array(data)
numpy_target = np.array(target)

X = numpy_data
Y = numpy_target

logreg = LogisticRegression(C=1e5)
logreg.fit(X, Y)

_, ax = plt.subplots(figsize=(4, 3))
DecisionBoundaryDisplay.from_estimator(
    logreg,
    X,
    cmap=plt.cm.Paired,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    xlabel="Sepal length",
    ylabel="Sepal width",
    eps=0.5,
)

plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors="k", cmap=plt.cm.Paired)

plt.xticks(())
plt.yticks(())

plt.show()
print(logreg.predict([[0.32098765432098764, 47.91666666666667]]))