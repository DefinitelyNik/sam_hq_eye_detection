import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import SGDClassifier

iris = datasets.load_iris()

colors = "bry"

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
y = numpy_target

clf = SGDClassifier(alpha=0.001, max_iter=10000).fit(X, y)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.Paired,
    ax=ax,
    response_method="predict",
    xlabel="mask_percentage",
    ylabel="bbox_percentage",
)
plt.axis("tight")

for i, color in zip(clf.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(
        X[idx, 0],
        X[idx, 1],
        c=color,
        label=iris.target_names[i],
        cmap=plt.cm.Paired,
        edgecolor="black",
        s=20,
    )
plt.title("Decision surface of multi-class SGD")
plt.axis("tight")

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = clf.coef_
intercept = clf.intercept_
print(clf.predict([[0.32098765432098764, 47.91666666666667]]))

def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

    plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)


for i, color in zip(clf.classes_, colors):
    plot_hyperplane(i, color)
plt.legend()
plt.show()