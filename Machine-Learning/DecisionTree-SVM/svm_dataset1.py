import pickle
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt


dataset, labels = pickle.load(open("../data/part2_dataset1.data", "rb"))


# Plot the decision boundary with the data points
def plot_decision_boundary(model, xx, yy, **params):
    plt.clf()
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, **params)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap=plt.cm.coolwarm)
    return plt


kernels = ["linear", "rbf"]
c_values = [0.1, 100.0]

# Train the model for each configuration and plot the decision boundary
for kernel in kernels:
    for c in c_values:
        model = svm.SVC(kernel=kernel, C=c)
        model.fit(dataset, labels)
        print("Kernel: {}, C: {}, Accuracy: {}".format(
            kernel, c, model.score(dataset, labels)))

        # Plot the decision boundary for the current configuration
        x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
        y_min, y_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        plot = plot_decision_boundary(
            model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        plot.savefig("Dataset1_Kernel={}_C={}.png".format(kernel, c))
