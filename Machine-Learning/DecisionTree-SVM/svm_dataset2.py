import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

# Perform 10-fold cross validation for 4 configurations
parameters = {"kernel": ["linear", "rbf"], "C": [0.1, 100.0]}
model = svm.SVC()
grid_search = GridSearchCV(model, parameters, cv=10, n_jobs=-1)

confidence_intervals = []
print("-" * 70)
print("Performing 10-fold cross validation for 4 configurations...")
for _ in range(5):
    # Data preprocessing
    dataset = StandardScaler().fit_transform(dataset)

    # Shuffle the dataset and labels
    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)
    dataset = dataset[indices]
    labels = labels[indices]

    grid_search.fit(dataset, labels)

    # Calculate each hyperparameter configuration's accuracy confidence interval
    for i, params in enumerate(grid_search.cv_results_["params"]):
        mean_accuracy = grid_search.cv_results_["mean_test_score"][i]
        std_accuracy = grid_search.cv_results_["std_test_score"][i]
        sample_size = grid_search.cv_results_["split0_test_score"].shape[0]
        confidence_interval = [(mean_accuracy - 1.96 * std_accuracy / np.sqrt(
            sample_size)) * 100, (mean_accuracy + 1.96 * std_accuracy / np.sqrt(sample_size)) * 100]

        # Make sure the confidence interval is in the range [0, 1]
        if confidence_interval[0] < 0:
            confidence_interval[0] = 0
        if confidence_interval[1] > 100:
            confidence_interval[1] = 100

        confidence_intervals.append(
            [params, mean_accuracy * 100, confidence_interval])

print("-" * 70)
print("10-fold cross validation results:")
for no, (params, mean_accuracy, confidence_interval) in enumerate(confidence_intervals):
    if no % 4 == 0:
        print("\nCV {}:".format(no//4+1))
    print("Config No:{}: Kernel: {}, C: {}, Accuracy: {:5f}, Confidence Interval: {}".format(
        no + 1, params["kernel"], params["C"], mean_accuracy, confidence_interval))


best_config = np.argmax([confidence_interval[1]
                        for confidence_interval in confidence_intervals])

print("-" * 70)
print("Best configuration:\n")
print("Config No: {}, Kernel: {}, C: {}, Accuracy: {:.5f}, Confidence Interval: {}".format(best_config + 1,
                                                                                           confidence_intervals[best_config][0]["kernel"], confidence_intervals[best_config][0]["C"], confidence_intervals[best_config][1], confidence_intervals[best_config][2]))
