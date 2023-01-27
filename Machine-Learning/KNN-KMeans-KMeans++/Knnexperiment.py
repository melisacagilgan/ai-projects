import pickle
from Part1.Knn import KNN
from Distance import Distance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))
print(
    f"-----------------------------------\nDataset is loaded! Shape: {dataset.shape}")

similarity_functions = {Distance.calculateCosineDistance: 'cosine',
                        Distance.calculateMinkowskiDistance: 'minkowski', Distance.calculateMahalanobisDistance: 'mahalanobis'}
similarity_function_parameters = [None, 2, np.linalg.inv(np.cov(dataset.T))]
n_neigbors = [3, 5, 10]
cv_results = {}

print("-----------------------------------\nTraining begins...")
config_no = 1

# Apply 5 times 10-fold cross validation on the dataset and calculate the average accuracy, precision and recall
for similarity_function, similarity_function_parameter in zip(similarity_functions.keys(), similarity_function_parameters):
    for n in n_neigbors:
        accuracies = []
        precisions = []
        recalls = []
        for _ in range(5):
            k_fold = StratifiedKFold(n_splits=10, shuffle=True)
            for train_index, test_index in k_fold.split(dataset, labels):
                x_train, y_train = dataset[train_index], labels[train_index]
                x_test, y_test = dataset[test_index], labels[test_index]

                knn = KNN(x_train, y_train, similarity_function,
                          similarity_function_parameter, n)

                predictions = []
                for instance in x_test:
                    predictions.append(knn.predict(instance))
                accuracies.append(accuracy_score(y_test, predictions)*100)
                precisions.append(precision_score(
                    y_test, predictions, average='macro')*100)
                recalls.append(recall_score(
                    y_test, predictions, average='macro')*100)

        mean_accuracy = float("{:.3f}".format(np.mean(accuracies)))
        mean_precision = float("{:.3f}".format(np.mean(precisions)))
        mean_recall = float("{:.3f}".format(np.mean(recalls)))
        cv_results[config_no] = {'distance': similarity_functions[similarity_function], 'n_neigbors': n, 'accuracy_confidence_interval': [np.mean(accuracies) - 1.96 * np.std(accuracies) / np.sqrt(
            len(accuracies)), np.mean(accuracies) + 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))], 'accuracy': mean_accuracy, 'precision': mean_precision, 'recall': mean_recall}
        config_no += 1

print("-----------------------------------\nCross-Validation Results")
for no, value in cv_results.items():
    print(f'Config {no}: {value}')

max_accuracy = max(cv_results, key=lambda x: cv_results[x]['accuracy'])
best_accuracy = [k for k, v in cv_results.items(
) if v['accuracy'] == cv_results[max_accuracy]['accuracy']]

max_precision = max(cv_results, key=lambda x: cv_results[x]['precision'])
best_precision = [k for k, v in cv_results.items(
) if v['precision'] == cv_results[max_precision]['precision']]

max_recall = max(cv_results, key=lambda x: cv_results[x]['recall'])
best_recall = [k for k, v in cv_results.items() if v['recall'] ==
               cv_results[max_recall]['recall']]

print(f"-----------------------------------\nBest configuration number based on accuracy score is {best_accuracy}\n"
      f"Best configuration number based on precision score is {best_precision}\n"
      f"Best configuration number based on recall score is {best_recall}")

# Best configuration chosen as the most frequent configuration in the best_accuracy, best_precision and best_recall lists
best_config = max(set(best_accuracy + best_precision + best_recall),
                  key=(best_accuracy + best_precision + best_recall).count)
print(
    f"-----------------------------------\nBest configuration number is {best_config}")
