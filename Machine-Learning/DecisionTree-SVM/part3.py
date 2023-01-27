from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from DataLoader import DataLoader
from sklearn.preprocessing import MinMaxScaler


data_path = "data/credit.data"
dataset, labels = DataLoader.load_credit_with_onehot(data_path)


# models to be used
models = {"KNN": KNeighborsClassifier(), "SVM": SVC(
), "Decision Tree": DecisionTreeClassifier(), "Random Forest": RandomForestClassifier()}


# hyperparameters for each model to be tuned
model_params = {}
for model_name, model in models.items():
    if model_name == "KNN":
        model_params[model] = {"n_neighbors": [3, 5]}
    elif model_name == "SVM":
        model_params[model] = {"C": [0.1, 100]}
    elif model_name == "Decision Tree":
        model_params[model] = {"max_depth": [2, 3]}
    elif model_name == "Random Forest":
        model_params[model] = {"n_estimators": [10, 100]}


# Shuffle the dataset and labels
dataset, labels = shuffle(dataset, labels, random_state=7)


outer_cv = RepeatedStratifiedKFold(
    n_splits=3, n_repeats=5, random_state=7)
inner_cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=5, random_state=7)


print("-" * 50 + "\nStarting Nested Cross Validation...")

inner_cv_no = 0
inner_config = {}
outer_cv_no = 0
outer_cv_config = {}

# nested cross validation
for train_index, test_index in outer_cv.split(dataset, labels):
    x_train_val, x_test = dataset[train_index], dataset[test_index]
    y_train_val, y_test = labels[train_index], labels[test_index]

    # Shuffle the training set
    x_train_val, y_train_val = shuffle(
        x_train_val, y_train_val, random_state=7)

    # grid search according to accuracy and f1 score
    for model in models.values():
        for key, values in model_params[model].items():
            for value in values:
                inner_cv_no += 1

                if isinstance(model, RandomForestClassifier):
                    param = value
                else:
                    param = {key: list([value])}

                accuracies = np.array([])
                f1_scores = np.array([])

                # inner cross validation
                for train_index, val_index in inner_cv.split(x_train_val, y_train_val):
                    x_train, x_val = x_train_val[train_index], x_train_val[val_index]
                    y_train, y_val = y_train_val[train_index], y_train_val[val_index]

                    # min-max normalization between -1 and 1
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    x_train = scaler.fit_transform(x_train)
                    x_val = scaler.transform(x_val)

                    if isinstance(model, RandomForestClassifier):
                        accs = np.array([])
                        f1s = np.array([])

                        # run it at least 5 times on a particular test and training partitioning
                        for _ in range(5):
                            model = RandomForestClassifier(param, n_jobs=-1)
                            model.fit(x_train, y_train)
                            y_pred = model.predict(x_val)
                            accs = np.append(
                                accs, accuracy_score(y_val, y_pred))
                            f1s = np.append(f1s, f1_score(y_val, y_pred))

                        f1_scores = np.append(f1_scores, np.mean(f1s))
                        accuracies = np.append(accuracies, np.mean(accs))

                    else:
                        grid_search = GridSearchCV(model, param, cv=inner_cv,
                                                   scoring=["accuracy", "f1"], refit="f1", n_jobs=-1)
                        grid_search.fit(x_train, y_train)

                        # Calculate each hyperparameter configuration's accuracy and confidence interval
                        f1_scores = np.append(
                            f1_scores, grid_search.cv_results_["mean_test_f1"])
                        accuracies = np.append(accuracies, grid_search.cv_results_[
                            "mean_test_accuracy"])

                mean_f1_score = np.mean(f1_scores)
                std_f1_score = np.std(f1_scores)
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                f1_confidence_interval = [mean_f1_score - 1.96 * std_f1_score / np.sqrt(
                    len(f1_scores)), mean_f1_score + 1.96 * std_f1_score / np.sqrt(len(f1_scores))]
                accuracy_confidence_interval = [mean_accuracy - 1.96 * std_accuracy / np.sqrt(
                    len(accuracies)), mean_accuracy + 1.96 * std_accuracy / np.sqrt(len(accuracies))]

                # make sure the confidence interval is between 0 and 100
                if f1_confidence_interval[0] < 0:
                    f1_confidence_interval[0] = 0
                if accuracy_confidence_interval[0] < 0:
                    accuracy_confidence_interval[0] = 0
                if f1_confidence_interval[1] > 100:
                    f1_confidence_interval[1] = 100
                if accuracy_confidence_interval[1] > 100:
                    accuracy_confidence_interval[1] = 100

                # save the configuration's results
                inner_config[inner_cv_no] = {"model": model, "model_params": param, "f1_score": mean_f1_score * 100,
                                             "f1_confidence_interval": f1_confidence_interval, "accuracy": mean_accuracy * 100, "accuracy_confidence_interval": accuracy_confidence_interval}

    outer_cv_no += 1

    # find the best configuration
    best_f1_config = max(
        inner_config, key=lambda x: inner_config[x]["f1_score"])
    best_accuracy_config = max(
        inner_config, key=lambda x: inner_config[x]["accuracy"])

    # train the best model according to f1 score on the whole training set
    best_f1_model = inner_config[best_f1_config]["model"]
    best_f1_model.fit(x_train_val, y_train_val)
    f1_accuracy = accuracy_score(y_test, best_f1_model.predict(x_test))
    f1_f1 = f1_score(y_test, best_f1_model.predict(x_test))

    # train the best model according to accuracy on the whole training set
    best_accuracy_model = inner_config[best_accuracy_config]["model"]
    best_accuracy_model.fit(x_train_val, y_train_val)
    acc_accuracy = accuracy_score(y_test, best_accuracy_model.predict(x_test))
    acc_f1 = f1_score(y_test, best_accuracy_model.predict(x_test))

    # save the best configuration for each metric
    outer_cv_config[outer_cv_no] = {"best_f1_model": best_f1_model, "best_f1_model_params": inner_config[best_f1_config]["model_params"], "best_f1_model_f1_score": f1_f1 * 100, "best_f1_model_accuracy": f1_accuracy * 100,
                                    "best_accuracy_model": best_accuracy_model, "best_accuracy_model_params": inner_config[best_accuracy_config]["model_params"], "best_accuracy_model_f1_score": acc_f1 * 100, "best_accuracy_model_accuracy": acc_accuracy * 100}


# print the inner cross validation results
print("-" * 50 + "\nInner Cross Validation Results\n")
for no, config in inner_config.items():
    print("Configuration {}:\n\tModel: {}\n\tModel Parameters: {}\n\tF1 Score: {}\n\tF1 Confidence Interval: {}\n\tAccuracy: {}\n\tAccuracy Confidence Interval: {}".format(
        no, config["model"], config["model_params"], config["f1_score"], config["f1_confidence_interval"], config["accuracy"], config["accuracy_confidence_interval"]))

# print the outer cross validation results
print("-" * 50 + "\nOuter Cross Validation Results\n")
for no, config in outer_cv_config.items():
    print("Configuration {}:\n\tBest F1 Model: {}\n\tBest F1 Model Parameters: {}\n\tBest F1 Model F1 Score: {}\n\tBest F1 Model Accuracy: {}\n\tBest Accuracy Model: {}\n\tBest Accuracy Model Parameters: {}\n\tBest Accuracy Model F1 Score: {}\n\tBest Accuracy Model Accuracy: {}".format(
        no, config["best_f1_model"], config["best_f1_model_params"], config["best_f1_model_f1_score"], config["best_f1_model_accuracy"], config["best_accuracy_model"], config["best_accuracy_model_params"], config["best_accuracy_model_f1_score"], config["best_accuracy_model_accuracy"]))


# find the best configuration according to each metric
best_f1_config = max(
    outer_cv_config, key=lambda x: outer_cv_config[x]["best_f1_model_f1_score"])
best_accuracy_config = max(
    outer_cv_config, key=lambda x: outer_cv_config[x]["best_accuracy_model_accuracy"])

# print the best configuration according to each metric
print("-" * 50 + "\nBest Configuration\n")
print("Best F1 Model: {}\n\tBest F1 Model Parameters: {}\n\tBest F1 Model F1 Score: {}\n\tBest F1 Model Accuracy: {}\n\tBest Accuracy Model: {}\n\tBest Accuracy Model Parameters: {}\n\tBest Accuracy Model F1 Score: {}\n\tBest Accuracy Model Accuracy: {}".format(
    outer_cv_config[best_f1_config]["best_f1_model"], outer_cv_config[best_f1_config]["best_f1_model_params"], outer_cv_config[best_f1_config]["best_f1_model_f1_score"], outer_cv_config[best_f1_config]["best_f1_model_accuracy"], outer_cv_config[best_accuracy_config]["best_accuracy_model"], outer_cv_config[best_accuracy_config]["best_accuracy_model_params"], outer_cv_config[best_accuracy_config]["best_accuracy_model_f1_score"], outer_cv_config[best_accuracy_config]["best_accuracy_model_accuracy"]))
print("Best Accuracy Model: {}\n\tBest Accuracy Model Parameters: {}\n\tBest Accuracy Model F1 Score: {}\n\tBest Accuracy Model Accuracy: {}\n\tBest F1 Model: {}\n\tBest F1 Model Parameters: {}\n\tBest F1 Model F1 Score: {}\n\tBest F1 Model Accuracy: {}".format(
    outer_cv_config[best_accuracy_config]["best_accuracy_model"], outer_cv_config[best_accuracy_config]["best_accuracy_model_params"], outer_cv_config[best_accuracy_config]["best_accuracy_model_f1_score"], outer_cv_config[best_accuracy_config]["best_accuracy_model_accuracy"], outer_cv_config[best_f1_config]["best_f1_model"], outer_cv_config[best_f1_config]["best_f1_model_params"], outer_cv_config[best_f1_config]["best_f1_model_f1_score"], outer_cv_config[best_f1_config]["best_f1_model_accuracy"]))
