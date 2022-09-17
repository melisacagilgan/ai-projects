import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def main():
    # Folder paths
    train_folder = "TrainingSet"
    valid_folder = "ValidationSet"
    test_folder = "TestSet"

    # Feature extraction
    train_features, train_labels = read_file(train_folder)
    valid_features, valid_labels = read_file(valid_folder)
    test_features, test_labels = read_file(test_folder)

    k_list = [1, 3, 5, 7, 9]
    valid_dict = {}
    valid_acc = []
    for k_val in k_list:
        # Validation
        _, valid_accuracy = training_and_validation(
            train_features, train_labels, valid_features, valid_labels, k_val)
        valid_dict[k_val] = valid_accuracy
        valid_acc.append(valid_accuracy)

    print("Validation Phase:")
    for k_val in valid_dict:
        print("K:", k_val, "  Accuracy: {:.2f}".format(valid_dict[k_val]))

    # Plotting the graph of k values vs validation accuracy
    plt.scatter(k_list, valid_acc, c='purple', marker='o')
    plt.title("K-Values vs. Validation Accuracy")
    plt.xlabel("K-Values")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.show()

    best_configuration = max(valid_dict, key=lambda k_val: valid_dict[k_val])
    knn_classifier, _ = training_and_validation(
        train_features, train_labels, valid_features, valid_labels, best_configuration)

    # Testing
    test_accuracy, predictions, match_count = testing(
        knn_classifier, test_features, test_labels)

    print("\nTesting Phase\nK:", best_configuration,
          " Accuracy: {:.2f}".format(test_accuracy), " Correct Match:", match_count)
    print("\nTesting Performance:")
    for index, prediction in enumerate(predictions):
        print("Image", (index+1), "\tPredicted:",
              prediction, "\tActual:", test_labels[index])


def read_file(folder_path):
    # Reading images and labels into "images" and "labels" arrays
    images = []
    labels = []
    features = []
    for folders in os.listdir(folder_path):
        for image in os.listdir(folder_path + '/' + folders):
            # Checking if the image ends with jpg
            if image.endswith(".jpg"):
                img = cv2.imread(
                    folder_path + '/' + folders + '/' + image)
                images.append(img)
                # Getting labels of each image
                labels.append(folders)
    features = apply_sift(images)
    return np.array(features), np.array(labels)


def apply_sift(images):
    descriptors = []
    # Applying SIFT(scale-invariant feature transform) as a feature extraction method
    for image in images:
        # Initializing SIFT object with a feature limit to retain the number of best features
        # nfeatures --> The features are ranked by their scores
        sift = cv2.SIFT_create(nfeatures=250)
        # Detecting keypoints & descriptors
        _, des = sift.detectAndCompute(image, None)
        des_mean = np.mean(des, axis=0)
        descriptors.append(des_mean)
    return descriptors


def training_and_validation(train_features, train_labels, valid_features, valid_labels, k_val):
    # n_jobs -> Number of jobs to run in parallel and -1 is for using all threads
    knn_classifier = KNeighborsClassifier(n_neighbors=k_val, n_jobs=-1)
    knn_classifier.fit(train_features, train_labels)
    accuracy = knn_classifier.score(valid_features, valid_labels)

    return knn_classifier, accuracy


def testing(knn_classifier, test_features, test_labels):
    score = 0
    label_number = len(test_labels)
    predictions = knn_classifier.predict(test_features)
    for index, prediction in enumerate(predictions):
        # To calculate the overall accuracy
        if prediction == test_labels[index]:
            score += 1
    test_accuracy = score / label_number
    return test_accuracy, predictions, score


if __name__ == "__main__":
    main()
