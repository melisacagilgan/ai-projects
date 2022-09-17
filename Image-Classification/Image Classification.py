import glob
import os
import shutil
import numpy as np
import cv2
from os import path
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt


# Reading images from the dataset and splitting them into folders
def read_images():
    images = []
    # Reading images from the dataset
    for image in glob.glob("*.jpg") + glob.glob("*.jpeg"):
        images.append(image)
    np.array(images)

    # Splitting dataset into training, validation and test set
    train, valid, test = split_dataset(images)

    # Creating directories for each class in the dataset
    make_directory("TrainingSet", train)
    make_directory("ValidationSet", valid)
    make_directory("TestSet", test)


# Splitting dataset into training(%75), validation(%15) and test set(%10)
def split_dataset(images):
    train_valid, test = train_test_split(images, test_size=0.1)
    train, valid = train_test_split(train_valid, test_size=0.15)
    return train, valid, test


# Creating directories for each class in the dataset
def make_directory(dir_name, image_set):
    os.mkdir(dir_name)
    for img in image_set:
        shutil.move(img, dir_name + '/' + img)
    print("Directory " + dir_name + " created and " +
          str(len(image_set)) + " images moved to it.")


# Getting images in the given directory
def get_images(folder_name):
    images = []
    for image in glob.glob(folder_name + '/' + "*.jpg") + glob.glob(folder_name + '/' + "*.jpeg"):
        images.append(image)
    return np.array(images, dtype="object")


# Getting labels from each image in the dataset
def get_label(image):
    label = ''.join([i for i in image if not i.isdigit()])
    label = label.removeprefix("TrainingSet\\")
    if label.endswith(".jpg"):
        label = label.removesuffix(".jpg")
    elif label.endswith(".jpeg"):
        label = label.removesuffix(".jpeg")
    return label


# Getting pixels and labels from each image in the dataset and converting them to numpy array
def get_features_labels(image_set):
    pixels = []
    labels = []
    label_dict = {}
    count = 0
    fixed_size = (28, 28)
    for image in image_set:
        try:
            img = cv2.imread(image)
            img = cv2.resize(img, fixed_size)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pixels.append(gray_img)

            label = get_label(image)
            if label not in label_dict:
                label_dict[label] = count
                count += 1
            labels.append(label_dict[label])
        except:
            print("Error reading image: " + image)
    pixels = np.array(pixels)
    # Normalizing the pixels
    pixels = pixels.astype('float32') / 255.0
    return pixels, np.array(labels), count


# Training the model with the given number of hidden layers
def classifier_model(train_set, train_labels, batch_size, label_no, valid_set, valid_labels, input_shape, hidden_layer_no):
    if path.exists('cnn_model' + str(hidden_layer_no) + '.h5'):
        print("Model already trained.")
        classifier = load_model('cnn_model' + str(hidden_layer_no) + '.h5')
    else:
        # Creating the model
        classifier = Sequential()

        # Adding the flatten layer
        classifier.add(Flatten())

        # Adding fully connected input layer
        unit = 512
        classifier.add(
            Dense(units=unit, activation="relu", input_shape=input_shape))

        # Adding fully connected hidden layers
        for i in range(hidden_layer_no):
            unit /= 2
            # Adding hidden layers with the number of units equal to the half of the previous layer
            classifier.add(Dense(units=unit, activation="relu"))

        # Adding output layer with the number of classes
        classifier.add(Dense(units=label_no, activation="softmax"))

        # Compiling the CNN
        classifier.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        model_name = 'cnn_model' + str(hidden_layer_no) + '.h5'

        # Saving the best model
        callbacks = [ModelCheckpoint(
            model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

        # Fitting the CNN to the images
        history = classifier.fit(train_set, train_labels, epochs=40,
                                 batch_size=batch_size, validation_data=(valid_set, valid_labels), callbacks=callbacks)

        print("Model with " + str(hidden_layer_no) +
              " hidden layers saved as " + model_name)

        plot_history(history, "HistoryWith" +
                     str(hidden_layer_no) + "HiddenLayer(s).png")

    # Evaluating the model on the validation set
    _, accuracy = classifier.evaluate(valid_set, valid_labels)

    return accuracy


# Plotting the training and validation loss and accuracy over epochs
def plot_history(history, file_name):
    # Plotting the loss of the model
    plt.plot(history.history['loss'], 'r', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.savefig("Loss" + file_name)
    plt.clf()

    # Plotting the accuracy of the model
    plt.plot(history.history['accuracy'], 'g', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'],
             'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.savefig("Accuracy" + file_name)
    plt.clf()


def main():
    train_dir, valid_dir, test_dir = "TrainingSet", "ValidationSet", "TestSet"

    if not (path.exists(train_dir) and path.exists(valid_dir) and path.exists(test_dir)):
        read_images()
    else:
        print("Dataset already splitted.")
    training_set, validation_set, test_set = get_images(
        train_dir), get_images(valid_dir), get_images(test_dir)

    input_shape = (28, 28, 1)

    train_features, train_labels, label_no = get_features_labels(training_set)
    valid_features, valid_labels, _ = get_features_labels(validation_set)
    test_features, test_labels, _ = get_features_labels(test_set)

    best_accuracy = 0.0
    best_hidden_layer_no = 1
    best_model = None

    # Training and validating the model with 3 different configurations of hidden layers
    for hidden_layer_no in range(1, 4):
        accuracy = classifier_model(train_features, train_labels, int((len(training_set)-1)/3), label_no,
                                    valid_features, valid_labels, input_shape, hidden_layer_no)
        # Getting the highest accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hidden_layer_no = hidden_layer_no
            best_model_name = 'cnn_model' + str(hidden_layer_no) + '.h5'

    best_model = load_model(best_model_name)
    results = best_model.evaluate(test_features, test_labels)
    print("Testing Results:\n" + "  Error Rate: %.2f%%\n" %
          (results[0]*100) + "  Accuracy: %.2f%%\n" % (results[1]*100) + "  Hidden Layers: " + str(best_hidden_layer_no))


if __name__ == "__main__":
    main()
