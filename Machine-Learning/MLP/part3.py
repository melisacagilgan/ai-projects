import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt

# we load all the datasets of Part 3
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(
    open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)


# Multi-layer perceptron neural network
class MLPModel(nn.Module):
    def __init__(self, hidden_layer_number, hidden_layer_neurons, activation_function):
        super(MLPModel, self).__init__()
        self.hidden_layer_number = hidden_layer_number
        self.layer1 = nn.Linear(784, hidden_layer_neurons)
        self.layer2 = nn.Linear(hidden_layer_neurons, hidden_layer_neurons)
        self.layer3 = nn.Linear(hidden_layer_neurons, 10)
        self.activation_function = activation_function

    def forward(self, x):
        hidden_layer_output = self.activation_function(self.layer1(x))
        for _ in range(self.hidden_layer_number):
            hidden_layer_output = self.activation_function(
                self.layer2(hidden_layer_output))
        output_layer = self.layer3(hidden_layer_output)
        return output_layer


loss_function = nn.CrossEntropyLoss()
soft_max_function = torch.nn.Softmax(dim=1)


# Grid search for the best hyperparameters
def grid_search(nn_model, ITERATION, optimizer, config_number):
    confidence_scores = {'train_confidence': [], 'validation_confidence': []}
    train_results = {'model': nn_model, 'train_accuracy': [], 'train_loss': [
    ], 'validation_accuracy': [], 'validation_loss': [], 'iteration': []}
    means = {'train_loss': [], 'validation_loss': []}
    print("Grid search begins for the configuration %d" % config_number)
    for _ in range(10):
        minimum_validation_loss = 100000
        patience = 0
        print("------------------------------------")
        for iteration in range(1, ITERATION + 1):

            optimizer.zero_grad()
            predictions = nn_model(x_train)
            loss_value = loss_function(predictions, y_train)
            loss_value.backward()
            optimizer.step()

            with torch.no_grad():
                train_prediction = nn_model(x_train)
                # Calculating the training accuracy and loss
                train_loss = loss_function(train_prediction, y_train)
                train_probability_prediction = soft_max_function(
                    train_prediction)
                train_accuracy = (torch.sum(
                    (torch.argmax(train_probability_prediction, dim=1) == y_train).int()) / y_train.size(0)) * 100
                train_results['train_accuracy'].append(train_accuracy.item())
                train_results['train_loss'].append(train_loss.item())

                validation_prediction = nn_model(x_validation)
                # Calculating the validation accuracy and loss
                validation_loss = loss_function(
                    validation_prediction, y_validation)
                validation_probability_prediction = soft_max_function(
                    validation_prediction)
                validation_accuracy = (torch.sum((torch.argmax(
                    validation_probability_prediction, dim=1) == y_validation).int()) / y_validation.size(0)) * 100
                train_results['validation_accuracy'].append(
                    validation_accuracy.item())
                train_results['validation_loss'].append(validation_loss.item())

            if iteration % 10 == 0:
                print("Iteration : %d   Mean Train Loss : %f   Mean Train Accuracy : %f   Mean Validation Loss : %f   Mean Validation Accuracy : %f" % (
                    iteration, np.mean(train_results['train_loss']), np.mean(train_results['train_accuracy']), np.mean(train_results['validation_loss']), np.mean(train_results['validation_accuracy'])))

            # Threshold for early stopping
            threshold = 0.0001
            # Early stopping condition
            if validation_loss.item() + threshold < minimum_validation_loss:
                # Update the minimum validation loss
                minimum_validation_loss = validation_loss.item()
                # Reset the patience
                patience = 0

            else:
                patience += 1
                if patience == 10:
                    print("Early stopping at iteration %d" % iteration)
                    train_results['iteration'].append(iteration)
                    break

        # Calculate the confidence interval for the validation set
        train_accuracy_mean = np.mean(train_results['train_accuracy'])
        train_accuracy_std = np.std(train_results['train_accuracy'])
        validation_accuracy_mean = np.mean(
            train_results['validation_accuracy'])
        validation_accuracy_std = np.std(train_results['validation_accuracy'])
        confidence_scores['train_confidence'].append([train_accuracy_mean - 1.96 * train_accuracy_std / np.sqrt(len(
            train_results['train_accuracy'])), train_accuracy_mean + 1.96 * train_accuracy_std / np.sqrt(len(train_results['train_accuracy']))])
        confidence_scores['validation_confidence'].append([validation_accuracy_mean - 1.96 * validation_accuracy_std / np.sqrt(len(
            train_results['validation_accuracy'])), validation_accuracy_mean + 1.96 * validation_accuracy_std / np.sqrt(len(train_results['validation_accuracy']))])

        # Calculate the mean of the training and validation losses
        means['train_loss'].append(np.mean(train_results['train_loss']))
        means['validation_loss'].append(
            np.mean(train_results['validation_loss']))

    # Plot the training and validation losses for each run
    plot_grid_search_results(means, config_number)

    return confidence_scores, train_results


# Testing the model
def test_model(nn_model):
    with torch.no_grad():
        test_prediction = nn_model(x_test)
        test_loss = loss_function(test_prediction, y_test)
        test_probability_prediction = soft_max_function(test_prediction)
        test_accuracy = (torch.sum((torch.argmax(
            test_probability_prediction, dim=1) == y_test).int()) / y_test.size(0)) * 100

        return test_accuracy.item(), test_loss.item()


# Training the model
def train_model(nn_model, ITERATION, optimizer):
    x_train_validation = torch.cat((x_train, x_validation), 0)
    y_train_validation = torch.cat((y_train, y_validation), 0)
    print("------------------------------------\nTraining the model")
    accuracy = []
    loss = []
    for i in range(10):
        print("Iteration : %d begins..." % (i + 1))
        for iteration in range(1, ITERATION + 1):

            optimizer.zero_grad()
            predictions = nn_model(x_train_validation)
            loss_value = loss_function(predictions, y_train_validation)
            loss_value.backward()
            optimizer.step()

            with torch.no_grad():
                train_prediction = nn_model(x_train_validation)
                # Calculating the training accuracy and loss
                train_loss = loss_function(
                    train_prediction, y_train_validation)
                train_probability_prediction = soft_max_function(
                    train_prediction)
                train_accuracy = (torch.sum(
                    (torch.argmax(train_probability_prediction, dim=1) == y_train_validation).int()) / y_train_validation.size(0)) * 100

                # Printing the training accuracy and loss for each 50 iteration
                if iteration % 50 == 0:
                    print("Iteration : %d - Training Loss : %f - Training Accuracy : %f" %
                          (iteration, train_loss.item(), train_accuracy.item()))

        # Calculating the confidence interval for the test set
        test_accuracy, test_loss = test_model(nn_model)
        accuracy.append(test_accuracy)
        loss.append(test_loss)

    test_accuracy_mean = np.mean(accuracy)
    test_accuracy_std = np.std(accuracy)
    confidence_score = ([test_accuracy_mean - 1.96 * test_accuracy_std / np.sqrt(len(accuracy)),
                        test_accuracy_mean + 1.96 * test_accuracy_std / np.sqrt(len(accuracy))])
    test_loss_mean = np.mean(loss)
    return accuracy, test_accuracy_mean, confidence_score, loss, test_loss_mean


# Plotting the mean of both train and validation losses over 10 runs
def plot_grid_search_results(mean_loss, config_number):
    iteration = range(1, 11)
    plt.figure(figsize=(10, 5))
    plt.plot(iteration, mean_loss['train_loss'], label='Train Loss')
    plt.plot(iteration, mean_loss['validation_loss'],
             label='Validation Loss')
    plt.xticks(iteration)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over 10 Runs (Configuration %d)' % config_number)
    plt.savefig('Configuration %d Loss.png' % config_number)
    print("Configuration %d Loss.png saved" % config_number)


# Hyperparameters
iteration = 250
learning_rates = [0.001, 0.0001]
hidden_layer_numbers = [1, 2]
hidden_layer_neurons = [24, 48]
activation_functions = [nn.Sigmoid(), nn.Tanh()]

results = []
config_number = 1
for neuron in hidden_layer_neurons:
    for activation_fuction in activation_functions:
        for hidden_layer_number in hidden_layer_numbers:
            nn_model = MLPModel(hidden_layer_number, neuron,
                                activation_fuction)
            for lr in learning_rates:
                optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
                confidence_scores, result = grid_search(
                    nn_model, iteration, optimizer, config_number)
                result['accuracy_confidence_interval'] = confidence_scores
                result['mean_training_loss'] = np.mean(result['train_loss'])
                result['mean_training_accuracy'] = np.mean(
                    result['train_accuracy'])
                result['mean_validation_loss'] = np.mean(
                    result['validation_loss'])
                result['mean_validation_accuracy'] = np.mean(
                    result['validation_accuracy'])
                result['hidden_layer_number'] = hidden_layer_number
                result['hidden_layer_neurons'] = neuron
                result['activation_function'] = activation_fuction
                result['learning_rate'] = lr
                result['optimizer'] = optimizer
                results.append(result)

                config_number += 1

# Print the results of the hyperparameter tuning
minimum_validation_loss = 100000
maximum_mean_validation_accuracy = 0
print("------------------------------------\nHyperparameter tuning results")
for i, result in enumerate(results):
    print("Configuration %d" % (i + 1))
    print("Hidden layer number : %d - Hidden layer neurons : %d - Activation function : %s - Learning rate : %f - Mean training loss : %f - Mean training accuracy : %f - Training confidence interval : %s - Mean training loss : %f - Mean validation accuracy : %f - Validation confidence interval : %s Mean validation loss : %f" %
          (result['hidden_layer_number'], result['hidden_layer_neurons'], result['activation_function'], result['learning_rate'], result['mean_training_loss'], result['mean_training_accuracy'], result['accuracy_confidence_interval']['train_confidence'], result['mean_validation_loss'], result['mean_validation_accuracy'], result['accuracy_confidence_interval']['validation_confidence'], result['mean_validation_loss']))
    if maximum_mean_validation_accuracy < result['mean_validation_accuracy']:
        maximum_mean_validation_accuracy = result['mean_validation_accuracy']
        best_model = result

print("------------------------------------\nBest hyperparameters")
print("Iteration : %d - Hidden layer number : %d - Hidden layer neurons : %d - Activation function : %s - Learning rate : %f - Mean training accuracy : %f - Training confidence interval : %s - Mean training loss : %f - Mean validation accuracy : %f - Validation confidence interval : %s Mean validation loss : %f" % (iteration,
      best_model['hidden_layer_number'], best_model['hidden_layer_neurons'], best_model['activation_function'], best_model['learning_rate'], best_model['mean_training_accuracy'], best_model['accuracy_confidence_interval']['train_confidence'], best_model['mean_training_loss'], best_model['mean_validation_accuracy'], best_model['accuracy_confidence_interval']['validation_confidence'], best_model['mean_validation_loss']))
test_accuracy, test_accuracy_mean, confidence_score, test_loss, test_loss_mean = train_model(
    best_model['model'], iteration, best_model['optimizer'])
print("------------------------------------\nTest Results")
print("Test accuracies : %s - Test accuracy mean : %f - Test accuracy confidence interval : %s - Test losses : %s - Test loss mean : %f" %
      (test_accuracy, test_accuracy_mean, confidence_score, test_loss, test_loss_mean))
