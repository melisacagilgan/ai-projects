import numpy as np
import math
from ID3 import DecisionTree


features = ["Temperature", "Outlook", "Humidity", "Windy"]
# Golf played?...
labels = [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1]
dataset = [
    ["hot", "sunny", "high", "false"],
    ["hot", "sunny", "high", "true"],
    ["hot", "overcast", "high", "false"],
    ["cool", "rain", "normal", "false"],
    ["cool", "overcast", "normal", "true"],
    ["mild", "sunny", "high", "false"],
    ["cool", "sunny", "normal", "false"],
    ["mild", "rain", "normal", "false"],
    ["mild", "sunny", "normal", "true"],
    ["mild", "overcast", "high", "true"],
    ["hot", "overcast", "normal", "false"],
    ["mild", "rain", "high", "true"],
    ["cool", "rain", "normal", "true"],
    ["mild", "rain", "high", "false"]]


dt = DecisionTree(dataset, labels, features, "gain ratio")
dt.train()
correct = 0
wrong = 0
for data_index in range(len(dataset)):
    data_point = dataset[data_index]
    data_label = labels[data_index]

    predicted = dt.predict(data_point)
    if predicted == data_label:
        correct += 1
    else:
        wrong += 1

print("Accuracy : %.2f" % (correct/(correct+wrong)*100))
