import numpy as np


# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}


# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label


class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None
        self.heuristic_function = self.calculate_information_gain__ if self.criterion == "information gain" else self.calculate_gain_ratio__

        # further variables and functions can be added...

    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        # Count the number of positive and negative examples
        positives = np.array([labels[i]
                             for i in range(len(labels)) if labels[i] == 1])
        p_positives = positives.shape[0] / np.array(dataset).shape[0]
        p_negatives = 1 - p_positives

        # If there are no positive or negative examples, entropy is zero, otherwise entropy is calculated
        if p_positives == 0:
            entropy_value = -p_negatives * np.log2(p_negatives)
        elif p_negatives == 0:
            entropy_value = -p_positives * np.log2(p_positives)
        else:
            entropy_value = -p_positives * np.log2(p_positives) - \
                p_negatives * np.log2(p_negatives)

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0

        dataset = np.array(dataset)

        # Get the unique values of the given attribute
        examples = np.array([dataset[i][attribute]
                            for i in range(len(dataset))])
        unique_examples = np.unique(examples)

        # Calculate the entropy for each unique value of the attribute and sum them up
        for s in unique_examples:
            new_dataset = []
            new_labels = []
            for i in range(len(dataset)):
                if dataset[i][attribute] == s:
                    new_dataset.append(dataset[i])
                    new_labels.append(labels[i])

            new_dataset = np.array(new_dataset)
            new_labels = np.array(new_labels)
            average_entropy += new_dataset.shape[0] / dataset.shape[0] * \
                self.calculate_entropy__(new_dataset, new_labels)

        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain = 0.0

        # Information gain of an attribute is calculated by subtracting the average entropy of the attribute from the entropy of the dataset
        information_gain = self.calculate_entropy__(dataset, labels) - \
            self.calculate_average_entropy__(dataset, labels, attribute)

        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = 0.0

        # Intrinsic information of an attribute is calculated by summing up the entropy of each unique value of the attribute
        dataset = np.array(dataset)
        examples = np.array([dataset[i][attribute]
                            for i in range(len(dataset))])
        unique_examples = np.unique(examples)
        for s in unique_examples:
            new_dataset = np.array([dataset[i] for i in range(
                len(dataset)) if dataset[i][attribute] == s])
            new_labels = np.array([labels[i] for i in range(
                len(labels)) if dataset[i][attribute] == s]).astype(int)

            intrinsic_info += self.calculate_entropy__(new_dataset, new_labels)

        return intrinsic_info

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        Calculates the gain ratio for a given attribute in the dataset.
        The gain ratio is a measure of the relative importance of an attribute in
        the decision tree, calculated as the ratio of the information gain of the attribute
        to the intrinsic information of the attribute.
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated
        :return: the calculated gain ratio score, or 0 if the information gain or intrinsic information is zero
        """
        information_gain = self.calculate_information_gain__(
            dataset, labels, attribute)
        intrinsic_info = self.calculate_intrinsic_information__(
            dataset, labels, attribute)

        # check if intrinsic info is not zero before dividing
        if intrinsic_info == 0:
            return 0
        # check if information gain is not zero before dividing
        elif information_gain == 0:
            return 0
        else:
            gain_ratio = information_gain / intrinsic_info
            return gain_ratio

    def calculate_gain_list__(self, dataset, labels, used_attributes):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :return: the calculated gain ratio score for each attribute as a dictionary
        """

        # Calculate the heuristic value for each attribute and store them in a dictionary if they are not used yet
        attribute_dict = {}
        for i in range(len(self.features)):
            if i not in used_attributes:
                attribute_dict[i] = self.heuristic_function(dataset, labels, i)

        # Sort the dictionary in descending order
        attribute_dict = sorted(attribute_dict.items(),
                                key=lambda x: x[1], reverse=True)

        # Return the index of the attribute with the highest heuristic value
        return attribute_dict[0][0]

    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """

        # If all the labels are the same, return a leaf node with that label
        if len(np.unique(labels)) == 1:
            return TreeLeafNode(dataset, labels[0])

        # If there are no more attributes to use, return a leaf node with the labels
        elif len(used_attributes) == len(self.features):
            return TreeLeafNode(dataset, labels)

        else:
            # Find the attribute with the highest heuristic value and create a non-leaf node with that attribute
            best_attribute = self.calculate_gain_list__(
                dataset, labels, used_attributes)

            node = TreeNode(best_attribute)

            # Recursively create subtrees for each unique value of the attribute and return the created node
            for s in np.unique(np.array(dataset)[:, best_attribute]):
                indices = np.array([])
                for i, data in enumerate(dataset):
                    if data[best_attribute] == s:
                        indices = np.append(indices, i)

                new_dataset = np.array([dataset[int(i)] for i in indices])
                new_labels = np.array([labels[int(i)] for i in indices])
                node.subtrees[s] = self.ID3__(
                    new_dataset, new_labels, used_attributes + [best_attribute])

            return node

    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array
        :return: predicted label of x

        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """

        predicted_label = None
        try:
            # Traverse the tree until a leaf node is reached and return the predicted label
            node = self.root
            while not isinstance(node, TreeLeafNode):
                node = node.subtrees[x[node.attribute]]
        except KeyError:
            return "Data not found"
        # If there are multiple labels in the leaf node, return the majority label, otherwise return the label
        if isinstance(node.labels, np.ndarray):
            predicted_label = np.bincount(node.labels.astype(int)).argmax()
        else:
            predicted_label = node.labels

        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")
