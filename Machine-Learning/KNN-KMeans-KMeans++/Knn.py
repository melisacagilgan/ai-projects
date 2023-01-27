class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def predict(self, instance):
        """ Predicts the class label of a single instance using Majority Voting method"""

        # Calculate distances between the instance and all other samples in the dataset
        distances = []
        for i in range(len(self.dataset)):
            if self.similarity_function_parameters is None:
                distance = self.similarity_function(instance, self.dataset[i])
            else:
                distance = self.similarity_function(
                    instance, self.dataset[i], self.similarity_function_parameters)
            distances.append((self.dataset_label[i], distance))

        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[1])

        # Get the K nearest neighbors
        neighbors = []
        for i in range(self.K):
            neighbors.append(distances[i][0])
        return max(set(neighbors), key=neighbors.count)
