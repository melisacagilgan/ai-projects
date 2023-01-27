from Distance import Distance
import numpy as np


class KMeansPlusPlus:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""

        # Calculate the distance between each data point and its cluster center
        loss = 0
        for k in range(self.K):
            for s in self.clusters[k]:
                loss += Distance.calculateMinkowskiDistance(
                    s, self.cluster_centers[k], 2)
        return loss

    def run(self):
        """Kmeans++ algorithm implementation"""

        # Initialize the first cluster center with a random data point from the dataset
        self.cluster_centers[0] = self.dataset[np.random.randint(
            0, len(self.dataset))]

        # For the remaining K-1 cluster centers, choose the data points with the highest probability of being selected as cluster centers based on the distance to the nearest cluster center
        for k in range(1, self.K):
            distances = []
            for s in self.dataset:
                distances.append(min([Distance.calculateMinkowskiDistance(
                    s, self.cluster_centers[i], 2) for i in range(k)]))
            distances = np.array(distances)
            distances = distances**2
            distances = distances/np.sum(distances)
            self.cluster_centers[k] = self.dataset[np.random.choice(
                len(self.dataset), p=distances)]

        # For each data point, find the closest cluster center and assign it to that cluster
        while True:
            # For each data point, find the closest cluster center and assign it to that cluster
            distances = np.linalg.norm(self.dataset[:, np.newaxis] - self.cluster_centers, axis=2)
            closest_clusters = np.argmin(distances, axis=1)
            self.clusters = {i:[] for i in range(self.K)}
            for i, c in enumerate(closest_clusters):
                self.clusters[c].append(self.dataset[i])

            # update the cluster centers with the mean of the data points in the cluster
            new_centers = np.array([np.mean(self.clusters[c], axis=0) for c in range(self.K)])

            # check for convergence
            if np.allclose(self.cluster_centers, new_centers, rtol=1e-4):
                break
            self.cluster_centers = new_centers

        return self.cluster_centers, self.clusters, self.calculateLoss()

