import pickle
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


dataset = pickle.load(open("../data/part3_dataset.data", "rb"))
print("Dataset is loaded! Shape: ", dataset.shape)

k_values = [2, 3, 4, 5]
linkage_methods = ['single', 'complete']
distance_metrics = ['euclidean', 'cosine']
results = {}
config_no = 1

# Calculate each configuration's silhouette score for dataset and store them in results
for k in k_values:
    for linkage_ in linkage_methods:
        for distance in distance_metrics:
            clustering = AgglomerativeClustering(
                n_clusters=k, linkage=linkage_, affinity=distance).fit(dataset)
            labels = clustering.labels_
            s_score = silhouette_score(dataset, labels, metric=distance)
            results[config_no] = {'k': k, 'linkage': linkage_,
                                  'distance': distance, 'silhouette_score': s_score}

            # Plot the silhouette score for each configuration
            plt.figure()
            plt.title("Config " + str(config_no) +
                      " - " + linkage_ + " - " + distance)
            plt.xlabel("Silhouette score")
            plt.bar(1, s_score)
            plt.show()
            plt.clf()

            # Plot the dendrogram for each configuration
            Z = linkage(dataset, linkage_, distance)
            dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
            plt.title("Config " + str(config_no) +
                      " - " + linkage_ + " - " + distance)
            plt.show()

            config_no += 1

# Print the results
max_s_score = 0
print("-----------------------------------\nResults:")
for no, config in results.items():
    s_score = "{:.5f}".format(config['silhouette_score'])
    print(
        f"Config no: {no} - k: {config['k']}, linkage: {config['linkage']}, distance: {config['distance']}, silhouette_score: {s_score}")
    if config['silhouette_score'] > max_s_score:
        max_s_score = config['silhouette_score']
        best_config_no = no
        best_config = config

# Print the best configuration
print("-----------------------------------\nBest config:")
s_score = "{:.3f}".format(best_config['silhouette_score'])
print(
    f"Config no: {best_config_no} - k: {best_config['k']}, linkage: {best_config['linkage']}, distance: {best_config['distance']}, silhouette_score: {s_score}")
