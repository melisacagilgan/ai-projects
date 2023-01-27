from Part2.KMeansPlusPlus import KMeansPlusPlus
import pickle
import numpy as np
import matplotlib.pyplot as plt


dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
print("-----------------------------------\nDataset1 is loaded! Shape: ", dataset1.shape)

dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))
print("Dataset2 is loaded! Shape: ", dataset2.shape)


k_values = [i for i in range(2, 11)]
config_no = 1
ds1_results = {}
ds2_results = {}

# Calculate each configuration's loss value for dataset1 and dataset2 and store them in ds1_results and ds2_results
for k in k_values:
    ds1_losses = []
    ds2_losses = []
    for _ in range(10):
        ds1_kmeans = KMeansPlusPlus(dataset1, k)
        ds2_kmeans = KMeansPlusPlus(dataset2, k)
        _, _, ds1_loss = ds1_kmeans.run()
        _, _, ds2_loss = ds2_kmeans.run()
        ds1_losses.append(ds1_loss)
        ds2_losses.append(ds2_loss)
    ds1_results[config_no] = {'k_value': k, 'minimum loss': np.min(ds1_losses), 'loss_confidence_interval': [np.mean(
        ds1_losses) - 1.96 * np.std(ds1_losses) / np.sqrt(len(ds1_losses)), np.mean(ds1_losses) + 1.96 * np.std(ds1_losses) / np.sqrt(len(ds1_losses))]}
    ds2_results[config_no] = {'k_value': k, 'minimum loss': np.min(ds1_losses), 'loss_confidence_interval': [np.mean(
        ds2_losses) - 1.96 * np.std(ds2_losses) / np.sqrt(len(ds2_losses)), np.mean(ds2_losses) + 1.96 * np.std(ds2_losses) / np.sqrt(len(ds2_losses))]}
    config_no += 1

# Print the results for dataset1
print("-----------------------------------\nDataset1 Results:")
for no, config in ds1_results.items():
    print("Config", no, "k =", config['k_value'], "loss =", config['minimum loss'],
          "loss_confidence_interval =", config['loss_confidence_interval'])

# Plot the results for dataset1
plt.figure(figsize=(10, 5))
plt.title("Dataset1 K vs Loss")
plt.xlabel("K")
plt.ylabel("Loss")
plt.plot([config['k_value'] for config in ds1_results.values()], [
         config['minimum loss'] for config in ds1_results.values()], 'o-')
plt.savefig("Dataset1 K vs Loss(KPP).png")
print("K vs Loss plot for Dataset1 is saved!")


# Print the results for dataset2
print("-----------------------------------\nDataset2 Results:")
for no, config in ds2_results.items():
    print("Config", no, "k =", config['k_value'], "loss =", config['minimum loss'],
          "loss_confidence_interval =", config['loss_confidence_interval'])

# Plot the results for dataset2
plt.figure(figsize=(10, 5))
plt.title("Dataset2 K vs Loss")
plt.xlabel("K")
plt.ylabel("Loss")
plt.plot([config['k_value'] for config in ds2_results.values()], [
         config['minimum loss'] for config in ds2_results.values()], 'o-')
plt.savefig("Dataset2 K vs Loss(KPP).png")
print("K vs Loss plot for Dataset2 is saved!")
