import matplotlib.pyplot as plt
import numpy as np
################################################################################
def plot_cluster(clusters:'np.array', targets:'np.array',
    alpha=0.7, figure_size:'tuple'=(20,20)):

    labels = np.unique(targets)

    clusters_dictionary = {}

    for label in labels:
        clusters_dictionary[f'{label}'] = clusters[targets==label]

    for key, value in clusters_dictionary.items():
        plt.scatter(value[:, 1], value[:, 0], alpha=alpha, label=key)

    plt.legend()


    pass
################################################################################
################################################################################
