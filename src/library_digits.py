import matplotlib.pyplot as plt
import numpy as np
################################################################################
def plot_clusters(clusters:'np.array', targets:'np.array',
    alpha=0.7, figure_size:'tuple'=(20,20)):

    labels = np.unique(targets)

    clusters_dictionary = {}

    for label in labels:
        clusters_dictionary[f'{label}'] = clusters[targets==label]

    for key, value in clusters_dictionary.items():
        plt.scatter(value[:, 1], value[:, 0], alpha=alpha, label=key)

    plt.gca().invert_yaxis()

    plt.legend()
################################################################################
def images_from_archetypes(archetypes:'np.array'):

    if archetypes.ndim == 3:
        number_rows = archetypes.shape[0]
        number_columns = archetypes.shape[1]
        images = archetypes.reshape((number_rows, number_columns, 8, 8))

        return images

    images = archetypes.reshape((8,8))

    return images
################################################################################
