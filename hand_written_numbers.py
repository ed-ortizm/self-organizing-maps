#! /bin/env python3
from configparser import ConfigParser, ExtendedInterpolation
import time

import matplotlib.pyplot as plt
import numpy as np
import susi
from susi.SOMPlots import plot_umatrix
################################################################################
t0 = time.time()
################################################################################
# loading data from configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read('hand_written.ini')
    ############################################################################
# relevant directories
data_directory = parser.get('directories', 'data')
    ############################################################################
# get data
digits = np.load(f"{data_directory}/digits.npy")
labels = np.load(f"{data_directory}/digits_target.npy")
    ############################################################################
# parameters for SOM
number_rows = parser.getint('parameters', 'rows')
number_columns = parser.getint('parameters', 'columns')
################################################################################
# training self organizing map
som = susi.SOMClustering(n_rows=number_rows, n_columns=number_columns)
som.fit(digits)

# spec ... som ... bmu ... som.T... (regressor to get rec)
# sklearn for shape parameter, remove that checking
################################################################################
# inspecting trained model
u_matrix = som.get_u_matrix()
np.save((f"{data_directory}/"
    f"u_matrix_{number_rows}x{number_columns}.npy"), u_matrix)
# plot_umatrix(u_matrix, number_rows, number_columns)
# plt.show()
    ############################################################################
################################################################################
################################################################################
t1 = time.time()
print(f"running time {t1 -t0}")
