"""
Hacked together benchmarking to test the effectiveness of the data structure
for pairwise distance computations.
"""
import numpy as np
import pytest
import cells
import distributors
import time
from scipy.spatial.distance import cdist


NUM_PTS = 40000
DIM = 2
AMPLITUDE = 10
EPS = .9

x = AMPLITUDE * np.random.randn(NUM_PTS,DIM)

min_x = np.min(x[:,0])
max_x = np.max(x[:,0])
min_y = np.min(x[:,1])
max_y = np.max(x[:,1])

# Ensure the data is within
x[:,0] = x[:,0] + np.abs(min_x) + 1
x[:,1] = x[:,1] + np.abs(min_y) + 1

bounds = np.array([max_x + np.abs(min_x) + 10, max_y + np.abs(min_y) + 10])

v = np.zeros(x.shape)
masses = np.ones(NUM_PTS)
start_setup_time = time.time()
dist = distributors.DistributorSPH2D((x,v,masses),(cells.CellSPH2D,EPS,bounds))
end_setup_time = time.time()

start_dist_time = time.time()
dist.distribute_computation(cells.CellSPH2D.compute_distances,*[])
end_dist_time = time.time()

start_brute_time = time.time()
distances = cdist(x,x)
end_brute_time = time.time()
setup_time = end_setup_time - start_setup_time
dist_time = end_dist_time - start_dist_time
brute_time = end_brute_time - start_brute_time

print("Setup:" + str(setup_time))
print("Distributed:" + str(dist_time))
print("Brute Force:" + str(brute_time))
