"""
Hacked together benchmarking to test the effectiveness of the data structure
for pairwise distance computations.
"""
import time
import numpy as np
import cells
import distributors
from functools import partial


NUM_PTS = 10000
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

dist.distribute_procedure(cells.CellSPH2D.compute_distances,())
dist.distribute_procedure(cells.CellSPH2D.compute_density_kernel,())
dist.distribute_procedure(cells.CellSPH2D.compute_densities,())

A = np.ones((NUM_PTS,2))
B = 2* np.ones((NUM_PTS,2))
C = 3* np.ones(NUM_PTS)

physical_data = (A,B,C)
dist.distribute_method(cells.CellSPH2D.populate,physical_data)