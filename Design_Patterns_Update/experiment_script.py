"""
Hacked together benchmarking to test the effectiveness of the data structure
for pairwise distance computations.
"""
import time
import numpy as np
import cells
import distributors


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

dist = distributors.SPHDistributor((x,v,masses),(cells.FluidSimCell,EPS,bounds))

PRESSURE_CONSTANT =1
REST_DENSITY = 0

densities,debug = dist.distribute_method(cells.FluidSimCell.compute_forces_step_1, returns=True)


pressures = PRESSURE_CONSTANT * (densities - REST_DENSITY)
pressures_duplicated = dist.duplicate(pressures)
pressures_n = dist.arrange(pressures_duplicated)

dist.distribute_method(cells.FluidSimCell.set_pressures,pressures)
dist.distribute_method(cells.FluidSimCell.set_pressures_n,pressures_n,domain='n',codomain='n')
