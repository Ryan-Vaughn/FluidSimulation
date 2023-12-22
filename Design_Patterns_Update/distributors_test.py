import pytest
import numpy as np
from distributors import DistributorSPH2D
from cells import CellSPH2D

@pytest.fixture
def cell_generator():
    _num_pts = 10
    _dim = 2
    x = np.random.randn(_num_pts,_dim)
    v = np.zeros(x.shape)
    masses = np.ones(x.shape[0])
    bounds = [[0,2][0,2]]
    eps = .3
    cell_type = CellSPH2D

    return 
