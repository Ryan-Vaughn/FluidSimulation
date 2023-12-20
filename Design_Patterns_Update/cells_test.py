import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from cells import Cell, CellSPH2D


def test_populate():
    """
    Tests if a 2d dataset of 100 random points can be passed into cell memory.
    """
    num_pts = 100
    cell = CellSPH2D()
    x_c = np.random.randn(num_pts,2)
    v_c = np.random.randn(num_pts,2)
    masses_c = np.random.randn(num_pts)
    cell.populate(x_c,v_c,masses_c)
    assert (cell.x_c, cell.v_c,cell.masses_c) == (x_c, v_c,masses_c)

def test_populate_neighbors():
    """
    Tests if a 2d dataset of 100 random points can be passed into cell 
    neighbors memory.
    """
    num_pts = 100
    cell = CellSPH2D()
    x_n = np.random.randn(num_pts,2)
    v_n = np.random.randn(num_pts,2)
    masses_n = np.random.randn(num_pts)
    cell.populate_neighbors(x_n,v_n,masses_n)
    assert (cell.x_n, cell.v_n,cell.masses_n) == (x_n, v_n,masses)    

def test_compute_distances():
    """
    Tests a simple dataset for the correct pairwise distances.
    """
    cell = CellSPH2D()
    cell.x_c = np.array([[0,0]])
    cell.x_n = np.array([[0,0],[0,1],[1,1],[1,0]])
    distances = np.array([[0,1, np.sqrt(2),1]])
    cell.compute_distances()
    assert np.array_equal(cell.distances,distances)

def test_compute_densities():
    """
    
    """
    pass

def test_compute_density_kernel(self):
    """
    
    """
    pass

def compute_densities(self):
    """
     
    """
    pass

def get_densities(self):
    """
    
    """
    pass

def set_neighbor_densities(self,densities_n):
    """
    
    """
    pass