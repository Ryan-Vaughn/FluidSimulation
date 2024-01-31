"""
Testing cells
"""
import numpy as np
from cells import FluidSimCell
import pytest

@pytest.fixture(name='ring_dataset')
def fixture_ring_dataset():
    """
    Generates a single center point with a unit circle around the center
    point. Used to check symmetry of various forces/quantities.
    """
    _num_pts = 12
    _dim = 2
    _thetas  = 2 * np.pi * np.arange(0,1,1/_num_pts)
    _max_bounds = 3
    _eps = .3
    x = np.zeros((_num_pts + 1,_dim))
    x[1:,0] = _eps / 2 * np.cos(_thetas) + _max_bounds/2
    x[1:,1] = _eps / 2 * np.sin(_thetas) + _max_bounds/2
    x[0,:] = _max_bounds / 2 * np.ones(2)
    v = np.zeros(x.shape)
    masses = np.ones(_num_pts)

    return (x,v,masses)

@pytest.fixture(name='cell')
def fixture_cell():
    """
    Creates a SPH2D cell with eps = .3.
    """
    eps = .3
    return FluidSimCell(eps)

def test_populate(ring_dataset, cell):
    """
    Tests if a 2d dataset of 100 random points can be passed into cell memory.
    """
    x_c,v_c,masses_c = ring_dataset
    cell.populate(x_c,v_c,masses_c)
    assert (cell.c.x, cell.c.v, cell.c.masses) == ring_dataset

def test_populate_neighbors(ring_dataset,cell):
    """
    Tests if a 2d dataset of 100 random points can be passed into cell 
    neighbors memory.
    """
    x_n,v_n,masses_n = ring_dataset
    cell.populate(x_n,v_n,masses_n)
    assert (cell.n.x, cell.n.v, cell.n.masses) == ring_dataset

def test_compute_distances():
    """
    Tests a simple dataset for the correct pairwise distances.
    """


def test_compute_densities():
    """
    Tests a simple dataset for the correct densities.
    """


def test_compute_density_kernel():
    """
    Tests a simple dataset getting the correct kernel.
    """
