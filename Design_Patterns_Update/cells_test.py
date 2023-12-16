import numpy as np
import numpy.typing as npt

from cells import CellSPH2D

def test_can_populate():
    num_pts = 100
    cell = CellSPH2D()
    x_c = np.random.randn(num_pts,2)
    v_c = np.random.randn(num_pts,2)
    cell.populate(x_c,v_c)
    assert (cell.x_c, cell.v_c) == (x_c, v_c)

def test_can_populate_neighbors():
    num_pts = 100
    cell = CellSPH2D()
    x_c = np.random.randn(num_pts,2)
    v_c = np.random.randn(num_pts,2)
    cell.populate(x_n,v_n)
    assert (cell.x_n, cell.v_n) == (x_n, v_n)    