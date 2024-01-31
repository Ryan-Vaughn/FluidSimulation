"""
Testing distributors
"""
import numpy as np
import pytest
import cells
import distributors

@pytest.fixture(name='grid_dataset')
def fixture_grid_dataset():
    """
    Generates a data set consisting of one point in each of a 3x3 grid.
    """
    cell_type = cells.FluidSimCell
    eps = .3
    bounds = np.array([1,1])
    meta_data = (cell_type,eps,bounds)

    x = eps * np.array([[i,j] for i in range(1,4) for j in range(1,4)])
    noise  = eps / 2 * np.random.rand(*x.shape)
    x = x + noise
    v = np.zeros(x.shape)
    masses = np.ones(x.shape[0])
    physical_data = (x,v,masses)

    return physical_data, meta_data

@pytest.fixture(name='sph2d_distributor')
def fixture_sph2d_distributor(grid_dataset):
    """
    Generates a 2D SPH distributor using the grid dataset.
    """
    physical_data, meta_data = grid_dataset
    dist = distributors.SPHDistributor(physical_data,meta_data)
    return dist

def test_sort_particles_unique(sph2d_distributor):
    """
    A test that checks that the correct neighboring cells are displayed.

    A 3*eps by 3*eps uniform grid of points has random noise applied that
    has norm less than eps. These points should always map to the original
    3 by 3 grid.

    """
    targets_coord = np.array([[i,j] for i in range(1,4) for j in range(1,4)]).astype(int)
    a = targets_coord[:,0]
    b = targets_coord[:,1]
    targets = ((a + b) * (a + b + 1) / 2 + b).astype(int)
    targets = np.sort(targets)
    outputs = sph2d_distributor.c.nonempty_cells_id

    assert np.array_equal(outputs,targets)

def test_sort_neighbor_particles_unique(sph2d_distributor):
    """
    A test that checks that the correct unique occupied neighbors are 
    returned.

    A 3*eps by 3*eps uniform grid of points has random noise applied that
    has norm less than eps. These points should always map to the original
    3 by 3 grid, and the unique neighboring cells should be the 5x5 grid.

    """
    targets_coord = np.array([[i,j] for i in range(0,5) for j in range(0,5)]).astype(int)
    a = targets_coord[:,0]
    b = targets_coord[:,1]
    targets = ((a + b) * (a + b + 1) / 2 + b).astype(int)
    targets = np.sort(targets)

    outputs = sph2d_distributor.n.nonempty_cells_id
    assert np.array_equal(outputs,targets)

def test_sort_particles_quantity(sph2d_distributor):
    """
    Checks that the correct cells are populated and the correct number of
    particles are included in each cell that is populated.
    """
    ids_targets = np.array([4,7,8,11,12,13,17,18,24])
    nums_targets = np.ones(9)

    ids_outputs = np.sort(sph2d_distributor.c.nonempty_cells_id)
    nums_outputs = np.array([sph2d_distributor.id_cells_dict[i].c.x.shape[0]
                             for i in ids_targets])

    correct_ids = np.array_equal(ids_outputs,ids_targets)
    correct_nums = np.array_equal(nums_outputs,nums_targets)
    print(nums_outputs)
    assert correct_ids and correct_nums
