"""
Module for distributor classes. Distributors are in charge of allocating
particle data to different cells. The cells compute quantites which are
only dependent on local information (particles that are close to one 
another) and pass the data back to the distributor.

"""
from abc import ABC, abstractmethod
import dataclasses
import numpy as np
from numpy.typing import NDArray
from cells import Cell


@dataclasses.dataclass
class SPHInputData():
    """
    Data class to book keep all the internal data of the Distributor.

    One is used for the cell data, the other is for the neighboring cell data.
    """
    x : NDArray[np.float32] = None
    v : NDArray[np.float32] = None
    masses : NDArray[np.float32] = None

    x_id : NDArray[np.int_] = None
    x_g : NDArray[np.int_] = None
    cuts : NDArray[np.int_] = None
    nonempty_cells_id : NDArray[np.int_] = None
    sort_indices : NDArray[np.int_] = None

class Distributor(ABC):
    """
    Abstract class for Distributor

    Distributors break up whole datasets of particles and distribute them into
    a specified type of cell. When a cell method is called through the Manager,
    the Distributor collects the output of the cell method and stores it in a 
    manner consistent with the corresponding particles position in the whole dataset.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def distribute_computation(self,data_container,method,*args):
        """
        method for applying a cell method to all cells in the distributor.
        """

class DistributorSPH2D(Distributor):
    """
    Distributor for distributing computations into Cells.

    """

    def __init__(self,physical_data,meta_data):
        self.dim = 2
        self.num_pts, self.eps,self.bounds = meta_data

        self.id_cells_dict = self.initialize_lookup(*meta_data)
        # Data storage for cells
        self.c = SPHInputData()

        # Data storage for neighboring cells.
        self.n = SPHInputData()

        # Generate data for c and n depending on physical inputs.
        self.sort_particles(*physical_data).sort_neighbor_particles()

        # Populate cells with position, velocity, and mass data
        self.map_particles()
        self.map_neighbors()

    def initialize_lookup(self,cell_type,eps,bounds):
        """
        Procedure to create dictionaries which index the Cells in a grid.

        In the 2d SPH case, this is done by constructing a cell for each
        pair of integers inside of the given rectangular domain, then 
        applying a bijective map between the set of integers and the 
        set of pairs of integers, resulting in a unique integer id for 
        each cell.
        """

        grid_bounds = np.ceil(bounds/eps)
        grid_x_coords = range(-1,int(grid_bounds[0]) + 1)
        grid_y_coords = range(-1,int(grid_bounds[1]) + 1)

        id_cells_dict = {int((i + j) *(i + j + 1) / 2 + j) : cell_type(eps)
                                                for i in grid_x_coords
                                                for j in grid_y_coords}
        return id_cells_dict

    def sort_particles(self,x,v,masses):
        """
        Procedure which updates all the internal data of the cell given the
        input physical data.

        Specifically, each particle is assigned an id corresponding to a unique
        integer grid point. The physical quantites are then all sorted according
        to this id.
        """
        self.c.x = x
        self.c.v = v
        self.c.masses = masses

        self.c.x_g = np.rint(self.c.x / self.eps).astype(int)
        self.c.x_id = self.map_to_id(self.c.x_g)

        self.c.sort_indices = np.argsort(self.c.x_id)

        self.c.x = self.c.x[self.c.sort_indices,:]
        self.c.v = self.c.v[self.c.sort_indices,:]
        self.c.masses = self.c.masses[self.c.sort_indices]

        self.c.x_id = self.c.x_id[self.c.sort_indices]
        self.c.x_g = self.c.x_g[self.c.sort_indices,:]

        self.c.nonempty_cells_id = np.unique(self.c.x_id)
        self.c.cuts = np.searchsorted(self.c.x_id, self.c.nonempty_cells_id)

        # cuts needs one more entry.
        self.c.cuts = np.append(self.c.cuts,self.c.x.shape[0])

        return self

    def map_to_id(self,x):
        """
        Helper function mapping integer pairs to a unique integer point.
        """
        a = x[:,0]
        b = x[:,1]
        return ((a + b) * (a + b + 1) / 2 + b).astype(int)

    def map_particles(self):
        """
        Method which populates all nonempty cells.
        """
        for i in range(self.c.nonempty_cells_id.shape[0]):
            id_tag = self.c.nonempty_cells_id[i]

            x_cell = self.c.x[self.c.cuts[i]:self.c.cuts[i + 1],:]
            v_cell = self.c.v[self.c.cuts[i]:self.c.cuts[i + 1],:]
            masses_cell =  self.c.masses[self.c.cuts[i]:self.c.cuts[i + 1]]

            self.id_cells_dict[id_tag].populate(x_cell,v_cell,masses_cell)

    def sort_neighbor_particles(self):
        """
        Procedure which updates all the internal data of the neighbors given 
        the of each cell that is dependent on input physical data.

        Specifically, the collection of neighbors of a given cell are mapped
        to its id. The same values computed for the cell in sort_particles 
        are then computed for all neighboring points.
        """
        num_neighbors = 9
        _eps_e1 = np.arange(2)
        _eps_e2 = np.arange(1,-1,-1)
        _ul = -1 * _eps_e1 + _eps_e2
        _um = _eps_e2
        _ur = _eps_e1 + _eps_e2
        _ml = -1 * _eps_e1
        _mm = np.zeros(2)
        _mr = _eps_e1
        _bl = -1 * _eps_e1 + -1 * _eps_e2
        _bm = -1 * _eps_e2
        _br = _eps_e1 + -1 * _eps_e2

        direction_vectors =  np.array([_ul,_um,_ur,_ml,_mm,_mr,_bl,_bm,_br])

        # Copy the physical data of c 9 times.
        x_copies = np.repeat(self.c.x[:, :, np.newaxis], num_neighbors, axis=2)
        g_copies = np.repeat(self.c.x_g[:, :, np.newaxis], num_neighbors, axis=2)
        v_copies = np.repeat(self.c.v[:, :, np.newaxis], num_neighbors, axis=2)
        masses_copies = np.repeat(self.c.masses[:,np.newaxis],num_neighbors,axis=1)
        # collapse neighboring grid points to grid points offset by each neighboring direction.
        # The 3.1 is just to deal with rounding issues.
        cells_copies = np.rint((g_copies + direction_vectors.T) / (3*self.eps))

        # Flatten both X values and neighbor copies in the same way
        self.n.x_g = cells_copies.transpose(2,0,1).reshape(-1,self.dim)
        self.n.x = x_copies.transpose(2,0,1).reshape(-1,self.dim)
        self.n.v = v_copies.transpose(2,0,1).reshape(-1,self.dim)

        self.n.masses = masses_copies.transpose(1,0).reshape(-1)
        # Apply hash to full_cells
        self.n.x_id = self.map_to_id(self.n.x_g)

        # argsort hash values
        self.n.sort_indices = np.argsort(self.n.x_id)

        # sort X and V copies with the arg
        self.n.x = self.n.x[self.n.sort_indices,:]
        self.n.v = self.n.v[self.n.sort_indices,:]

        # sort hash value
        self.n.x_id = self.n.x_id[self.n.sort_indices]
        self.n.nonempty_cells_id = np.unique(self.n.x_id)
        return self

    def map_neighbors(self):
        """
        Method which populates neighbors of all nonempty cells.
        """
        num_neighbors = 9

        self.n.cuts = np.searchsorted(self.n.x_id, self.n.nonempty_cells_id)
        # needs to add the endpoint
        num_duplicated_pts = num_neighbors * self.c.x.shape[0]
        self.n.cuts = np.append(self.n.cuts,num_duplicated_pts)

        for i in range(self.n.nonempty_cells_id.shape[0]):
            cell_id = self.n.nonempty_cells_id[i]
            x_cell_neighbors = self.n.x[self.n.cuts[i]:self.n.cuts[i + 1],:]
            v_cell_neighbors = self.n.v[self.n.cuts[i]:self.n.cuts[i + 1],:]
            masses_neighbors = self.n.masses[self.n.cuts[i]:self.n.cuts[i + 1]]

            # Assign X_cell to the cell corresponding to hash
            self.id_cells_dict[cell_id].populate_neighbors(x_cell_neighbors,
                                                v_cell_neighbors, masses_neighbors)

    def distribute_computation(self,method,*args):
        """
        Method that applies an input cell method using input *args to all
        nonempty cells in the cell lookup.
        """
        for i in self.c.nonempty_cells_id:
            method(self.id_cells_dict[i],*args)
