import numpy as np
from abc import ABC, abstractmethod

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
    def distribute_computation(self,method,*args):
        """
        method for applying a cell method to all cells in the distributor.
        """
        pass

class DistributorSPH2D(Distributor):
    """
    Distributor for distributing computations into Cells.
    """

    def __init__(self,x,v,masses,bounds,eps,cell_type):        
        self.num_pts,self.dim = x.shape
        self.eps = eps

        self.x = x
        self.v = v
        self.masses = masses
        self.id_cells_dict = self.initialize_lookup(cell_type, bounds, eps)


        self.x_id = None
        self.x_g = None
        self.cuts = None

        self.x_neighbors = None
        self.v_neighbors = None
        self.masses_neighbors = None
        self.neighbors_id = None
        self.neighbors_g = None
        self.neighbors_cuts = None

        self.nonempty_cells_id = None
        self.nonempty_cell_neighbors_id = None

        self.sort_indices = None
        self.sort_indices_neighbors = None


    def initialize_lookup(self,cell_type,bounds,eps):
        """
        Procedure to create dictionaries which index the Cells in a grid.

        In the 2d SPH case, this is done by constructing a cell for each
        pair of integers inside of the given rectangular domain, then 
        applying a bijective map between the set of integers and the 
        set of pairs of integers, resulting in a unique integer id for 
        each cell.
        """
        grid_bounds = np.ceil(bounds/eps)
        grid_x_coords = range(-1,int(grid_bounds[0,0]) + 1)
        grid_y_coords = range(-1,int(grid_bounds[0,1]) + 1)

        id_cells_dict = {int((grid_x_coords + grid_y_coords) *
                             (grid_x_coords + grid_y_coords + 1) / 2
                               + grid_y_coords) : cell_type()}
        return id_cells_dict

    def assign_particles(self,x,eps):
        """
        Main method that ...
        """
        self.sort_particles(x,eps).map_particles()
        self.sort_neighbor_particles().map_neighbors()
    
    def sort_particles(self,x,eps):
        """
        Procedure which generates x_id, which gives the corresponding id to each particle in x.
        """
        # map X to grid
        self.x_g = np.rint(x / eps).astype(int)
        self.x_id = self.map_to_id(self.x_g)
        
        self.sort_indices = np.argsort(self.x_id)

        self.x_id = self.x_id[self.sort_indices]
        self.x = self.x[self.sort_indices,:]
        self.x_g = self.x_g[self.sort_indices,:]
        self.v = self.v[self.sort_indices,:]

        self.nonempty_cells_id = np.unique(self.x_id)
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
        
        """
        self.cuts = np.searchsorted(self.x_id, self.nonempty_cells_id)
        # needs to add the endpoint
        self.cuts = np.append(self.cuts,self.dim)
        
        # for each unique hash ID:
        
        for i in range(self.nonempty_cells_id.shape[0]):
            id_tag = self.nonempty_cells_id[i]

            x_cell = self.x[self.cuts[i]:self.cuts[i + 1],:]
            v_cell = self.v[self.cuts[i]:self.cuts[i + 1],:]
            masses_cell =  self.masses[self.cuts[i]:self.cuts[i + 1]]
            # Assign X_cell to the cell corresponding to hash
            self.id_cells_dict[id_tag].populate(x_cell,v_cell,masses_cell)

    def sort_neighbor_particles(self):
        """

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
        # Copy X num_neighbors times (num_neighbors depends on dim)
        # This copy is to book keep positions.
        x_copies = np.repeat(self.x[:, :, np.newaxis], num_neighbors, axis=2)
        g_copies = np.repeat(self.x_g[:, :, np.newaxis], num_neighbors, axis=2)
        v_copies = np.repeat(self.v[:, :, np.newaxis], num_neighbors, axis=2)
        
        # collapse neighboring grid points to grid points offset by each neighboring direction.
        # The 3.1 is just to deal with rounding issues.
        cells_copies = np.rint((g_copies + direction_vectors.T) / 3)
        
        # Flatten both X values and neighbor copies in the same way
        cells_copies = cells_copies.transpose(2,0,1).reshape(-1,self.dim)
        self.x_neighbors = x_copies.transpose(2,0,1).reshape(-1,self.dim)
        self.v_neighbors = v_copies.transpose(2,0,1).reshape(-1,self.dim)
        
        # Apply hash to full_cells
        self.neighbors_id = self.map_to_id(cells_copies)
        
        # argsort hash values
        self.sort_indices_neighbors = np.argsort(self.neighbors_id)
        
        # sort X and V copies with the arg
        self.x_neighbors = self.x_neighbors[self.sort_indices_neighbors,:]
        self.v_neighbors = self.v_neighbors[self.sort_indices_neighbors,:]

        # sort hash value
        self.neighbors_id = self.neighbors_id[self.sort_indices_neighbors]
        self.nonempty_cell_neighbors_id = np.unique(self.neighbors_id)
        return self
    
    def map_neighbors(self):
        """
        
        """
        num_neighbors = 9

        self.neighbors_cuts = np.searchsorted(self.neighbors_id, self.nonempty_cell_neighbors_id)
        # needs to add the endpoint
        num_duplicated_pts = num_neighbors * self.num_pts
        self.neighbors_cuts = np.append(self.neighbors_cuts,num_duplicated_pts)
        
    
        for i in range(self.nonempty_cell_neighbors_id.shape[0]):
            cell_id = self.nonempty_cell_neighbors_id[i]
            x_cell_neighbors = self.x_neighbors[self.neighbors_cuts[i]:self.neighbors_cuts[i + 1],:]
            v_cell_neighbors = self.v_neighbors[self.neighbors_cuts[i]:self.neighbors_cuts[i + 1],:]
             
            # Assign X_cell to the cell corresponding to hash
            self.id_cells_dict[cell_id].set_neighbors(x_cell_neighbors,
                                                v_cell_neighbors)

    def distribute_computation(self,method,*args):
        """
        Method that applies an input cell method using input *args to all
        nonempty cells in the cell lookup.
        """
        for i in self.nonempty_cells_id:
            method(self.id_cells_dict[i],*args)

