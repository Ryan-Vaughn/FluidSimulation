import numpy as np
from abc import ABC, abstractmethod

class Manager(ABC):
    """
    Abstract class for Manager.

    Managers distribute computations to cells. They compute local and global
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compute_global_forces(self):
        """
        Method for computing forces that do not require distribution
        to local cells for computation (i.e. ~O(nlogn) or less.)
        """
        pass
    
    @abstractmethod
    def compute_local_forces(self):
        """
        Method for computing forces that require distribution to local
        cells for computation (i.e. depends on pairwise distance using
        polynomial decay kernel)          
        """
        pass
    
    @abstractmethod
    def combine_forces(self):
        """
        Procedure for combining the local and global forces.          
        """
        pass

class ManagerSPH2D(Manager):
    def __init(self):

        self.cell_bounds = None
        self.cells_loc_dict = None
        self.cells_hash_dict = None
        self.cells_hash_dict_inv = None

        self.neighbors_hash_id = None

        self.x_neighbors = None
        self.v_neighbors = None
        self.neighbors_hash_id = None
        self.nonempty_cell_hashes = None
        self.nonempty_cell_neighbors_hashes = None

    def initialize_cells(self):
        """
        Procedure to create a dictionary  of all cells in the simulation 
        linked to their integer grid coordinates. Also constructs lookup
        dictionaries from hash values to integer coordinates.
        """

        self.cell_bounds = np.ceil(self.bounds/self.eps)
        self.cell_x_coords = range(-1,int(self.cell_bounds[0,0]) + 1)
        self.cell_y_coords = range(-1,int(self.cell_bounds[0,1]) + 1)

        #dictionary which maps integer multiples of epsilon to the coresponding Cell
        self.cells_loc_dict = { (x,y) : Cell((x,y)) 
                           for x in self.cell_x_coords 
                           for y in self.cell_y_coords}

        self.cells_hash_dict = { (x,y) :  int((x + y) * (x + y + 1) / 2 + y)
                           for x in self.cell_x_coords 
                           for y in self.cell_y_coords}
        
        self.cells_hash_dict_inv = { h : (x,y) for (x,y),h 
                                    in self.cells_hash_dict.items() }
        
        # Dictionary mapping hash values to corresponding Cell
        self.cells_dict = { k : self.cells_loc_dict[v] for k,v in self.cells_hash_dict_inv.items()}

        
    def assign_particles(self):
        self.sort_particles()
        self.get_nonempty_cells()
        self.map_particles()
        self.sort_neighbor_particles()
        self.get_nonempty_neighbors()
        self.map_neighbors()

    def sort_particles(self):
        # map X to grid
        self.G = np.rint(self.X / self.eps).astype(int)
        # map grid coordinates to hash ID.
        self.x_g = self.G[:,0]
        self.y_g = self.G[:,1]

        # Applying hash to G
        self.hash_id = ((self.x_g + self.y_g) * (self.x_g + self.y_g + 1) / 2 + self.y_g).astype(int)
        # arg sort hash IDs
        self.sort_indices = np.argsort(self.hash_id)
        # sort hash_id
        self.hash_id = self.hash_id[self.sort_indices]
        
        self.X = self.X[self.sort_indices,:]
        # We need this sort for the later neighbors computation
        self.G = self.G[self.sort_indices,:]
        self.V = self.V[self.sort_indices,:]

    def get_nonempty_cells(self):
        self.nonempty_cell_hashes = np.unique(self.hash_id)

    def map_particles(self):
        self.cuts = np.searchsorted(self.hash_id, self.nonempty_cell_hashes)
        # needs to add the endpoint
        self.cuts = np.append(self.cuts,self.num_pts)
        
        # for each unique hash ID:
        
        for i in range(self.nonempty_cell_hashes.shape[0]):
            hash = self.nonempty_cell_hashes[i]

            X_cell = self.X[self.cuts[i]:self.cuts[i + 1],:]
            V_cell = self.V[self.cuts[i]:self.cuts[i + 1],:]
             
            # Assign X_cell to the cell corresponding to hash
            self.cells_dict[hash].populate(X_cell,V_cell)

    def sort_neighbor_particles(self):
        num_neighbors = 9

        # generate a list of direction vectors for the grid.
        eps_e1 = np.arange(2)
        eps_e2 = np.arange(1,-1,-1)

        ul = -1 * eps_e1 + eps_e2
        um = eps_e2
        ur = eps_e1 + eps_e2
        ml = -1 * eps_e1
        mm = np.zeros(2)
        mr = eps_e1
        bl = -1 * eps_e1 + -1 * eps_e2
        bm = -1 * eps_e2
        br = eps_e1 + -1* eps_e2

        direction_vectors = np.array([ul,um,ur,ml,mm,mr,bl,bm,br])

        # Copy X num_neighbors times (num_neighbors depends on dim)
        # This copy is to book keep positions.
        x_copies = np.repeat(self.x[:, :, np.newaxis], num_neighbors, axis=2)
        g_copies = np.repeat(self.g[:, :, np.newaxis], num_neighbors, axis=2)
        v_copies = np.repeat(self.v[:, :, np.newaxis], num_neighbors, axis=2)
        
        # collapse neighboring grid points to grid points offset by each neighboring direction.
        # The 3.1 is just to deal with rounding issues.
        cells_copies = np.rint((g_copies + direction_vectors.T) / 3)
        
        # Flatten both X values and neighbor copies in the same way
        cells_copies = cells_copies.transpose(2,0,1).reshape(-1,self.dim)
        self.x_neighbors = x_copies.transpose(2,0,1).reshape(-1,self.dim)
        self.v_neighbors = v_copies.transpose(2,0,1).reshape(-1,self.dim)
        
        # Apply hash to full_cells
        x_g = cells_copies[:,0]
        y_g = cells_copies[:,1]
        self.neighbors_hash_id = ((x_g + y_g) * (x_g + y_g + 1) / 2 + y_g).astype(int)
        
        # argsort hash values
        self.neighbors_sort_indices = np.argsort(self.neighbors_hash_id)
        
        # sort X and V copies with the arg
        self.x_neighbors = self.x_neighbors[self.neighbors_sort_indices,:]
        self.v_neighbors = self.v_neighbors[self.neighbors_sort_indices,:]

        # sort hash value
        self.neighbors_hash_id = self.neighbors_hash_id[self.neighbors_sort_indices]

    def get_nonempty_neighbors(self):
        self.nonempty_cell_neighbors_hashes = np.unique(self.neighbors_hash_id)
    
    def map_neighbors(self):
        num_neighbors = 9

        self.neighbors_cuts = np.searchsorted(self.neighbors_hash_id, self.nonempty_cell_neighbors_hashes)
        # needs to add the endpoint
        num_duplicated_pts = num_neighbors * self.num_pts
        self.neighbors_cuts = np.append(self.neighbors_cuts,num_duplicated_pts)
        
    
        for i in range(self.nonempty_cell_neighbors_hashes.shape[0]):
            hash = self.nonempty_cell_neighbors_hashes[i]
            x_cell_neighbors = self.x_neighbors[self.neighbors_cuts[i]:self.neighbors_cuts[i + 1],:]
            v_cell_neighbors = self.v_neighbors[self.neighbors_cuts[i]:self.neighbors_cuts[i + 1],:]
             
            # Assign X_cell to the cell corresponding to hash
            self.cells_dict[hash].set_neighbors(x_cell_neighbors,
                                                v_cell_neighbors)
