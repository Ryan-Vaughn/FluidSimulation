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

@dataclasses.dataclass
class SPHInputData():
    """
    Data class to book keep all the internal data of the Distributor.

    Upon initialization of DistributorSPH2D, two SPHInputData classes
    are constructed. One (c) is used to track all internal data of cells
    on the grid. The other (n) is used to track the data for all neighbors
    of a given cell.

    In the neighbors case, the total number of data points tracked is
    num_neighbors * num_pts, where num_neighbors is the number of faces of
    the cube + 1 (9). This allows to properly allocate each point to its
    neighboring cells.

    Parameters
    ----------
    x : NDArray[np.float32], shape = (num_pts, dim)
        An array of all particles in the simulation.

    v : NDArray[np.float32], shape = (num_pts, dim)
        An array of the velocity of all particles in the simulation.

    masses : NDArray[np.float32], shape = num_pts
        An array of the masses of all particles in the simulation.

    x_id : NDArray[np.int_], shape = num_pts,
        An array whose entreees track the unique integer id of the cell
        that currently contains the corresponding particle in x.

    x_g : NDArray[np.int_], shape = (num_pts, dim)
        An array whose entrees are the nearest integer multiple of the scaling
        parameter eps. The entrees of x_g are used to generate x_id. The array
        x_g is kept public because it is useful in the construction of the 
        second SPHInputData class tracking neighboring points.

    nonempty_cells_id : NDArray[np.int_] shape = num_nonempty_cells,
        An array consisting of all nonempty cell ids.

    sort_indices : NDArray[np.int_] shape = num_pts
        The output of np.argsort(x_id). Sorting indices used to sort the
        physical data of the simulation (x,v,masses for instance)
    
    cuts : NDArray[np.int_], shape = nonempty_cell_id.shape
        An array used to allocate particles in x to and from the cells which 
        they occupy. After sorting x_id and collecting the unique id values, 
        the ith and i+1-th entry of cuts such that x[cuts[i]:cuts[i+1],:] is
        the set of particles in the ith nonempty cell.

    """
    x : NDArray[np.float32] = None
    v : NDArray[np.float32] = None
    masses : NDArray[np.float32] = None

    x_id : NDArray[np.int_] = None
    x_g : NDArray[np.int_] = None
    cuts : NDArray[np.int_] = None
    nonempty_cells_id : NDArray[np.int_] = None
    sort_indices : NDArray[np.int_] = None

@dataclasses.dataclass
class SPHOutputData():
    """
    Data class to book keep output data from cells.

    This data is then passed to the manager/solver to update the simulation.
    Pressures needs to be tracked in global memory so that 

    Parameters
    ----------
    x : NDArray[np.float32], shape = (num_pts, dim)
        An array of all particles in the simulation.

    v : NDArray[np.float32], shape = (num_pts, dim)
        An array of the velocity of all particles in the simulation.

    masses : NDArray[np.float32], shape = num_pts
        An array of the masses of all particles in the simulation.

    pressures : NDArray[np.float32], shape = num_pts
        An array of the pressures of all particles in the simulation.

    """
    x : NDArray[np.float32] = None
    v : NDArray[np.float32] = None
    masses : NDArray[np.float32] = None
    pressures : NDArray[np.float32] = None


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
    def distribute_computation_void(self,data_container,method,*args):
        """
        A method which applies a cell method to all nonempty cells,
        then returns the outputs into global memory.

        Parameters
        ----------

        data_container: dict
            A data structure (usually a dictionary) which is used to index
            all of the cells in the distributor.

        method: function
            A cell method to apply on all nonempty cells

        *args: arguments
            The arguments to pass into the cell methods.    
        """

class DistributorSPH2D(Distributor):
    """
    Distributor for distributing computations into Cells.

    """

    def __init__(self,physical_data,meta_data):
        self.dim = None
        self.num_pts = None
        self.num_neighbors = None

        self.cell_type, self.eps,self.bounds = meta_data

        self.id_cells_dict = self.initialize_lookup(*meta_data)
        # Data storage for cells
        self.c = SPHInputData()

        # Data storage for neighboring cells.
        self.n = SPHInputData()

        # Generate data for c and n depending on physical inputs.
        self.populate_particles(*physical_data).sort_particles(self.c)
        self.populate_neighbor_particles().sort_particles(self.n)
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

    def populate_particles(self,x,v,masses):
        """
        Procedure which updates all the internal data of the cell given the
        input physical data.

        Specifically, each particle is assigned an id corresponding to a unique
        integer grid point.
        """
        self.num_pts,self.dim = x.shape
        self.num_neighbors = 9 #T ODO: make this into a function of dim later

        self.c.x = x
        self.c.v = v
        self.c.masses = masses

        self.c.x_g = np.rint(self.c.x / self.eps).astype(int)
        self.c.x_id = self.map_to_id(self.c.x_g)
        return self

    def sort_particles(self,data_object):
        """
         The physical quantites are then all sorted according
        to this id.
        """
        data_object.sort_indices = np.argsort(data_object.x_id)

        data_object.x = data_object.x[data_object.sort_indices,:]
        data_object.v = data_object.v[data_object.sort_indices,:]
        data_object.masses = data_object.masses[data_object.sort_indices]

        data_object.x_id = data_object.x_id[data_object.sort_indices]
        data_object.x_g = data_object.x_g[data_object.sort_indices,:]

        data_object.nonempty_cells_id = np.unique(data_object.x_id)
        data_object.cuts = np.searchsorted(data_object.x_id, data_object.nonempty_cells_id)

        # cuts needs one more entry.
        data_object.cuts = np.append(data_object.cuts,data_object.x.shape[0])

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

    def populate_neighbor_particles(self):
        """
        Procedure which updates all the internal data of the neighbors given 
        the of each cell that is dependent on input physical data.

        Specifically, the collection of neighbors of a given cell are mapped
        to its id. The same values computed for the cell in sort_particles 
        are then computed for all neighboring points.
        """
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
        self.n.x = self.duplicate(self.c.x)
        self.n.v = self.duplicate(self.c.v)
        self.n.masses = self.duplicate(self.c.v)

        # Copy -> shift by direction vectors -> arrange in consistent manner.
        g_copies = np.repeat(self.c.x_g[:, :, np.newaxis],
                            self.num_neighbors,
                            axis=2)
        cells_copies = g_copies + direction_vectors.T
        self.n.x_g = cells_copies.transpose(2,0,1).reshape(-1,self.dim)

        # Apply hash to full_cells
        self.n.x_id = self.map_to_id(self.n.x_g)

        return self

    def sort_neighbor_particles(self):
        """
        Procedure which updates all the internal data of the neighbors given 
        the of each cell that is dependent on input physical data.

        Specifically, the collection of neighbors of a given cell are mapped
        to its id. The same values computed for the cell in sort_particles 
        are then computed for all neighboring points.
        """
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

    def distribute_computation_void(self,method,*args):
        """
        Method that applies an input cell method using input *args to all
        nonempty cells in the cell lookup. Used for procedures that do not
        return data.
        """

        for i in self.c.nonempty_cells_id:
            method(self.id_cells_dict[i],*args)

    def distribute_computation_return(self,output_dim,method,*args):
        """
        Method that applies an input cell method using input *args to all
        nonempty cells in the cell lookup.
        """
        num_nonempty_cells = self.c.nonempty_cells_id.shape[0]
        if output_dim == 2:
            output = np.zeros((self.num_pts,output_dim))
            for i in range(num_nonempty_cells):
                tag = self.c.nonempty_cells_id[i]
                output[self.c.cuts[i]:self.c.cuts[i+1],:] = method(self.id_cells_dict[tag],*args)

        if output_dim == 1:
            output = np.zeros(self.num_pts)
            for i in range(num_nonempty_cells):
                tag = self.c.nonempty_cells_id[i]
                print((i,tag))
                output[self.c.cuts[i]:self.c.cuts[i+1]] = method(self.id_cells_dict[tag],*args)
            output = output.reshape(self.num_pts)

        return output

    def duplicate(self,d):
        """
        Helper function that duplicates input data to a form that is consistent
        with neighbor data. I.e. duplicate(densities) maps densities of each
        particle to an array of size num_neighbors times the original array.
        This array then book keeps the densities in neighboring cells.
        """
        if len(d.shape) == 2:
            _, d_dim = d.shape
        elif len(d.shape) == 1:
            d_dim = 1
        else:
            print("Error: Tried to duplicate a 3+ dimensional array.")

        if d_dim == 2:
            d_copies = np.repeat(d[:, :, np.newaxis], self.num_neighbors, axis=2)
            d_copies = d_copies.transpose(2,0,1).reshape(-1,self.dim)
        if d_dim == 1:
            d_copies = np.repeat(d_copies[:,np.newaxis],self.num_neighbors,axis=1)
            self.n.masses = d_copies.transpose(1,0).reshape(-1)

        return d_copies
