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
class SPHDistInputData:
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
    
    slices : list[np.s_], shape = nonempty_cell_id.shape
        An array of numpy slice objects, used to partition all arrays of
        interest into smaller arrays corresponding to the particles in 
        a given nonempty cell. I.e. self.c.x[slices[0]] returns all 
        positions of the particles in the sim while self.n.x[slices[1]]
        returns the positions of all particles neighboring the second
        nonempty cell in the simulation.  

    """
    x : NDArray[np.float32] = None
    v : NDArray[np.float32] = None
    masses : NDArray[np.float32] = None

    x_id : NDArray[np.int_] = None
    x_g : NDArray[np.int_] = None
    slices : list[np.s_] = None
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


class SPHDistributor(Distributor):
    """
    Distributor for distributing computations into Cells.

    """

    def __init__(self,initial_physical_data,dist_meta_data):

        self.dim = None
        self.num_pts = None
        self.num_neighbors = None

        self.cell_type, self.eps,self.bounds = dist_meta_data

        self.id_cells_dict = self.initialize_lookup(*dist_meta_data)

        # Data class for storing global data for cells
        self.c = SPHDistInputData()

        # Data class for storing global data for neighbors.
        self.n = SPHDistInputData()

        self.update(initial_physical_data)

    def update(self,physical_data):
        """
        Update the positions, velocities, and masses in the simulation, then
        assign corresponding data to the cells. This happens once on
        initialization, then repeats at every computation step in the sim.
        """
        # Load all the global data into the distributor and generate cell
        # assignment ids.
        self.load_particles(*physical_data).sort_particles(self.c)
        self.load_neighbors().sort_particles(self.n)

        # Assign local data to appropriate cells.
        self.populate_particles(self.c)
        self.populate_particles(self.n)

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

    def load_particles(self,x,v,masses):
        """
        Procedure which updates all the internal data of the cell given the
        input physical data.

        Specifically, each particle is assigned an id corresponding to a unique
        integer grid point.
        """

        if len(x.shape) == 1: # handling 1d array case
            self.dim = 1
            self.num_pts = x.shape[0]
        else:
            self.num_pts,self.dim = x.shape

        self.num_neighbors = 3 ** self.dim

        self.c.x = x
        self.c.v = v
        self.c.masses = masses

        self.c.x_g = np.rint(self.c.x / self.eps).astype(int)
        self.c.x_id = self.map_to_id(self.c.x_g)
        return self
    
    def load_neighbors(self):
        """
        Procedure which updates the data necessary to track all particles 
        neighboring each cell.

        This is achieved through composing three functions: duplicate, shift,
        arrange. Duplicate copies the list of all particles once for each
        neighboring cell. Shift maps each individual copy of the data to the
        corresponding neighboring cell, and arrange organizes the duplicated
        data in a consistent manner.
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

        shift_vectors =  np.array([_ul,_um,_ur,_ml,_mm,_mr,_bl,_bm,_br])

        # Duplicate the array num_neighbors times then arrange to a 2d array.
        # This is to keep track of physical data corresponding to a particle.
        self.n.x = self.arrange(self.duplicate(self.c.x))
        self.n.v = self.arrange(self.duplicate(self.c.v))
        self.n.masses = self.arrange(self.duplicate(self.c.masses))

        # Duplicate the grid points, then shift each grid point to one of its
        # neighbors, finally arrange in the same manner as the physical data.
        g_copies = self.duplicate(self.c.x_g)
        shifted_g_copies = self.shift(g_copies,shift_vectors)
        self.n.x_g = self.arrange(shifted_g_copies)

        # Map the grid points uniquely to an integer id value.
        self.n.x_id = self.map_to_id(self.n.x_g)

        return self

    def sort_particles(self,data_object):
        """
        Procedure that sorts all relevant data according to an integer
        id value assigned.

        Parameters
        ----------
        data_object : SPHDistInputData
            Data object holding information tracking either cell data or
            neighboring cell data. Either self.c or self.n are the only
            valid inputs.
        """
        data_object.sort_indices = np.argsort(data_object.x_id)

        data_object.x = data_object.x[data_object.sort_indices,:]
        data_object.v = data_object.v[data_object.sort_indices,:]
        data_object.masses = data_object.masses[data_object.sort_indices]

        data_object.x_id = data_object.x_id[data_object.sort_indices]
        data_object.x_g = data_object.x_g[data_object.sort_indices,:]

        data_object.nonempty_cells_id = np.unique(data_object.x_id)
        cuts = np.searchsorted(data_object.x_id, data_object.nonempty_cells_id)

        # cuts needs one more entry which is num_pts in either c or n.
        cuts = np.append(cuts,data_object.x.shape[0])

        data_object.slices = [np.s_[cuts[i]:cuts[i+1],...] for i in range(len(cuts)-1)]
        return self

    def map_to_id(self,x):
        """
        Helper function mapping integer pairs to a unique integer point.
        """
        a = x[:,0]
        b = x[:,1]
        return ((a + b) * (a + b + 1) / 2 + b).astype(int)

    def populate_particles(self,data_object):
        """
        Helper function for initialization procedure. Loads each particle
        into its corresponding cell memory.
        """
        self.distribute_method(self.cell_type.populate,
                               data_object.x,
                               data_object.v,
                               data_object.masses)

    def duplicate(self,d):
        """
        Helper function that duplicates input data to a form that is consistent
        with neighbor data. I.e. duplicate(densities) maps densities of each
        particle to an array of size num_neighbors times the original array.
        This array then book keeps the densities in neighboring cells.
        """
        d_copies = np.repeat(d[..., np.newaxis], self.num_neighbors, axis=-1)

        return d_copies

    def arrange(self,d_copies):
        """
        Helper function that arranges a duplicated array in an organized manner
        that allows for neighbor data to be tracked and sorted.
        """
        if len(d_copies.shape) == 3:
            _, d_dim, _ = d_copies.shape
        elif len(d_copies.shape) == 2:
            d_dim = 1
        else:
            print("Error: Array has too many dimensions. Check the duplicate method.")

        if d_dim == 2:
            d_arranged_copies = d_copies.transpose(2,0,1).reshape(-1,self.dim)
        if d_dim == 1:
            d_arranged_copies = d_copies.transpose(1,0).reshape(-1)

        return d_arranged_copies

    def shift(self, d_copies, shifts):
        """
        A helper function that maps each copy of the points in the direction
        of the neighbor.
        """
        shifted_d_copies = d_copies + shifts.T
        return shifted_d_copies

    def distribute_method(self,cell_method,*args, domain='c',returns = False, codomain ='c',
                        **kwargs):
        """
        A decorator that distributes a given cell method across all cells in
        the distributor.

        Parameters
        ----------

        cell_method : callable
            A function that acts on a single cell.

        *args : tuple
            The non-keyword arguments that will be passed to each cell in the
            distributor

        domain : str
            Controls which indexing cut points to use. The input 'c' dictates
            that the method iterates over all points in the simulation. The
            input 'n' dictates that the method iterates over all neighbor
            point in the simulation (which will be more than the number
            of total points due to double counting.)

        returns : bool
            Controls whether or not the cell method needs to aggregate the
            output values of the cell methods into a single global array.

        codomain : str
            Similar to domain, except this dictates the indexing for the 
            output global array. Uses the same standard as domain, but
            only relevant if returns == True.
        """

        if domain == 'c':
            domain_data_object = self.c
        if domain == 'n':
            domain_data_object = self.n

        if codomain == 'c':
            codomain_data_object = self.c
            _num_pts = self.c.x.shape[0]
        if domain == 'n':
            codomain_data_object = self.n
            _num_pts = self.n.x.shape[0]

        if returns is False:
            generators = map(lambda A:
                            (A[i] for i in
                            domain_data_object.slices), args)
            iterator = zip(*generators)

            for cell_id,inputs in zip(self.n.nonempty_cells_id,iterator):
                cell_method(self.id_cells_dict[cell_id],*inputs,**kwargs)

        if returns is True:
            output_dim = cell_method.output_dim
            output_shape = (_num_pts, output_dim)
            
            if output_dim == 1:
                output_shape = (_num_pts,)
            
            output = np.zeros(output_shape)

            generators = map(lambda A:
                            (A[i] for i in
                            domain_data_object.slices), args)
            iterator = zip(*generators)

            output_iterator = (i for i in codomain_data_object.slices)
            debug = (self.c.nonempty_cells_id,output_iterator,iterator)
            
            for cell_id,output_slices,inputs in zip(self.c.nonempty_cells_id,
                                                    output_iterator,
                                                    iterator):

                output[output_slices] = cell_method(self.id_cells_dict[cell_id],
                                                    *inputs,
                                                    **kwargs)

            
            return output,debug
