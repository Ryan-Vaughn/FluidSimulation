import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

from abc import ABC, abstractmethod


class Cell(ABC):
    """
    Object which performs local computations on particles.

    The locality of the computations is dictated by some compactly
    supported kernel with bandwidth parameter eps.          
    """

    def __init__(self):
        self.dim = 2

        self.x_c = None
        self.v_c = None
        self.num_pts = None

        self.x_n = None
        self.v_n = None
        self.num_n_pts = None

        self.eps = None
        self.distances = None
        self.densities_c = None
        self.densities_n = None
        self.pressure_gradients = None
        self.density_kernel_matrix = None

    def populate(self,x: npt.NDArray ,v: npt.NDArray) -> None:
        """
        Method to load position and velocity of particles in the cell into
        memory.
        """
        self.x_c = x
        self.v_c = v

        self.num_pts, _ = self.x_c.shape()

    def populate_neighbors(self, x: npt.NDArray, v: npt.NDArray) -> None:
        """
        Method to load position and velocity of particles in neighboring cells
        into memory.
        """
        self.x_n = x
        self.v_n = v

        self.num_n_pts, _ = self.x_n.shape()

    @abstractmethod
    def compute_distances(self):
        """
        Abstract method to compute distances between particles.
        """
        pass

class CellSPH2D(Cell):
    """
    Cell for computing forces in 2d Smoothed Particle Hydrodynamics.          
    """

    def compute_distances(self):
        """
        Computes the pairwise Euclidean distances between particles in x_c
         and x_n.
        """
        self.distances = cdist(self.x_c,self.x_n)

    def compute_density_kernel(self):
        self.density_kenrnel_matrix = None

    def compute_densities(self):
        """
        Computes the density for each x_c in the cell from 
        """
        self.densities_c = self.mass_constant * np.sum(self.density_kernel_matrix, axis=0)

    def get_densities(self):
        """
        Getter method to return density values because they are needed by the 
        manager for symmetric density computation.
        """
        return self.densities
    
    def set_neighbor_densities(self,densities_n):
        """
        Setter method to return density values of neighboring cells because
        they are needed by the 
        manager for symmetric density computation.
        """
        self.densities_n = densities_n
