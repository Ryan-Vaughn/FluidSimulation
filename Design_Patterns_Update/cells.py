"""
Module for cells classes. Cells are in charge of making local computations
in the simulation. Local computations are computations only requiring
measurements between particles that are close together.

"""
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

class Cell(ABC):
    """
    Object which performs local computations on particles.

    The locality of the computations is dictated by some compactly
    supported kernel with bandwidth parameter eps.          
    """

    def __init__(self,eps):
        self.dim = 2
        self.eps = eps

        self.x_c = None
        self.v_c = None
        self.masses_c = None
        self.num_pts_c = None
        self.densities_c = None
        
        self.x_n = None
        self.v_n = None
        self.masses_n = None
        self.num_pts_n = None
        self.densities_n = None

        self.distances = None
        self.pressure_gradients = None
        self.density_kernel_matrix = None

    def populate(self, x: npt.NDArray, v: npt.NDArray,masses: npt.NDArray) -> None:
        """
        Method to load position and velocity of particles in the cell into
        memory.
        """
        self.x_c = x
        self.v_c = v
        self.masses_c = masses

        self.num_pts_c, _ = self.x_c.shape

    def populate_neighbors(self, x: npt.NDArray, v: npt.NDArray, masses: npt.NDArray) -> None:
        """
        Method to load position and velocity of particles in neighboring cells
        into memory.
        """
        self.x_n = x
        self.v_n = v
        self.masses_n = masses

        self.num_pts_n, _ = self.x_n.shape

    @abstractmethod
    def compute_distances(self):
        """
        Abstract method to compute distances between particles.
        """


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
        """"
        Returns the density kernel
        """
    def compute_densities(self):
        """
        Computes the density for each x_c in the cell from 
        """
        self.densities_c = self.masses_c * np.sum(self.density_kernel_matrix, axis=0)

    def get_densities(self):
        """
        Getter method to return density values because they are needed by the 
        manager for symmetric density computation.
        """

    def set_neighbor_densities(self,densities_n):
        """
        Setter method to return density values of neighboring cells because
        they are needed by the 
        manager for symmetric density computation.
        """
        self.densities_n = densities_n
