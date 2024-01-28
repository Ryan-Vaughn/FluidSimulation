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
        self.pressures_c = None

        self.x_n = None
        self.v_n = None
        self.masses_n = None
        self.num_pts_n = None
        self.pressures_n = None

        self.distances = None
        
        self.pressure_constant = None
        self.rest_density = None

        self.pressure_gradients = None
        self.density_kernel_matrix = None

    def populate(self, x: npt.NDArray, v: npt.NDArray, masses: npt.NDArray) -> None:
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
        _coeff = 315 / (64 * np.pi * self.eps ** 9)
        _matrix = (self.eps ** 2 - self.distances ** 2) ** 3

        self.density_kernel_matrix = _coeff * _matrix
        self.density_kernel_matrix[self.distances > self.eps] = 0

    def compute_densities(self):
        """
        Computes the density for each x_c in the cell from 
        """

        self.densities_c = self.masses_c * np.sum(self.density_kernel_matrix, axis=1)

    ## NEED TO BE MERGED:
    """
    def compute_distance_gradients(self):
        # compute the gradients of the distance function between points in cell and 
        # distance
        self.distance_gradients = np.zeros((self.num_pts,self.num_pts,self.dim))
        # We make an artificial change here to avoid division by zero on the diagonal.
        modified_distances =  self.distances + np.eye(self.num_pts)
        
        # Another crappy hack to keep things from breaking
        modified_distances[modified_distances <=0] = 3 * np.finfo(float).eps

        for i in range(self.dim):
            self.distance_gradients[:,:,i] =  np.subtract.outer(self.X_c[:,i],self.X_n[:,i]) / modified_distances

    def compute_pressure_kernel_gradients(self):
        # Compute the pressure kernel gradient (only the shape function.)
        # First compute the gradient of the shape function using distance = distance ** 2.
        kernel_derivative = 45 * (self.eps - self.distances) ** 2
        # Assume that derivative is zero outside the support of the shape function.
        kernel_derivative[self.distances>self.eps] = 0
        
        # Compute the total derivative from the input distance gradients and the computed
        # kernel gradient.
        self.kernel_gradients = np.zeros((self.num_pts_c,self.num_pts_n,self.dim))
        
        for i in range(self.dim):
            self.kernel_gradients[:,:,i] = kernel_derivative * self.distance_gradients[:,:,i]


    def compute_pressures(self):
       # Using the Simulation variables rest_density and pressure_constant, 
       # compute the pressure from the density
       self.pressures = self.pressure_constant * (self.densities - self.rest_density)
    
    def get_pressures(self):
        return self.pressures

          
    # TODO: MERGE IN
    
    def compute_symmetric_pressures(self):
        # Query all neighbors and take an average of the pairwise pressures.
        # NOTE: We need to have some indication that all pressures in the
        # simulation have been computed already (or at least for all neighbors)

        # Compute the symmetric pressure so that the SPH pressure computation is symmetric.
        self.symmetric_pressures = np.add.outer(self.pressures_c,self.pressures_n) / (2 * self.densities)

    def compute_pressure_forces(self):
        # combine pressure kernel gradient, distance gradient, and symmetric pressure
        # to obtain the pressure force.

        # Weight the gradients by the symmetric pressures
        self.pressure_forces = np.zeros((self.num_pts,self.num_pts, self.dim))
        
        for i in range(self.dim):
            self.pressure_forces[:,:,i] = self.symmetric_pressures.T * self.kernel_gradients[:,:,i]

        #sum over one axis to obtain estimate for the pressure forces at each point.
        self.pressure_forces = self.mass_constant *  np.sum(self.pressure_forces,axis=1)

    def compute_viscosity_kernel(self):
        # Compute the viscosity kernel matrix in analogous fashion to density kernel.                    
        self.viscosity_kernel_matrix = 15/(2 * np.pi * self.eps ** 3) * ((- self.distances ** 3)/(2 * self.eps ** 3) +  (self.distances ** 2) / self.eps ** 2 + self.eps / (2 * self.distances) - 1)
        self.viscosity_kernel_matrix[self.distances > self.eps] = 0

    def compute_viscosity_kernel_laplacian(self):
        # Compute the viscosity kernel matrix in analogous fashion to density kernel.
        self.viscosity_kernel_laplacian = 45 / (np.pi * self.eps ** 6) * (self.eps - self.distances)
    
    def compute_symmetric_velocities(self):
        # Query all neighbors and take an average of the pairwise velocity vectors.
        # NOTE: We need to have some indication that all pressures in the
        # simulation have been computed already (or at least for all neighbors)
        pass

    def compute_viscosity_force(self):
        # Compute the velocity force
        symmetric_velocities = np.zeros((self.num_pts,self.num_pts,self.dim))
        self.viscosity_forces = np.zeros((self.num_pts,self.num_pts,self.dim))
        
        for i in range(self.dim):
            symmetric_velocities[:,:,i] = np.subtract.outer(self.V[:,i],self.V[:,i]) / self.densities
            self.viscosity_forces[:,:,i] = symmetric_velocities[:,:,i] * self.viscosity_kernel_laplacian
        
        self.viscosity_forces = self.mass_constant * self.viscosity_constant * np.sum(self.viscosity_forces,axis=1) 
    """