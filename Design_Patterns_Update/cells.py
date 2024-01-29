"""
Module for cells classes. Cells are in charge of making local computations
in the simulation. Local computations are computations only requiring
measurements between particles that are close together.

"""
from abc import ABC, abstractmethod
import dataclasses
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

class Cell(ABC):
    """
    Object which performs local computations on particles.

    The locality of the computations is dictated by some compactly
    supported kernel with bandwidth parameter eps.          
    """
    def __init__(self,eps):
        self.eps = eps
        self.distances = None
        self.c = None
        self.n = None


    @abstractmethod
    def compute_distances(self):
        """
        Abstract method to compute distances between particles.
        """


@dataclasses.dataclass
class FluidSimCellInputData():
    """
    Data class to book keep all the internal data of the Cell.


    Parameters
    ----------
    x : NDArray[np.float32], shape = (num_pts, dim)
        An array of all particles in the cell.

    v : NDArray[np.float32], shape = (num_pts, dim)
        An array of the velocity of all particles in the cell.

    masses : NDArray[np.float32], shape = num_pts
        An array of the masses of all particles in the cell.

    """
    x : NDArray[np.float32] = None
    v : NDArray[np.float32] = None
    masses : NDArray[np.float32] = None
    num_pts: int = None
    pressures: NDArray[np.float32] = None

class FluidSimCell(Cell):
    """
    Cell for computing forces in 2d Smoothed Particle Hydrodynamics.          
    """
    def __init__(self,eps):
        super().__init__(eps)

        self.dim = None

        self.c = FluidSimCellInputData
        self.n = FluidSimCellInputData

        self.density_kernel_matrix = None
        self.distance_gradients = None
        self.pressure_kernel_gradients = None

    def populate(self, x, v, masses):
        """
        Helper method to load physical data of particles into cell.
        """

        self.c.x = x
        self.c.v = v
        self.c.masses = masses

        self.c.num_pts, self.dim = self.c.x.shape

    def populate_n(self, x, v, masses):
        """
        Helper method to load physical data of neighboring particles into cell.
        """

        self.n.x = x
        self.n.v = v
        self.n.masses = masses

        self.n.num_pts, _ = self.n.x.shape

    def compute_distances(self):
        """
        Computes the pairwise Euclidean distances between particles in the
        cell and all neighboring particles (including those in the cell).
        """
        self.distances = cdist(self.c.x,self.n.x)


    def compute_density_kernel(self):
        """"
        Returns the density kernel. This density kernel is from ___ and is
        adapted to __ dimensions.
        """
        _coeff = 315 / (64 * np.pi * self.eps ** 9)
        _matrix = (self.eps ** 2 - self.distances ** 2) ** 3

        self.density_kernel_matrix = _coeff * _matrix
        self.density_kernel_matrix[self.distances > self.eps] = 0

    def compute_densities(self):
        """
        Computes the density at each particle's location in the cell.
        """
        self.c.densities = self.c.masses * np.sum(self.density_kernel_matrix, axis=1)

    def get_densities(self):
        """
        Method to return densities back to the distributor. The pressure will
        then be computed using global memory because it relies only on global
        variables and can be done more efficiently that way. 
        
        The pressure must be returned to global memory regardless because the
        pressure of neighboring particles is computed inside the neighboring
        cell and can only be updated after all of the pressures in the sim
        have been computed.
        """
        return self.c.densities

    def set_pressures(self,pressures):
        """
        Helper function used to set the pressures in the cell. Purely for
        convenience and clarity.
        """
        self.c.pressures = pressures

    def set_pressures_n(self,pressures):
        """
        Helper function used to set the pressures in the neighboring cells.
        Purely for convenience and clarity.
        """
        self.n.pressures = pressures

    def compute_distance_gradients(self):
        """
        Compute the gradient vectors of the kernel sections for each pair of points.
        """
        self.distance_gradients = np.zeros((self.c.num_pts,self.n.num_pts,self.dim))

        # Artificial hack to avoid division by zero.
        mod_dists =  self.distances + np.eye(self.c.num_pts,self.n.num_pts)

        # Another hack to avoid breaking stuff due to numerical epsilon.
        mod_dists[mod_dists <=0] = 3 * np.finfo(float).eps

        for i in range(self.dim):
            l1_dists = np.subtract.outer(self.c.x[:,i],self.n.x[:,i])
            self.distance_gradients[:,:,i] =  l1_dists / mod_dists

    def compute_pressure_kernel_gradients(self):
        """
        Compute the gradient of the pressure kernel. This is done by
        first computing the derivative of the shape function, then
        using chain rule by composing with distance gradients. 
        """
        kernel_deriv = 45 * (self.eps - self.distances) ** 2

        kernel_deriv[self.distances>self.eps] = 0 # zero outside eps

        self.pressure_kernel_gradients = np.zeros((self.c.num_pts,
                                                   self.n.num_pts,
                                                   self.dim))

        for i in range(self.dim):
            pderivs = kernel_deriv * self.distance_gradients[:,:,i]
            self.pressure_kernel_gradients[:,:,i] = pderivs
# TODO: MERGE THESE IN.

"""
    def compute_pressure_forces(self):
        # combine pressure kernel gradient, distance gradient, and symmetric pressure
        # to obtain the pressure force.

        # Weight the gradients by the symmetric pressures
        self.pressure_forces = np.zeros((self.c.num_pts,self.n.num_pts, self.dim))

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
