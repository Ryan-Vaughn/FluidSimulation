import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

class Interface(ABC):
    """
    Abstract base class for an interface that tracks all simulation data.

    Will likely be implemented as a dataclass.
    """
    def __init__(self):
        pass

class Solver(ABC):
    """
    Abstract base class for a numerical solver.

    Numerical solvers are responsible for updating the position of a collection
    of particles X according to some dynamical system taking position, velocity,
    and acceleration into account.
    """
    def __init__(self,X_0,V_0,dt):
        self.x = x_0
        self.v = v_0
        self.t = 0
        self.dt = 1/30
        self.f = None

    @abstractmethod
    def iterate(self):
        """
        Method used to propagate x,v,f forward in time (t) according to some rule.

        Testing if this works now.
        """
        pass

    @abstractmethod
    def get_f(self,manager):
        """
        Method used to update f using a manager that determines force generation
        algorithm.
        """
        pass

    @abstractmethod
    def boundary_check(self,bounds):
        """
        Final check used to enforce boundary conditions.
        """
        pass

class Manager(ABC):
    """
    Abstract class for Manager.

    Managers determine the strategy for computing the total force on the
    particles.
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


class Distributor(ABC):
    """
    Method for computing forces that require distribution to local
    cells for computation (i.e. depends on pairwise distance using
    polynomial decay kernel)          
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def assign(self,X,V,complex_dict,cell_method):
        pass


class Cell(ABC):
    """
    Object which performs computatations          
    """    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def process(self):
        pass