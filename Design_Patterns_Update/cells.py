import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

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

class CellSPH2D(ABC):
    """
    Cell for computing forces in 2d Smoothed Particle Hydrodynamics.          
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def process(self):
        pass