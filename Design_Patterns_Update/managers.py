"""
Module for managers class. The manager is in charge of carrying out global
computations that do not benefit from being distributed, as well as
controlling the distributor and solver.
"""
from abc import ABC
import distributors
import cells
import solvers

class Manager(ABC):
    """
    Manager class for doing Fluid Simulations using Smoothed Particle
    Hydrodynamics (SPH.)
    """

    def __init__(self,physical_data,meta_data,global_variables):

        self.pressure_constant = None
        self.rest_density = None

        self.dist = distributors.SPHDistributor(physical_data,meta_data)
        self.solver = solvers.FluidSimForwardEulerSolver(dt)

    def iterate(self,physical_data):
        
        forces = self.compute_forces()
        a = forces / masses
        self.solver.solve(x,v,a)

    def compute_forces(self):
        """
        
        """
        pressure_forces = self.compute_pressure_force()
        viscosity_forces = self.compute_viscosity_force()
        wind_forces = self.compute_wind_force()
        gravitational_forces = self.compute_gravitational_force()

        return forces
    
    def compute_pressure_force(self):
        densities = self.dist.distribute_method(cells.FluidSimCell.compute_forces_step_1, returns=True)
        pressures = self.pressure_constant * (densities - self.rest_density)
        pressures_duplicated = self.dist.duplicate(pressures)
        pressures_n = self.dist.arrange(pressures_duplicated)

        self.dist.distribute_method(cells.FluidSimCell.set_pressures,pressures, returns=True)
        self.dist.distribute_method(cells.FluidSimCell.set_pressures_n,pressures_n,domain='n')
        
        self.dist.distribute_method(cells.FluidSimCell.set_pressures,pressures)
        self.dist.distribute_method(cells.FluidSimCell.compute_forces_step_1, returns=True)

    def compute_viscosity_force(self):
        pass
    
    def compute_viscosity_force(self):
        pass

    def compute_gravitational_force(self):
        pass

    def compute_wind_force(self):
        pass


    def update_bounds(self):
        pass
    
    def update_solver(self):
        pass