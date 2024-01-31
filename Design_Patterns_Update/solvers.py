"""
Module for solvers classes. Solvers are objects that determine the numerical
procedure used to propagate the dynamics of the system in time.
"""

from abc import ABC

class Solver(ABC):
    """
    Object numerically propagates the dynamics according to some rule.
    """
    def __init__(self,dt):
        self.dt = dt

class FluidSimForwardEulerSolver(Solver):
    """
    A simple solver that propagates the position by x_i = x_(i-1) + v_(i-1) * dt
    and v_i = v_(i-1) + a(i-1) * dt. Boundary collisions are resolved by naively
    setting velocity equal to zero and moving particles inside the bounds at the
    end of every time step.
    """

    def solve(self,x,v,a):
        """
        Simple method to propagate physical dynamics using the forward Euler
        numerical method.
        """

        v_out = v + a * self.dt
        x_out = x + v * self.dt

        return x_out,v_out