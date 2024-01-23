# Fluid Simulation
A fluid simulation platform implemented in Python based on the Smoothed Particle Hydrodynamics discretization scheme. Smoothed Particle Hydrodynamics is a mesh-free kernel based particle
method commonly used in fluid simulations. It is a commonly used scheme, with many advantages. In particular, for an incompressible fluid, conservation of mass is manifestly preserved in
this scheme.

This project is mostly an implementation of methods presented in several papers. It is purely for my own educational use, and is a great test case for practicing software design principles.
In particular, I followed test-driven development with unit testing in PyTest. The package design is mostly comprised of a distributed computing data structure, that reduces the number of
pairwise interactions required for the simulation. The software architecture is designed to be highly general, and I plan to extend the functionality of the framework to encompass several
other applications of Smoothed Particle Dynamics, including relativistic fluid dynamics and modelling of Bose Einstein Condensates.

## References
https://matthias-research.github.io/pages/publications/sca03.pdf

http://www.ligum.umontreal.ca/Clavet-2005-PVFS/pvfs.pdf

https://web.archive.org/web/20140725014123/https://docs.nvidia.com/cuda/samples/5_Simulations/particles/doc/particles.pdf

## Features
- Automated unit testing using PyTest
- Custom data structure for computing pairwise particle interactions (reduces complexity from O(n^2) to O(nlogn)
- Data structure is dimension independent: can easily be generalized to K spatial dimensions (although algorithm has exponential dimension complexity O(3^dim)).
- Kernel based -- Originating paper uses custom polynomial kernel for computations.
- Data structure is kernel-independent and can be used for any one parameter family of kernels with support radius epsilon.
- Current example is rendered using Matplotlib

## Package Dependencies
- Numpy (Backend)
- Matplotlib (Visualizations, to be replaced with pyOpenGL)
- Python dataclasses (For convenience, used to book-keep large number of simulation variables)
- PyTest
- Scipy.spatial.cdist (For pairwise distance computations, mostly for convenience.)

## Planned Features
Optimizations:
- Visualizations rendered in pyOpenGL
- GPU acceleration by drag-and-drop replacement using CuPy
- JIT Compiling using Numba
