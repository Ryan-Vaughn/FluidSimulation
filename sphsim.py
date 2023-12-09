import numpy as np
from scipy.spatial.distance import cdist

# TODO: We need to find a way to fix the numerical errors breaking things in
# 1) distance computation sqrt(negative number)
# 2) distance gradient computations (division by zero)

class Simulation:
    
    # Class variables that specify the inital distribution and constants.
    
    initial_dataset = 'Gaussian'
    initial_velocities = 'Stationary'
    
    margin = .5
    mass_constant = 3
    gravity_constant = 9.8
    eps = .3
    rest_density = 1
    pressure_constant = 1
    viscosity_constant = .002
    dt = 1/30

    
    def __init__(self,num_pts,dim):
        
        self.num_pts = num_pts
        self.dim = dim
        self.t = 0

        # Generate points, bounds, intial velocity
        self.populate()
        # Set initial bounding behavior
        self.initialize_bounds()
        # Set instance constants to the default class variables.
        self.initialize_constants()
        #Initialize grid (must have been populated already)
        self.initialize_cells()

    def populate(self,dataset='Gaussian',velocities='Stationary'):
        
        if dataset == 'Two Different Boxes':
            self.X = np.random.rand(self.num_pts,self.dim) # This was chosen to be sklearn compatible.
            self.X[int(self.num_pts/2):self.num_pts,:] = 4 * self.X[int(self.num_pts/2):self.num_pts,:] + 2
        elif dataset == 'Two Equal Boxes':
            self.X = np.random.rand(self.num_pts,self.dim)
            self.X[int(self.num_pts/2):self.num_pts,0] = self.X[int(self.num_pts/2):self.num_pts,0] + 2
        else:
            self.X = np.random.rand(self.num_pts,self.dim)
        
        if velocities == 'Stationary':
            self.V =  np.zeros((self.num_pts,self.dim))
        else:
            self.V =  np.zeros((self.num_pts,self.dim))
        return self
    
    def initialize_bounds(self):
        # Set initial boundaries based on the initial points.
        self.bounds = np.zeros((2,self.dim))
        for i in range(self.dim):
            upper_bound = np.max(self.X[:,i])

            self.bounds[0,i] = upper_bound + Simulation.margin
            self.bounds[1,i] = 0

        return self
    
    def initialize_constants(self):
        # Set all constant parameters according to the class variables. This occurs at initialization only.
        self.mass_constant = Simulation.mass_constant
        self.gravity_constant = Simulation.gravity_constant
        self.eps = Simulation.eps
        self.rest_density = Simulation.rest_density
        self.pressure_constant = Simulation.pressure_constant    
        self.viscosity_constant = Simulation.viscosity_constant
        self.dt = Simulation.dt

    def update_constants(self,
                      mass_constant,
                      gravity_constant,
                      eps,
                      rest_density,
                      pressure_constant,
                      viscosity_constant,
                      dt):
        
        # Method to update the constants of a simulation without resetting the simulation.
        self.mass_constant = mass_constant
        self.gravity_constant = gravity_constant
        self.eps = eps
        self.rest_density = rest_density
        self.pressure_constant = pressure_constant    
        self.viscosity_constant = viscosity_constant
        self.dt = dt

    def update_bounds(self):
        # Method to adjust the bounds on the fly.
        pass
    
    def initialize_cells(self):
        """
        Helper function called in __init__().Procedure to create a dictionary 
        of all cells in the simulation linked to their integer grid 
        coordinates. Also constructs lookup dictionaries from hash values to
        integer coordinates.

        This procedure will be  called only once per resize.
        """

        self.cell_bounds = np.ceil(self.bounds/self.eps)
        self.cell_x_coords = range(-1,int(self.cell_bounds[0,0]) + 1)
        self.cell_y_coords = range(-1,int(self.cell_bounds[0,1]) + 1)

        #dictionary which maps integer multiples of epsilon to the coresponding Cell
        self.cells_loc_dict = { (x,y) : Cell((x,y)) 
                           for x in self.cell_x_coords 
                           for y in self.cell_y_coords}
        
        self.cells_hash_dict = { (x,y) :  int((x + y) * (x + y + 1) / 2 + y)
                           for x in self.cell_x_coords 
                           for y in self.cell_y_coords}
        
        self.cells_hash_dict_inv = { h : (x,y) for (x,y),h 
                                    in self.cells_hash_dict.items() }
        
        # Dictionary mapping hash values to corresponding Cell
        self.cells_dict = { k : self.cells_loc_dict[v] for k,v in self.cells_hash_dict_inv.items()}

        
    # ------------------------------------------------------------------------    
    # Helper Functions for initialize_cells
    # ------------------------------------------------------------------------    

    def assign_particles(self):
        self.sort_particles()
        self.get_nonempty_cells()
        self.map_particles()
        self.sort_neighbor_particles()
        self.get_nonempty_neighbors()
        self.map_neighbors()

    def sort_particles(self):
        # map X to grid
        self.G = np.rint(self.X / self.eps).astype(int)
        # map grid coordinates to hash ID.
        self.x_g = self.G[:,0]
        self.y_g = self.G[:,1]

        # Applying hash to G
        self.hash_id = ((self.x_g + self.y_g) * (self.x_g + self.y_g + 1) / 2 + self.y_g).astype(int)
        # arg sort hash IDs
        self.sort_indices = np.argsort(self.hash_id)
        # sort hash_id
        self.hash_id = self.hash_id[self.sort_indices]
        
        self.X = self.X[self.sort_indices,:]
        # We need this sort for the later neighbors computation
        self.G = self.G[self.sort_indices,:]
        self.V = self.V[self.sort_indices,:]

    def get_nonempty_cells(self):
        self.nonempty_cell_hashes = np.unique(self.hash_id)

    def map_particles(self):
        self.cuts = np.searchsorted(self.hash_id, self.nonempty_cell_hashes)
        # needs to add the endpoint
        self.cuts = np.append(self.cuts,self.num_pts)
        
        # for each unique hash ID:
        
        for i in range(self.nonempty_cell_hashes.shape[0]):
            hash = self.nonempty_cell_hashes[i]

            X_cell = self.X[self.cuts[i]:self.cuts[i + 1],:]
            V_cell = self.V[self.cuts[i]:self.cuts[i + 1],:]
             
            # Assign X_cell to the cell corresponding to hash
            self.cells_dict[hash].populate(X_cell,V_cell)

    def sort_neighbor_particles(self):

        self.num_neighbors = 9

        # generate a list of direction vectors for the grid.
        eps_e1 = np.arange(2)
        eps_e2 = np.arange(1,-1,-1)

        ul = -1 * eps_e1 + eps_e2
        um = eps_e2
        ur = eps_e1 + eps_e2
        ml = -1 * eps_e1
        mm = np.zeros(2)
        mr = eps_e1
        bl = -1 * eps_e1 + -1 * eps_e2
        bm = -1 * eps_e2
        br = eps_e1 + -1* eps_e2

        direction_vectors = np.array([ul,um,ur,ml,mm,mr,bl,bm,br])

        # Copy X num_neighbors times (num_neighbors depends on dim)
        # This copy is to book keep positions.
        X_copies = np.repeat(self.X[:, :, np.newaxis], self.num_neighbors, axis=2)
        G_copies = np.repeat(self.G[:, :, np.newaxis], self.num_neighbors, axis=2)
        V_copies = np.repeat(self.V[:, :, np.newaxis], self.num_neighbors, axis=2)
        
        # collapse neighboring grid points to grid points offset by each neighboring direction.
        # The 3.1 is just to deal with rounding issues.
        cells_copies = np.rint((G_copies + direction_vectors.T) / 3)
        
        # Flatten both X values and neighbor copies in the same way
        cells_copies = cells_copies.transpose(2,0,1).reshape(-1,self.dim)
        self.X_neighbors = X_copies.transpose(2,0,1).reshape(-1,self.dim)
        self.V_neighbors = V_copies.transpose(2,0,1).reshape(-1,self.dim)
        
        # Apply hash to full_cells
        x_g = cells_copies[:,0]
        y_g = cells_copies[:,1]
        self.neighbors_hash_id = ((x_g + y_g) * (x_g + y_g + 1) / 2 + y_g).astype(int)
        
        # argsort hash values
        self.neighbors_sort_indices = np.argsort(self.neighbors_hash_id)
        
        # sort X and V copies with the arg
        self.X_neighbors = self.X_neighbors[self.neighbors_sort_indices,:]
        self.V_neighbors = self.V_neighbors[self.neighbors_sort_indices,:]

        # sort hash value
        self.neighbors_hash_id = self.neighbors_hash_id[self.neighbors_sort_indices]

    def get_nonempty_neighbors(self):
        self.nonempty_cell_neighbors_hashes = np.unique(self.neighbors_hash_id)
    
    def map_neighbors(self):
        self.neighbors_cuts = np.searchsorted(self.neighbors_hash_id, self.nonempty_cell_neighbors_hashes)
        # needs to add the endpoint
        num_duplicated_pts = self.num_neighbors * self.num_pts
        self.neighbors_cuts = np.append(self.neighbors_cuts,num_duplicated_pts)
        
    
        for i in range(self.nonempty_cell_neighbors_hashes.shape[0]):
            hash = self.nonempty_cell_neighbors_hashes[i]
            X_cell_neighbors = self.X_neighbors[self.neighbors_cuts[i]:self.neighbors_cuts[i + 1],:]
            V_cell_neighbors = self.V_neighbors[self.neighbors_cuts[i]:self.neighbors_cuts[i + 1],:]
             
            # Assign X_cell to the cell corresponding to hash
            self.cells_dict[hash].set_neighbors(X_cell_neighbors,
                                                V_cell_neighbors)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    
    def simulate(self):
        # Assign Particles and Velocities to cells.
        self.assign_particles()
        
        # Compute the pairwise distances inside Cells
        self.assign_compute_distances()
        self.assign_compute_distance_gradients()

        # Compute the pressure
        self.assign_compute_density_kernel()
        self.assign_compute_densities()    
        
        # Densities need to be updated so that neighbors can be queried
        # for symmetric pressure computation.
        self.update_densities()
        
        # Finish Computing Pressure Forces
        self.assign_compute_symmetric_pressures()
        self.assign_compute_pressures()
        self.assign_compute_pressure_kernel_gradients()
        self.assign_compute_pressure_forces()
        
        # Compute the Viscosity Forces
        self.assign_compute_viscosity_kernel_laplacian()
        self.assign_compute_viscosity_force()

        self.update_pressure_forces()
        self.update_viscosity_forces()

        # Compute Gravity, Wind
        self.compute_global_forces()

        ## sum the forces
        self.forces = self.pressure_forces + self.viscosity_forces + self.surface_tension_forces + self.gravity_forces + self.wind_forces

        ## Update velocity using forward Euler
        self.V += self.forces/self.mass_constant * self.dt

        # Update position from velocity using forward Euler
        self.X = self.X + self.V * self.dt

        # Resolve boundary collisions
        self.apply_boundary_collisions()

        # Update the time
        self.t += self.dt
# ----------------------------------------------------------------------------
# Methods that query Cell information back into global memory.
# ----------------------------------------------------------------------------
    def update_densities(self):
        for i in range(self.nonempty_cell_hashes.shape[0]):
            hash = self.nonempty_cell_hashes[i]

            # Insert the cell's density values back into global memory.
            self.densities[self.cuts[i]:self.cuts[i + 1]] = self.cells_dict[self.cells_hash_dict_inv[hash]].densities

    def update_pressure_forces(self):
        for i in range(self.nonempty_cell_hashes.shape[0]):
            hash = self.nonempty_cell_hashes[i]
            
            # Insert the cell's density values back into global memory.
            self.pressure_forces[self.cuts[i]:self.cuts[i + 1],:] = self.cells_dict[self.cells_hash_dict_inv[hash],:].pressure_forces

    def update_viscosity_forces(self):
        for i in range(self.nonempty_cell_hashes.shape[0]):
            hash = self.nonempty_cell_hashes[i]
            
            # Insert the cell's density values back into global memory.
            self.viscosity_forces[self.cuts[i]:self.cuts[i + 1],:] = self.cells_dict[self.cells_hash_dict_inv[hash],:].viscosity_forces
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Methods that instruct all nonempty Cells to compute a quantity.
# ----------------------------------------------------------------------------
    # Density Estimation
    def assign_compute_distances(self):
        for i in self.nonempty_cell_hashes:
            print(i)
            self.cells_dict[i].compute_distances()

    def assign_compute_density_kernel(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_dict[hash].compute_density_kernel()

    def assign_compute_densities(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_dict[hash].compute_densitites()
       
    def assign_compute_pressures(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_dict[hash].compute_pressures()

    def assign_compute_symmetric_pressures(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_dict[hash].compute_symmetric_pressures()            
    
    def assign_compute_distance_gradients(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_dict[hash].compute_distance_gradients()
        
    def assign_compute_pressure_kernel_gradients(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_dict[hash].compute_pressure_kernel_gradient()
        
    def assign_compute_pressure_forces(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_dict[hash].compute_pressure_forces()

    # Viscosity Estimation
    def assign_compute_viscosity_kernel(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_dict[hash].compute_viscosity_kernel()        

    def assign_compute_viscosity_kernel_laplacian(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_dict[hash].compute_viscosity_kernel_laplacian()      

    def assign_compute_viscosity_force(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_viscosity_forces()
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

    # Global Force Computation
    def compute_global_forces(self):
            # Compute the surface tension (not implemented)
            self.surface_tension_forces = np.zeros(self.X.shape)
            
            # Compute Gravity 
            self.gravity_forces  = np.zeros((self.num_pts,self.dim))
            self.gravity_forces[:,1] = - self.gravity_constant * self.mass_constant * np.ones(self.num_pts)

            # Compute wind (just for fun)
            self.wind_forces  = np.zeros((self.num_pts,self.dim))
            self.wind_forces[:,0] =  5 * np.cos(2 * self.t) * np.ones(self.num_pts)

    # Collision Detection
    def apply_boundary_collisions(self):
        for i in range(self.dim):
        
            # find the points in X that are outside the bound
            too_large = self.X[:,i] > self.bounds[0,i]
            too_small = self.X[:,i] < self.bounds[1,i]
            
            # replace outside value with boundary value
            self.X[:,i][too_large] = self.bounds[0,i]
            self.X[:,i][too_small] = self.bounds[1,i]
            
            # reflect the velocity of the point
            self.V[:,i][too_large] = -self.V[:,i][too_large]
            self.V[:,i][too_small] = -self.V[:,i][too_small]

            # set force in the boundary direction on the point to zero
            self.forces[:,i][too_large | too_small] = 0

    
class Cell:
    '''
    The Cell class is used to handle any processing that only requires local
    data (i.e. nearby points.) Its main purpose is to take in a subset of
    points and velocities from the simulation, then output the pressure and
    viscosity forces acting on those points.
    '''

    def __init__(self,coords):
        # set coordinates and compute all neighbors coordinates
        # apply hash function to coords to get cell id
        # do we need neighbor id?
        
        self.X_c = None
        self.V_c = None
        self.coords = coords

        self.index = None
        self.neighbors_coords = None
        self.neighbors_id = None
        self.active = None

# ----------------------------------------------------------------------------
# Passing data to and from Simulation
# ----------------------------------------------------------------------------

    def populate(self,X_cell,V_cell):
        # query the parent simulation for the velocities V_C of X_C
        # query all neighboring cells to get all points in neighbors of X_C (X_neighbors?)
        # construct an array which stores the location of the points of X_C in X
        self.X_c = X_cell
        self.V_c = V_cell

    def set_neighbors(self,X_neighbors,V_neighbors):
        self.X_n = X_neighbors
        self.V_n = V_neighbors    

# ----------------------------------------------------------------------------
# Iterating the Dynamics
# ----------------------------------------------------------------------------

    def compute_distances(self):
        self.distances = cdist(self.X_c,self.X_n)
    
    def compute_distance_gradients(self):
        # compute the gradients of the distance function between points in cell and 
        # distance
        self.distance_gradients = np.zeros((self.num_pts,self.num_pts,self.dim))
        # We make an artificial change here to avoid division by zero on the diagonal.
        modified_distances =  self.distances + np.eye(self.num_pts)
        
        # Another crappy hack to keep things from breaking
        modified_distances[modified_distances <=0] = 3 * np.finfo(float).eps

        for i in range(self.dim):
            self.distance_gradients[:,:,i] =  np.subtract.outer(self.X[:,i],self.X[:,i]) / modified_distances
        pass

    def compute_density_kernel(self):
        # compute the density kernel matrix for all points in X. Kernel matrix
        # not necessarily square. One axis tracks points in cell, other tracks
        # points in cell + points in neighbor cells.
        self.density_kernel_matrix = 315/(64 * np.pi * self.eps ** 9) * (self.eps ** 2 - self.distances ** 2) ** 3
        self.density_kernel_matrix[self.distances > self.eps] = 0
        pass
    
    def compute_densities(self):
        # Sum over the axis with neighbor points to compute density
        self.densities = self.mass_constant * np.sum(self.density_kernel_matrix, axis=0)
        pass

    def compute_pressure_kernel_gradients(self):
        # Compute the pressure kernel gradient (only the shape function.)
        # First compute the gradient of the shape function using distance = distance ** 2.
        kernel_derivative = 45 * (self.eps - self.distances) ** 2
        # Assume that derivative is zero outside the support of the shape function.
        kernel_derivative[self.distances>self.eps] = 0
        
        # Compute the total derivative from the input distance gradients and the computed
        # kernel gradient.
        self.kernel_gradients = np.zeros((self.num_pts,self.num_pts,self.dim))
        
        for i in range(self.dim):
            self.kernel_gradients[:,:,i] = kernel_derivative * self.distance_gradients[:,:,i]
        pass

    def compute_pressures(self):
       # Using the Simulation variables rest_density and pressure_constant, 
       # compute the pressure from the density
       self.pressures = self.pressure_constant * (self.densities - self.rest_density)
       pass

    def compute_symmetric_pressures(self):
        # Query all neighbors and take an average of the pairwise pressures.
        # NOTE: We need to have some indication that all pressures in the
        # simulation have been computed already (or at least for all neighbors)

        # Compute the symmetric pressure so that the SPH pressure computation is symmetric.
        symmetric_pressures = np.add.outer(self.pressures,self.pressures) / (2 * self.densities)
        pass

    def compute_pressure_forces(self):
        # combine pressure kernel gradient, distance gradient, and symmetric pressure
        # to obtain the pressure force.

        # Weight the gradients by the symmetric pressures
        self.pressure_forces = np.zeros((self.num_pts,self.num_pts, self.dim))
        
        for i in range(self.dim):
            self.pressure_forces[:,:,i] = symmetric_pressures.T * self.kernel_gradients[:,:,i]

        #sum over one axis to obtain estimate for the pressure forces at each point.
        self.pressure_forces = self.mass_constant *  np.sum(self.pressure_forces,axis=1)
        pass

    def compute_viscosity_kernel(self):
        # Compute the viscosity kernel matrix in analogous fashion to density kernel.                    
        self.viscosity_kernel_matrix = 15/(2 * np.pi * self.eps ** 3) * ((- self.distances ** 3)/(2 * self.eps ** 3) +  (self.distances ** 2) / self.eps ** 2 + self.eps / (2 * self.distances) - 1)
        self.viscosity_kernel_matrix[self.distances > self.eps] = 0
        pass

    def compute_viscosity_kernel_laplacian(self):
        # Compute the viscosity kernel matrix in analogous fashion to density kernel.
        self.viscosity_kernel_laplacian = 45 / (np.pi * self.eps ** 6) * (self.eps - self.distances)
        pass
    
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
        pass