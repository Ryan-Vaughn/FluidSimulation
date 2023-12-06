import numpy as np

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
        self.cell_x_coords = range(int(self.cell_bounds[0,0]) + 1)
        self.cell_y_coords = range(int(self.cell_bounds[0,1]) + 1)

        #dictionary which maps integer multiples of epsilon to the coresponding Cell
        self.cells_dict = { (x,y) : Cell((x,y)) 
                           for x in self.cell_x_coords 
                           for y in self.cell_y_coords}
        
        self.cells_hash_dict = { (x,y) :  x + y * (x + y + 1) / 2 + y
                           for x in self.cell_x_coords 
                           for y in self.cell_y_coords}
        
        self.cells_hash_dict_inv = { h : (x,y) for (x,y),h 
                                    in self.cells_hash_dict.items() }
        
    # ------------------------------------------------------------------------    
    # Helper Functions for initialize_cells
    # ------------------------------------------------------------------------    

    def assign_particles(self):

        self.sort_particles()
        self.get_nonempty_cells()
        self.map_particles()
        self.sort_neighbor_particles()
        self.map_neighbors()

    def sort_particles(self):
        # map X to grid
        self.G = self.eps * np.rint(self.X / self.eps)
        # map grid coordinates to hash ID.
        self.x_g = self.G[:,0]
        self.y_g = self.G[:,1]

        # Applying hash to G
        self.hash_id = (self.x_g + self.y_g) * (self.x_g + self.y_g + 1) / 2 + self.y_g
        # arg sort hash IDs
        self.sort_indices = np.argsort(self.hash_id)
        # sort hash_id
        self.hash_id = self.hash_id[self.sort_indices]
        self.X = self.X[self.sort_indices]

    def get_nonempty_cells(self):
        self.nonempty_cell_hashes = np.unique(self.hash_id)

    def map_particles(self):
        cuts = np.searchsorted(self.hash_id, self.nonempty_cell_hashes)
        # for each unique hash ID:
        for hash in self.nonempty_cell_hashes:
            #Take out the
            X_cell = self.X[cuts[hash]:cuts[hash + 1]]
            # Assign X_cell to the cell corresponding to hash
            self.cells_dict[self.cells_hash_dict_inv[hash]].populate(X_cell)
            
    def sort_neighbor_particles(self):
        # Copy X num_neighbors times (num_neighbors depends on dim)
        # This copy is to book keep hash values
        
        # Copy X num_neighbors times (num_neighbors depends on dim)
        # This copy is to book keep positions.

        # map hash values X copies to grid
        # for each copy of hash values X:
        #   collapse neighboring grid points to grid points offset by each neighboring direction.

        # flatten hash values X copies to a 2d array
        # flatten positions X copies to a 2d array in the exact same way

        # map hash values X copies to hash ID
        # argsort hash values X copies

        # sort position X copies with the arg
        pass

    def map_neighbors(self):
        # for each unique hash ID for just X:
        #   cell.assign_neighbors(hash_ID,X_cell) (assign X_cell and all neighboring X_cell to neighbors)
        pass
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    
    def simulate(self):
        # --------------------------------------------------------------------
        # TODO:
        # --------------------------------------------------------------------
        # 1. For each possible Cell (computed from bounds)
        #   - Populate (Possibly involves a sorting of X?)
        self.assign_particles()
        #   - Compute distances, density kernel, densities.
        #   - Return cell density to global density vector.

        # 2. For each possible Cell
        #   - Compute symmetric density (Requires the global density vector update)
        #   - Compute distance gradients
        #   - Compute the pressure kernel shape function gradient
        #   - Compute the pressure force.
        # 
        #   - Compute the viscosity kernel shape function laplacian.
        #   - Compute the viscosity force.
        #   - Return pressure force and viscosity force to global simulation memory.
        # 3. Compute global forces
        # 4. Iterate the dynamics using forward Euler
        # 5. Resolve boundary collisions
        # 6. Update time.
        # --------------------------------------------------------------------

        # Compute the pairwise distances
        self.assign_compute_distances()
        self.assign_compute_distance_gradients()

        # Compute the pressure
        self.assign_compute_density_kernel_matrix()
        self.assign_compute_densities()    
        self.assign_compute_pressures()
        
        self.assign_compute_pressure_kernel_gradient()
        self.assign_compute_pressure_force()
        
        # Compute the viscosity
        self.assign_compute_viscosity_kernel_laplacian()
        self.assign_compute_viscosity_force()

        # Compute the surface tension (not implemented)
        self.surface_tension_forces = np.zeros(self.X.shape)
        
        # Compute Gravity 
        self.gravity_forces  = np.zeros((self.num_pts,self.dim))
        self.gravity_forces[:,1] = - self.gravity_constant * self.mass_constant * np.ones(self.num_pts)

        # Compute wind (just for fun)
        self.wind_forces  = np.zeros((self.num_pts,self.dim))
        self.wind_forces[:,0] =  5 * np.cos(2 * self.t) * np.ones(self.num_pts)

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

    #Density Estimation
    def assign_compute_distances(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_distances()

    def assign_compute_density_kernel(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_density_kernel()

    def assign_compute_densities(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_densitites()
       
    def assign_compute_pressures(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_pressures()
    
    def assign_compute_distance_gradients(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_distance_gradients()
        
    def assign_compute_pressure_kernel_gradients(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_pressure_kernel_gradient()
        
    def assign_compute_pressure_forces(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_pressure_forces()

    # Viscosity Estimation
    def assign_compute_viscosity_kernel(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_viscosity_kernel()        

    def assign_compute_viscosity_kernel_laplacian(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_viscosity_kernel_laplacian()      
        pass

    def assign_compute_viscosity_force(self):
        for hash in self.nonempty_cell_hashes:
            self.cells_hash_dict[hash].compute_viscosity_forces()
        
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
        
        self.coords = coords
        self.index = None
        self.neighbors_coords = None
        self.neighbors_id = None
        self.active = None

# ----------------------------------------------------------------------------
# Passing data to and from Simulation
# ----------------------------------------------------------------------------

    def populate(self,parent_sim):
        # query the parent simulation to populate the cell with points X_C (X in cell C)
        # query the parent simulation for the velocities V_C of X_C
        # query all neighboring cells to get all points in neighbors of X_C (X_neighbors?)
        
        # construct an array which stores the location of the points of X_C in X
        pass
    
    def update_density(self,parent_sim):
        # Return the densities of each point in the cell and their corresponding position in global memory
        pass

    def update_pressure(self,parent_sim):
        # Return the pressure force of each point in the cell and its corresponding position in global memory.
        pass

    def update_viscosity(self,parent_sim):
        # Return the viscosity force of each point in the cell and its corresponding position in global memory.
        pass

# ----------------------------------------------------------------------------
# Iterating the Dynamics
# ----------------------------------------------------------------------------

    def compute_distances(self):
        # compute the pairwise distances between points in the cell and points in and
        # neighboring the cell.

        xy = self.X @ self.X.T # O(num_pts ** 2 * dim)
        x2 = (self.X * self.X).sum(1) # O(num_pts * dim)
        y2 = (self.X * self.X).sum(1) # O(num_pts * dim)
        d2 = np.add.outer(x2,y2) - 2 * xy  # O(num_pts * dim)
        d2.flat[::len(self.X)+1] = 0 # Rounding issues? Don't understand this.
        
        # Another crappy hack to fix negative distances.
        d2[d2<=0] = 3 * np.finfo(float).eps

        self.distances = np.sqrt(d2)  # O (num_pts * dim)
        pass

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

    def compute_symmetric_pressure(self):
        # Query all neighbors and take an average of the pairwise pressures.
        # NOTE: We need to have some indication that all pressures in the
        # simulation have been computed already (or at least for all neighbors)
        pass

    def compute_pressure_forces(self):
        # combine pressure kernel gradient, distance gradient, and symmetric pressure
        # to obtain the pressure force.

        # Compute the symmetric pressure so that the SPH pressure computation is symmetric.
        symmetric_pressures = np.add.outer(self.pressures,self.pressures) / (2 * self.densities)

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
    
    