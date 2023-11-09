import numpy as np

class Simulation:
    
    # Class variables that specify the inital distribution and constants.
    
    initial_dataset = 'Gaussian'
    initial_velocities = 'Stationary'
    
    margin = 1.5

    mass_constant = 1
    gravity_constant = 9.8
    eps = .6
    rest_density = 1
    pressure_constant = .4
    viscosity_constant = .001

    dt = 1/60

    
    def __init__(self,num_pts,dim):
        
        self.num_pts = num_pts
        self.dim = dim
        self.t = 0

        # Generate points, bounds, intial velocity
        self.populate()
        # Set instance constants to the default class variables.
        self.initialize_constants()
        
        self.gravity_forces  = np.zeros((self.num_pts,self.dim))
        self.gravity_forces[:,1] = - self.gravity_constant * np.ones(self.num_pts)
    
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
            self.V =  np.zeros(self.dim)
        else:
            self.V =  np.zeros(self.dim)

        self.bounds = np.zeros((2,self.dim))
    
        # Set initial boundaries based on the initial points.
        for i in range(self.dim):
            upper_bound = np.max(self.X[:,i])
            lower_bound = np.min(self.X[:,i])
            self.bounds[0,i] = upper_bound + Simulation.margin
            self.bounds[1,i] = lower_bound - Simulation.margin

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

        
    def simulate(self):
        # Compute the pairwise distances
        self.update_distances()
        self.update_distance_gradients()

        

        # Compute the pressure
        self.update_density_kernel_matrix()
        self.update_densities()    
        self.update_pressures()
        
        self.update_pressure_kernel_gradient()
        self.compute_pressure_force()
        
        # Compute the viscosity
        self.update_viscosity_kernel_laplacian()
        self.compute_viscosity_force()

        # Compute the surface tension (not implemented)
        self.surface_tension_forces = np.zeros(self.X.shape)
        
        # Gravity is computed once outside the loop

        ## sum the forces
        self.forces = self.pressure_forces + self.viscosity_forces + self.surface_tension_forces + self.gravity_forces

        ## Update velocity using forward Euler
        self.V = self.V + self.forces * self.dt

        # Update position from velocity using forward Euler
        self.X = self.X + self.V * self.dt

        # Resolve boundary collisions
        self.apply_boundary_collisions()

        # Update the time
        self.t += self.dt

    #Density Estimation
    def update_distances(self): 
        xy = np.dot(self.X.T,self.X) # O(num_pts ** 2 * dim)
        x2 = (self.X * self.X).sum(1) # O(num_pts * dim)
        y2 = (self.X * self.X).sum(1) # O(num_pts * dim)
        d2 = np.add.outer(x2,y2) - 2 * xy  # O(num_pts * dim)
        d2.flat[::len(self.X)+1] = 0 # Rounding issues? Don't understand this.
        self.distances = np.sqrt(d2)  # O (num_pts * dim)

    def update_density_kernel_matrix(self):
        self.density_kernel_matrix = 315/(64 * np.pi * self.eps ** 9) * (self.eps ** 2 - self.distances ** 2) ** 3
        self.density_kernel_matrix[self.distances > self.eps] = 0

    def update_densities(self):
        #Compute densities using mass weighted kernel density estimation
        self.densities = self.mass_constant * np.sum(self.density_kernel_matrix, axis=0)

    # Pressure Force Estimation
    def update_pressures(self):
        self.pressures = self.pressure_constant * (self.densities - self.rest_density)
        

    def update_distance_gradients(self):
        self.distance_gradients = np.zeros((self.num_pts,self.num_pts,self.dim))
        # We make an artificial change here to avoid division by zero on the diagonal.
        modified_distances =  self.distances + np.eye(self.num_pts)
        for i in range(self.dim):
            self.distance_gradients[:,:,i] =  np.subtract.outer(self.X[:,i],self.X[:,i]) / modified_distances

    def update_pressure_kernel_gradient(self):
        # First compute the gradient of the shape function using distance = distance ** 2.
        kernel_derivative = 45 * (self.eps - self.distances) ** 2
        # Assume that derivative is zero outside the support of the shape function.
        kernel_derivative[self.distances>self.eps] = 0
        
        # Compute the total derivative from the input distance gradients and the computed
        # kernel gradient.
        self.kernel_gradients = np.zeros((self.num_pts,self.num_pts,self.dim))
        
        for i in range(self.dim):
            self.kernel_gradients[:,:,i] = kernel_derivative * self.distance_gradients[:,:,i]

    def compute_pressure_force(self):
        # Compute the symmetric pressure so that the SPH pressure computation is symmetric.
        symmetric_pressures = np.add.outer(self.pressures,self.pressures) / (2 * self.densities)

        # Weight the gradients by the symmetric pressures
        pressure_forces = np.zeros((self.num_pts,self.num_pts, self.dim))
        
        for i in range(self.dim):
            pressure_forces[:,:,i] = symmetric_pressures.T * self.kernel_gradients[:,:,i]

        #sum over one axis to obtain estimate for the pressure forces at each point.
        self.pressure_forces = self.mass_constant *  np.sum(self.pressure_forces,axis=1)

    # Viscosity Estimation
    def update_viscosity_kernel_matrix(self):
        self.viscosity_kernel_matrix = 15/(2 * np.pi * self.eps ** 3) * ((- self.distances ** 3)/(2 * self.eps ** 3) +  (self.distances ** 2) / self.eps ** 2 + self.eps / (2 * self.distances) - 1)
        self.viscosity_kernel_matrix[self.distances > self.eps] = 0
        

    def update_viscosity_kernel_laplacian(self):
        self.viscosity_kernel_laplacian = 45 / (np.pi * self.eps ** 6) * (self.eps - self.distances)
        

    def compute_viscosity_force(self):
        # Compute the symmetric pressure so that the SPH pressure computation is symmetric.
        symmetric_velocities = np.zeros((self.num_pts,self.num_pts,self.dim))
        self.viscosity_forces = np.zeros((self.num_pts,self.num_pts,self.dim))
        for i in range(self.dim):
            symmetric_velocities[:,:,i] = np.subtract.outer(self.V[:,i],self.V[:,i]) / self.densities
            self.viscosity_forces[:,:,i] = symmetric_velocities[:,:,i] * self.viscosity_kernel_laplacian
        
        self.viscosity_forces = self.mass_constant * self.viscosity_constant * np.sum(self.viscosity_forces,axis=1) 

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

    
    