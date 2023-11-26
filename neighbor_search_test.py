# Test Script to profile speed
import numpy as np
import matplotlib.pyplot as plt

# NEIGHBORS ARE DETECTED 

num_pts = 20
dim = 2
eps = 1

X = 10 * np.random.rand(num_pts,dim)
G = eps * np.rint(X / eps)

G_0 = np.rint(X / (3 * eps))

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

def map_to_grid(X):
    return eps * np.rint(X / eps)

# vectors that track neighbors
direction_vectors = [[ul,um,ur],[ml,mm,mr],[bl,bm,br]]

def collapse_neighbors(X,v):
    # Map data to nearest integer multiple of epsilon offset by v. 
    return 3 * eps * np.rint((X+eps*v) / (3 * eps)) - eps * v

# 
# Algorithm
# for eps vec in direction_vectors:
#   Points --> Snap to Grid --> Collapse Neighbors(offset by eps_vec)
# hash all outputs
# sort hash
grid_points = np.arange(0,10 + eps,eps)
grid_coords = np.array([[i,j] for i in grid_points for j in grid_points])

fig,axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))

for j in range(3):
    for i in range(3):
        axs[i][j].scatter(grid_coords[:,0],grid_coords[:,1])

        axs[i][j].scatter(X[:,0],X[:,1])
        nearest_grid = eps * np.rint(X / eps)
        axs[i][j].scatter(nearest_grid[:,0],nearest_grid[:,1])
        group_neighbors = collapse_neighbors(nearest_grid,direction_vectors[i][j])
        axs[i][j].scatter(group_neighbors[:,0],group_neighbors[:,1])

plt.legend()
plt.show()


# ---------------------------------------------------------------------------

x_g = G[:,0]
y_g = G[:,1]

# Applying hash to G
h_map = (x_g + y_g) * (x_g + y_g + 1) / 2 + y_g

inds = np.argsort(h_map)

sorted_keys = h_map[inds]
unique_keys = np.unique(sorted_keys)

cuts = np.searchsorted(sorted_keys,unique_keys)
