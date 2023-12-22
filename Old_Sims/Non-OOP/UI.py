import numpy as np
import tkinter as tk
import sphsim as sph

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.style as mplstyle
import matplotlib.collections as mc


sim = sph.Simulation(500,2)
#Initialize plot
mplstyle.use('fast')
fig = plt.Figure()

def animate(i):
    ax.cla()
    ax.set_xlim(sim.bounds[1,1], sim.bounds[0,1])
    ax.set_ylim(sim.bounds[1,1], sim.bounds[0,1])
    sim.simulate()
    collection.set_offsets(sim.X)
    collection.set_array(np.sum(sim.V**2,axis=1))
    ax.add_collection(collection)
    
#Run the GUI
vis = tk.Tk()
vis.title('Smooth Particle Hydrodynamics Visualization')

canvas = FigureCanvasTkAgg(fig, master=vis)
canvas.get_tk_widget().grid(column=0,row=1)

ax = fig.add_subplot(111)
ax.set_xlim(sim.bounds[1,1], sim.bounds[0,1])
ax.set_ylim(sim.bounds[1,1], sim.bounds[0,1])

sizes = 150 * np.ones(sim.num_pts)
collection = mc.CircleCollection(sizes, offsets=sim.X, transOffset=ax.transData,alpha=.1,linewidth=0,cmap = 'plasma')
ax.add_collection(collection)

ani = animation.FuncAnimation(fig,animate,interval=(sim.dt*1000))

vis.mainloop()
