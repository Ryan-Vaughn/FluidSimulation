import numpy as np
import tkinter as tk
import sphsim as sph

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as mc

sim = sph.Simulation(100,2)

#Initialize plot
fig = plt.Figure()

def animate(i):
    sim.simulate()
    collection.set_offsets(sim.X)
    ax.add_collection(collection)

#Run the GUI
vis = tk.Tk()
vis.title('Smooth Particle Hydrodynamics Visualization')

canvas = FigureCanvasTkAgg(fig, master=vis)
canvas.get_tk_widget().grid(column=0,row=1)

ax = fig.add_subplot(111)
ax.set_xlim(sim.bounds[1,1], sim.bounds[0,1])
ax.set_ylim(sim.bounds[1,1], sim.bounds[0,1])

sizes = sim.eps * np.ones(sim.num_pts)
collection = mc.CircleCollection(sizes, offsets=sim.X, transOffset=ax.transData, color='green')
ax.add_collection(collection)

ani = animation.FuncAnimation(fig,animate,frames=100)

vis.mainloop()    