import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np

import matplotlib.pyplot as plt



file = open("x_opt.p",'rb')
x = pickle.load(file)

 

fig = plt.figure()
ax = plt.axes(projection='3d')
#plt.plot(x[:,0], x[:,1], x[:,2])

line, = ax.plot(x[0:1,0], x[0:1,1], x[0:1,2])

quad = ax.scatter([], [], [])


ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.set_zlim([-10,10])
def update_points(idx, x, line, quad):
    
    # update properties
    line.set_data(x[:idx,0], x[:idx,1])
    line.set_3d_properties(x[:idx,2])
    
    
    offsets = (x[idx,0], x[idx,1], x[idx,2])
    print(offsets)
    quad._offsets3d = offsets

    
    
ani=animation.FuncAnimation(fig, update_points, fargs=(x, line, quad))
#ani.save(r'AnimationNew.mp4')
plt.show()