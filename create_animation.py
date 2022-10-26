from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from gp.data_loader import data_loader
from utils.utils import v_dot_q


def update(i):
    #particle.set_data(x[i],y[i])

    # Current position
    particle.set_data_3d(position[i,0], position[i,1], position[i,2])

    # Visited positions
    traj.set_data_3d(position[:i+1,0], position[:i+1,1], position[:i+1,2])

    # orientation
    vector_up.set_data_3d(np.array([position[i,0], position[i,0] + body_up[i,0]]), \
            np.array([position[i,1], position[i,1] + body_up[i,1]]), \
            np.array([position[i,2], position[i,2] + body_up[i,2]]))

    # Norm of veloctity to plot to a different ax
    v_traj.set_data(t[:i+1], speed[:i+1])

    return particle,traj,v_traj

def update_speed(i):

    return v_traj


trajectory_filename = 'data/trajectory.pkl'
dloader = data_loader(trajectory_filename, compute_reduction=200)


position = dloader.p
orientation = dloader.q
speed = np.linalg.norm(dloader.v, axis=1)

body_up = np.array([v_dot_q(np.array([0,0,10]), orientation[i,:]) for i in range(orientation.shape[0])])

t = dloader.t
number_of_frames = position.shape[0]

    
animation.writer = animation.writers['ffmpeg']
plt.ioff() # Turn off interactive mode to hide rendering animations

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')





#ax.set_ylim((0,40))
particle, = plt.plot([],[], marker='o', color='r')
vector_up, = plt.plot([],[], color='g')
traj, = plt.plot([],[], color='r', alpha=0.5)


min_lim = min(min(position[:,0]), min(position[:,1]), min(position[:,2]))
max_lim = max(max(position[:,0]), max(position[:,1]), max(position[:,2]))
ax.set_xlim((min_lim, max_lim))
ax.set_ylim((min_lim, max_lim))
ax.set_zlim((min_lim, max_lim))

ax.set_xlabel('Position x [m]')
ax.set_ylabel('Position y [m]')
ax.set_zlabel('Position z [m]')
ax.set_title('Trajectory')



ax_speed = fig.add_subplot(212)
v_traj, = plt.plot([],[], color='c')
ax_speed.set_xlim((0, max(t)))
ax_speed.set_ylim((0, max(speed)))
ax_speed.set_xlabel('Time [s]')
ax_speed.set_ylabel('Speed [m/s]')

ani = animation.FuncAnimation(fig, update, frames=number_of_frames, interval=25)
ani.save('animations/my_animation.mp4')
fig.tight_layout()
plt.show()
