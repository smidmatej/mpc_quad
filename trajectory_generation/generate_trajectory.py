import os
from . import uav_trajectory
import numpy as np


def main():
        

    waypoint_filename = 'waypoints/waypoints1.csv'
    output_trajectory_filename = 'trajectories/trajectory_sampled.csv'
    v_max = 20.0
    a_max = 10.0
    dt = 0.01
    create_trajectory_from_waypoints(waypoint_filename, output_trajectory_filename, v_max, a_max, dt)


def create_trajectory_from_waypoints(waypoint_filename, output_trajectory_filename, v_max, a_max, dt=0.01):
    polynom_filename = 'trajectory_generation/trajectories/polynomial_representation.csv'
    
    print("Loading waypoints from file: {}".format(waypoint_filename))
    print("Saving polynomial representation of trajectory to file: {}".format(polynom_filename))
    print("Maximum velocity: {}".format(v_max))
    print("Maximum acceleration: {}".format(a_max))

    os.system('./trajectory_generation/genTrajectory -i '+ waypoint_filename + ' -o ' + polynom_filename + ' --v_max ' + str(v_max) + ' --a_max ' + str(a_max))

    traj = uav_trajectory.Trajectory()
    print("Loading polynomial representation of trajectory from file: {}".format(polynom_filename))
    traj.loadcsv(polynom_filename)

    print(f'Saving sampled trajectory to file: {output_trajectory_filename} with dt={dt}')
    save_evals_csv(traj,output_trajectory_filename, dt=dt)
    

def save_evals_csv(traj, filename, dt=0.01):

    ts = np.arange(0, traj.duration, dt)
    evals = np.empty((len(ts), 15))
    for t, i in zip(ts, range(0, len(ts))):
        e = traj.eval(t)
        evals[i, 0:3]  = e.pos
        evals[i, 3:6]  = e.vel
        evals[i, 6:9]  = e.acc
    data = np.concatenate((ts.reshape(-1,1), evals), axis=1)
    np.savetxt(filename, data, fmt="%.6f", delimiter=",", header='t,x,y,z,vx,vy,vz,ax,ay,az')
    
if __name__ == '__main__':
    main()