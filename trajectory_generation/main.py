

import os
import uav_trajectory
import numpy as np


def main():
        
    trajectory_filename = 'trajectories/traj1.csv'

    os.system('./genTrajectory -i waypoints/waypoints1.csv -o ' + trajectory_filename + ' --v_max 2.0 --a_max 1.0')



    traj = uav_trajectory.Trajectory()
    traj.loadcsv(trajectory_filename)
    output_trajectory_filename = 'trajectories/trajectory_sampled.csv'
    save_evals_csv(traj,output_trajectory_filename)
    #os.system('python plot_trajectory.py trajectories/traj1.csv')




def save_evals_csv(traj, filename):

    ts = np.arange(0, traj.duration, 0.01)
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