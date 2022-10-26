import os

import sys

try:
    # Methods of this script are called from elsewhere
    import trajectory_generation.uav_trajectory as uav_trajectory
except ImportError:
    # This file is executed as main 
    import uav_trajectory as uav_trajectory
import numpy as np


def main():
        
    execution_path = os.path.dirname(os.path.realpath(__file__))
    #print(f'Execution path: {execution_path}')

    # Waypoints specifiing trajectory to follow
    waypoint_filename = execution_path + '/waypoints/waypoints1.csv'

    # Trajectory represented as a sequence of states
    output_trajectory_filename = execution_path + '/trajectories/trajectory_sampled.csv'


    hsize = float(sys.argv[1])
    num_waypoints = int(sys.argv[2])
    

    #print('Number of arguments:', len(sys.argv), 'arguments.'
    #print 'Argument List:', str(sys.argv)
    generate_random_waypoints(waypoint_filename, hsize=hsize, num_waypoints=num_waypoints)
    #create_trajectory_from_waypoints(waypoint_filename, output_trajectory_filename, v_max, a_max, dt)


def generate_random_waypoints(waypoint_filename, hsize=10, num_waypoints=10):
    # generate random waypoints
    print(f'Generating {num_waypoints} random waypoints in a {hsize}x{hsize} random walk and saving them to {waypoint_filename}')
    waypoints = list()
    waypoints.append(np.array([0.0, 0.0, 0.0]))
    for i in range(num_waypoints):
        newWaypoint = np.random.uniform(-hsize, hsize, 3)
        waypoints.append(newWaypoint)
    waypoints.append(np.array([0.0, 0.0, 3.0]))
    waypoints.append(np.array([0.0, 0.0, 0.0]))
    #waypoints.append(np.array([0.0, 0.0, 0.0]))
    np.savetxt(waypoint_filename, waypoints, fmt="%.6f", delimiter=",")


def create_trajectory_from_waypoints(waypoint_filename, output_trajectory_filename, v_max, a_max, dt=0.01):

    execution_path = os.path.dirname(os.path.realpath(__file__))


    # Trajectory represented as a polynomial
    polynom_filename = execution_path + '/trajectories/polynomial_representation.csv'


    print("Loading waypoints from file: {}".format(waypoint_filename))
    print("Saving polynomial representation of trajectory to file: {}".format(polynom_filename))
    #print("Maximum velocity: {}".format(v_max))
    #print("Maximum acceleration: {}".format(a_max))
    

    print(f"Executing: {execution_path +  '/genTrajectory -i '+ waypoint_filename + ' -o ' + polynom_filename + ' --v_max ' + str(v_max) + ' --a_max ' + str(a_max)}")
    os.system(execution_path +  '/genTrajectory -i '+ waypoint_filename + ' -o ' + polynom_filename + ' --v_max ' + str(v_max) + ' --a_max ' + str(a_max))
    #os.system(execution_path +  '/genTrajectory -i '+ waypoint_filename + ' -o ' + polynom_filename)

    traj = uav_trajectory.Trajectory()
    print("Loading polynomial representation of trajectory from file: {}".format(polynom_filename))
    traj.loadcsv(polynom_filename)

    
    print(f'Saving sampled trajectory to file: {output_trajectory_filename} with dt={dt}')

    save_evals_csv(traj,output_trajectory_filename, dt=dt)
    

def save_evals_csv(traj, filename, dt=0.01):

    #traj.stretchtime(0.1)
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