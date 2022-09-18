import quad
from tqdm import tqdm

import pdb

import numpy as np
from visualization import initialize_drone_plotter, draw_drone_simulation, trajectory_tracking_results


import os
#os.environ["PYTHONBREAKPOINT"] = "0"


real_time_artists = None
    
# Initialize real time plot stuff



if __name__ == "__main__":
	my_quad = quad.Quadrotor3D()
	real_time_artists = initialize_drone_plotter(n_props=10, quad_rad=my_quad.length,
                                                     world_rad=100)

	print(my_quad)

	simulation_steps = 100
	simulation_step_length = 0.1
	x_history = list()

	t = np.arange(0, simulation_steps*simulation_step_length, simulation_step_length)

	quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
	quad_trajectory = np.zeros((len(t), 13))
	#print(my_quad.get_state(quaternion=True))
	#print(range(t))
	for current_idx in tqdm(np.arange(1,simulation_steps)):
		u = np.ones((4,1))
		my_quad.update(u, simulation_step_length)

		#breakpoint()
		x_pred = np.zeros((13,1))
		quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
		quad_trajectory[current_idx, :] = np.expand_dims(quad_current_state, axis=0)

		breakpoint()
		#x_history.append(my_quad.get_state(quaternion=True))
		draw_drone_simulation(real_time_artists, quad_trajectory[:current_idx, :], my_quad, targets=None,
                                  targets_reached=None, pred_traj=x_pred)




