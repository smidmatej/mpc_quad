from utils.save_dataset import load_dict
import matplotlib.pyplot as plt
import numpy as np

import os

def main():
    gpe = 1
    v_max = 10
    fname_first = 'data/simulation_gpe' + str(gpe) + 'v_max' + str(v_max) + '.pkl'

    first = 'python execute_trajectory.py --gpe 1 --trajectory 0 --v_max 20 --a_max 20 --show 0'
    second = 'python execute_trajectory.py --gpe 0 --new_trajectory 0 --v_max 20 --a_max 20 --show 0'
    os.system(first)
    os.system(second)
    filename_no_gpe = 'data/simulated_flight_gpe0.pkl'
    filename_gpe = 'data/simulated_flight_gpe1.pkl'

    dict_no_gpe = load_dict(filename_no_gpe)
    dict_gpe = load_dict(filename_gpe)

    v_no_gpe = np.linalg.norm(dict_no_gpe['v'], axis=1)
    v_max_no_gpe = np.max(v_no_gpe)
    v_gpe = np.linalg.norm(dict_gpe['v'], axis=1)
    v_max_gpe = np.max(v_gpe)

    #breakpoint()

    mean_rmse_pos_no_gpe = np.mean(dict_no_gpe['rmse_pos'])
    mean_rmse_pos_gpe = np.mean(dict_gpe['rmse_pos'])

    print(f'v_max_no_gpe: {v_max_no_gpe}')
    print(f'v_max_gpe: {v_max_gpe}')
    print(f'mean_rmse_pos_no_gpe: {mean_rmse_pos_no_gpe}')
    print(f'mean_rmse_pos_gpe: {mean_rmse_pos_gpe}')

    plt.figure(figsize=(10,6), dpi=100)
    plt.scatter(v_max_no_gpe, mean_rmse_pos_no_gpe, label='No GPE')
    plt.scatter(v_max_gpe, mean_rmse_pos_gpe, label='GPE')
    plt.show()

if __name__ == '__main__':
    main()
