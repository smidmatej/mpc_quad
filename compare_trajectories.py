from utils.save_dataset import load_dict
import matplotlib.pyplot as plt
import numpy as np

import os
import json

def main():


    compare_config_file = 'comparisson_config.json'
    with open(compare_config_file) as json_file:
        config = json.load(json_file)

    
    keys = ['gpe', 'no_gpe']
    mean_rmse_pos = dict.fromkeys(keys)
    v_max = dict.fromkeys(keys)
    for key in keys:
        mean_rmse_pos[key] = [None]*len(config['runs'])
        v_max[key] = [None]*len(config['runs'])


    for n, run in zip(range(len(config['runs'])), config['runs']):
        print(run)
        simulation_result_fname = 'data/sim_' + str(run['gpe']) + '_trajectory' + str(run['trajectory']) + \
                                    '_v_max' + str(run['v_max']) + '_a_max' + str(run['a_max']) + '.pkl'

        os.system('python execute_trajectory.py -o ' + simulation_result_fname + ' --gpe ' + str(run['gpe']) + \
                    ' --trajectory ' + str(run['trajectory']) + ' --v_max ' + str(run['v_max']) \
                        + ' --a_max '+ str(run['a_max']) + ' --show 0')

        
        sim_result_dict = load_dict(simulation_result_fname)

        # leave out the last second of the simulation, because it tries to stop in place
        n_drop = int(1/sim_result_dict['dt'])
        print(f'Dropping last {n_drop} samples')
        v_norm = np.linalg.norm(sim_result_dict['v'][:-n_drop,:], axis=1)

        connect_dict = {'gpe':1, 'no_gpe':0}
        for key in keys:
            if connect_dict[key] == run['gpe']:
                v_max[key][n] = np.max(v_norm)
                mean_rmse_pos[key][n] = np.mean(sim_result_dict['rmse_pos'][:-n_drop])





        print(f'v_max: {v_max}')
        print(f'mean_rmse_pos: {mean_rmse_pos}')


    colors = {keys[0]: 'r', keys[1]: 'b'}
    plt.figure(figsize=(10,6), dpi=100)
    for key in keys:
        plt.scatter(v_max[key], mean_rmse_pos[key], c=colors[key], label=key)

    plt.show()

if __name__ == '__main__':
    main()
