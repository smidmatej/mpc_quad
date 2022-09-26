import numpy as np
import casadi as cs
import os
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

from quad import Quadrotor3D
from utils import skew_symmetric, quaternion_to_euler, unit_quat, v_dot_q

from quad_opt import quad_optimizer


x0 = np.array([0,0,0, 1,0,0,0, 0,0,0, 0,0,0])





quad_opt = quad_optimizer(Quadrotor3D(), optimization_dt=0.1, n_nodes=100)

yref, yref_N = quad_opt.set_reference_state(np.array([-1,1,5, 1,0.1,0,0, 0,0,0, 0,0,0]))

x_opt_acados, w_opt_acados = quad_opt.run_optimization(x0)

print(quad_opt.acados_ocp_solver.get_cost())

fig = plt.figure()
plt.plot(x_opt_acados[:,0],'r')
plt.plot(x_opt_acados[:,1],'g')
plt.plot(x_opt_acados[:,2],'b')


plt.plot(np.concatenate((yref[:quad_opt.n_nodes,0], [yref_N[0]])),'r--')
plt.plot(np.concatenate((yref[:quad_opt.n_nodes,1], [yref_N[1]])),'g--')
plt.plot(np.concatenate((yref[:quad_opt.n_nodes,2], [yref_N[2]])), 'b--')
plt.show()

fig = plt.figure()
plt.plot(w_opt_acados[:,0],'r')
plt.plot(w_opt_acados[:,1],'g')
plt.plot(w_opt_acados[:,2],'b')
plt.plot(w_opt_acados[:,3],'c')

plt.show()
