import numpy as np
import matplotlib.pyplot as plt

from planar_quadrotor_dynamics import PlanarQuadrotorDynamics
from multi_agent_planner import MultiAgentPlanner

quadrotor = PlanarQuadrotorDynamics()
deg2rad = np.pi/180
x_bounds = np.array([
    [-20., 20.],
    [-20., 20.],
    [-np.pi/4, np.pi/4.],
    [-5, 5],
    [-2, 2],
    [-np.inf, np.inf]
])
u_bounds = np.array([
    [0, 20],
    [0, 20]
])

planner = MultiAgentPlanner(quadrotor, 1, 100)
x0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
xf = np.array([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
planner.add_obstacle([2.5, 0.0], .3)


try:
    x, u, tf = planner.solve_min_time(x0, xf, x_bounds=x_bounds, u_bounds=u_bounds)
except:
    planner.x_sol = [planner.opti.debug.value(xi) for xi in planner.x]
    planner.u_sol = [planner.opti.debug.value(ui) for ui in planner.x]
    planner.tf_sol = planner.opti.debug.value(planner.tf)
    planner.draw_path()
# x, u, tf = planner.solve_min_time(x0, xf, x_bounds=x_bounds, u_bounds=u_bounds)


planner.draw_path()
fig, ax = plt.subplots()
x = planner.x_sol
tf = planner.tf_sol
for i in range(planner.M):
    ax.plot(np.arange(x[i].shape[1])*tf/x[i].shape[1], x[i][2,:])
    
plt.show()