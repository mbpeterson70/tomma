import numpy as np

from path_planner import PathPlanner
from dubins_dynamics import DubinsDynamics

dubins = DubinsDynamics()
planner = PathPlanner(dubins, 100)
x0 = np.array([0.0, 0.0, 0.0])
xf = np.array([2.0, 2*np.pi/3, 0.0])
# u_bounds = np.array([
#     [-1., 1.],
#     [-.5, .5]
# ])
u_bounds = np.array([
    [-1, 1.],
    [-.5, .5]
])
# planner.solve_min_time(x0, xf)
planner.solve_min_time(x0, xf, u_bounds)