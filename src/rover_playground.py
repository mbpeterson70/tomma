import numpy as np
import matplotlib.pyplot as plt

from path_planner import PathPlanner
from dubins_dynamics import DubinsDynamics

dubins = DubinsDynamics()
planner = PathPlanner(dubins, 100)
deg2rad = np.pi/180
x0 = np.array([0.0, 0.0, 0.0])
xf = np.array([2.0, 0.0, 2*np.pi/3])
u_bounds = np.array([
    [-1, 1.],
    [-45*deg2rad, 45*deg2rad]
])
x, u, tf = planner.solve_min_time(x0, xf, u_bounds)
fig, ax = plt.subplots()
ax.plot(np.arange(x.shape[1])*tf/x.shape[1], x[0,:])
ax.plot(np.arange(x.shape[1])*tf/x.shape[1], x[1,:])
ax.plot(np.arange(x.shape[1])*tf/x.shape[1], x[2,:])

fig, ax = plt.subplots()
ax.plot(np.arange(u.shape[1])*tf/u.shape[1], u[0,:])
ax.plot(np.arange(u.shape[1])*tf/u.shape[1], u[1,:])

fig, ax = plt.subplots()
ax.plot(x[0,:], x[1,:])

plt.show()