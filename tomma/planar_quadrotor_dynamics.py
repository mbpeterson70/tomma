import numpy as np
import casadi
from tomma.dynamics import Dynamics
from casadi import sin, cos

class PlanarQuadrotorDynamics(Dynamics):
    
    def __init__(self):
        self.physical_state_idx = [0, 1]

        # Masses of center and right(and left) motors
        self.mc = 1
        self.mr = 0.25 # kg

        # Distance from center to motor
        self.d = 0.3 # m

        # inertia of VTOL's center
        self.Jc = 0.0042

        # Damping coefficient, mu
        self.mu = .1 # kg/s

        self.g = 9.81 # m/s^2

        self._x_shape = 6
        self._u_shape = 2
    
    def f(self, x, u):
        z, y, th = x[0], x[1], x[2]
        zdot, ydot, thdot = x[3], x[4], x[5]
        l_motor, r_motor = u[0], u[1]

        M = np.array([[self.mc + 2.0*self.mr, 0.0, 0.0],
                      [0.0, self.mc + 2.0*self.mr, 0.0],
                      [0.0, 0.0, self.Jc + 2.0*self.mr*self.d**2]])
        C = np.array([[-(r_motor+l_motor)*sin(th) - self.mu*zdot],
                      [(r_motor+l_motor)*cos(th) - (self.mc+2.0*self.mr)*self.g],
                      [(r_motor-l_motor)*self.d]])
        M_inv_C = np.linalg.inv(M) @ C
        zddot = M_inv_C.item(0)
        yddot = M_inv_C.item(1)
        thddot = M_inv_C.item(2)

        xdot = casadi.vertcat(
            zdot,
            ydot,
            thdot,
            zddot,
            yddot,
            thddot
        )
        return xdot    
    
    @property
    def x_shape(self):
        return self._x_shape
    
    @property
    def u_shape(self):
        return self._u_shape