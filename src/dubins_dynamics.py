import numpy as np
import casadi

CONTROL_LIN_VEL_ANG_VEL = 0

class DubinsDynamics():
    
    def __init__(self, control=CONTROL_LIN_VEL_ANG_VEL):
        self.control = control
        assert self.control == CONTROL_LIN_VEL_ANG_VEL
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            self.physical_state_idx = [0, 1]
    
    def f(self, x, u):
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            if type(x) == casadi.casadi.MX:
                z, y, th = x[0], x[1], x[2]
            v, thdot = u[0], u[1]
            
            xdot = casadi.vertcat(
                v*np.cos(th),
                v*np.sin(th),
                thdot
            )
        return xdot

    def propagate(self, x, u, dt):
        k1 = self.f(x, u)
        k2 = self.f(x + .5*dt*k1, u)
        k3 = self.f(x + .5*dt*k2, u)
        k4 = self.f(x + dt*k3, u)
        return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    @property
    def x_shape(self):
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            return 3
    
    @property
    def u_shape(self):
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            return 2