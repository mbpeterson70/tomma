import numpy as np
import casadi
from tomma.dynamics import Dynamics

"""
CONTROL_LIN_VEL_ANG_VEL:
    x = [x, y, theta]
    u = [v, thetadot]
    
CONTROL_LIN_ACC_ANG_VEL:
    x = [x, y, v, theta]
    u = [vdot, thetadot]
"""

CONTROL_LIN_VEL_ANG_VEL = 0
CONTROL_LIN_ACC_ANG_VEL = 1

class DubinsDynamics(Dynamics):
    """
    Variable velocity Dubins car dynamics
    """
    
    def __init__(self, control=CONTROL_LIN_VEL_ANG_VEL):
        """
        Constructor.

        Args:
            control (enum, optional): Control type. See file header for description. Defaults to 
                CONTROL_LIN_VEL_ANG_VEL.
        """
        self.control = control
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            self.physical_state_idx = [0, 1]
        elif self.control == CONTROL_LIN_ACC_ANG_VEL:
            self.physical_state_idx = [0, 1]
    
    def f(self, x, u):
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            z, y, th = x[0], x[1], x[2]
            v, thdot = u[0], u[1]
            
            xdot = casadi.vertcat(
                v*np.cos(th),
                v*np.sin(th),
                thdot
            )
        elif self.control == CONTROL_LIN_ACC_ANG_VEL:
            z, y, v, th = x[0], x[1], x[2], x[3]
            vdot, thdot = u[0], u[1]
            xdot = casadi.vertcat(
                v*np.cos(th),
                v*np.sin(th),
                vdot,
                thdot
            )
        return xdot    
    
    @property
    def x_shape(self):
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            return 3
        elif self.control == CONTROL_LIN_ACC_ANG_VEL:
            return 4
    
    @property
    def u_shape(self):
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            return 2
        elif self.control == CONTROL_LIN_ACC_ANG_VEL:
            return 2