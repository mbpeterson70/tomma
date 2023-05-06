import numpy as np
import casadi

CONTROL_LIN_VEL_ANG_VEL = 0

def scalar_mult(scalar, vec):
    # res = np.zeros((vec.shape))
    # for i in range(len(vec)):
    #     import ipdb; ipdb.set_trace()
    res = np.array([scalar*vec.item(i) for i in range(len(vec))]).reshape(vec.shape)
    return res

def vector_add(vec1, vec2):
    # import ipdb; ipdb.set_trace()
    # res = np.zeros((vec1.shape))
    # for i in range(vec):
    #     res[i] = vec1[i] + vec2[i]
    if type(vec1) == casadi.casadi.MX and type(vec2) == np.ndarray:
        res = np.array([vec1[i] + vec2.item(i) for i in range(vec1.shape[0])]).reshape(vec1.shape)
    elif type(vec1) == np.ndarray and type(vec2) == np.ndarray:
        res = np.array([vec1.item(i) + vec2.item(i) for i in range(vec1.shape[0])]).reshape(vec1.shape)
    return res

class DubinsDynamics():
    
    def __init__(self, control=CONTROL_LIN_VEL_ANG_VEL):
        self.control = control
        assert self.control == CONTROL_LIN_VEL_ANG_VEL
    
    def f(self, x, u):
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            # z, y, th = x.reshape(-1)
            # v, thdot = u.reshape(-1)
            if type(x) == casadi.casadi.MX:
                z, y, th = x[0], x[1], x[2]
            # else:
            #     z, y, th = x.item(0), x.item(1), x.item(2)
            v, thdot = u[0], u[1]
            
            # xdot = np.array([
            #     [v*np.cos(th)],
            #     [v*np.sin(th)],
            #     [thdot]
            # ])
            xdot = casadi.vertcat(
                v*np.cos(th),
                v*np.sin(th),
                thdot
            )
        return xdot

    def propagate(self, x, u, dt):
        k1 = self.f(x, u)
        # k2 = self.f(vector_add(x, scalar_mult(.5*dt, k1)), u)
        # k3 = self.f(vector_add(x, scalar_mult(.5*dt, k2)), u)
        # k4 = self.f(vector_add(x, scalar_mult(dt, k3)), u)
        # import ipdb; ipdb.set_trace()
        k2 = self.f(x + .5*dt*k1, u)
        k3 = self.f(x + .5*dt*k2, u)
        k4 = self.f(x + dt*k3, u)
        return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        return vector_add(x, scalar_mult(dt/6, vector_add(k1, vector_add(2*k2, vector_add(2*k3, k4)))))
    
    @property
    def x_shape(self):
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            return 3
    
    @property
    def u_shape(self):
        if self.control == CONTROL_LIN_VEL_ANG_VEL:
            return 2