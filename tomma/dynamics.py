class Dynamics():

    def __init__(self, physical_state_idx):
        self.physical_state_idx = physical_state_idx

    def propagate(self, x, u, dt):
        k1 = self.f(x, u)
        k2 = self.f(x + .5*dt*k1, u)
        k3 = self.f(x + .5*dt*k2, u)
        k4 = self.f(x + dt*k3, u)
        return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def f(self, x, u):
        raise NotImplementedError

    @property
    def x_shape(self):
        raise NotImplementedError
    
    @property
    def u_shape(self):
        raise NotImplementedError