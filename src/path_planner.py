import numpy as np
import casadi
from casadi import Opti

class PathPlanner():
    
    def __init__(self, dynamics, num_timesteps):
        self.solver_opts = {"ipopt.tol":1e-3, "expand":False,
                            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.dynamics = dynamics
        self.N = num_timesteps
        
    def solve_min_time(self, x0, xf, u0=None, uf=None, x_bounds=None, u_bounds=None, u_diff_bounds=None):
        '''
        x0: nx1 initial state
        xf: nx1 goal state
        u_bounds: mx2 lower and upper bounds for each input
        '''
        opti = Opti()
        x = opti.variable(self.dynamics.x_shape, self.N+1)
        u = opti.variable(self.dynamics.u_shape, self.N)
        tf = opti.variable()
        opti.set_initial(tf, 10.0)

        opti.minimize(tf)
        
        dt = tf/self.N
        for k in range(self.N):
            opti.subject_to(x[:,k+1] == self.dynamics.propagate(x[:,k], u[:,k],  dt))
        
        for i in range(self.dynamics.x_shape):
            opti.subject_to(x[i,0] == x0.item(i))
            opti.subject_to(x[i,-1] == xf.item(i))
        opti.subject_to(tf > 0.0)

        if u0 is not None:
            for i in range(self.dynamics.u_shape):
                opti.subject_to(u[i,0] == u0.item(i))
        if uf is not None:
            for i in range(self.dynamics.u_shape):
                opti.subject_to(u[i,-1] == uf.item(i))

        if x_bounds is not None:
            for i in range(self.dynamics.x_shape):
                opti.subject_to(opti.bounded(x_bounds[i,0], x[i,:], x_bounds[i,1]))
        
        if u_bounds is not None:
            for i in range(self.dynamics.u_shape):
                opti.subject_to(opti.bounded(u_bounds[i,0], u[i,:], u_bounds[i,1]))
            
        if u_diff_bounds is not None:
            for i in range(self.dynamics.u_shape):
                if np.isinf(u_diff_bounds[i]):
                    continue
                for k in range(self.N - 1):
                    opti.subject_to(((u[i,k] - u[i,k+1])/dt)**2 <= (u_diff_bounds[i])**2)

        opti.solver('ipopt', self.solver_opts)

        sol = opti.solve()
        return sol.value(x), sol.value(u), sol.value(tf)