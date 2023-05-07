import numpy as np
import casadi
from casadi import Opti

class MultiAgentPlanner():
    
    def __init__(self, dynamics, num_agents, num_timesteps, min_allowable_dist=1.0):
        self.solver_opts = {"ipopt.tol":1e-3, "expand":False,
                            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.dynamics = dynamics
        self.N = num_timesteps
        self.M = num_agents
        self.min_allowable_dist = min_allowable_dist
        self.obstacles = []
        
    def solve_min_time(self, x0, xf, u0=None, uf=None, x_bounds=None, u_bounds=None, u_diff_bounds=None):
        '''
        x0: nx1 initial state
        xf: nx1 goal state
        u_bounds: mx2 lower and upper bounds for each input
        '''
        opti = Opti()
        x = [opti.variable(self.dynamics.x_shape, self.N+1) for m in range(self.M)]
        u = [opti.variable(self.dynamics.u_shape, self.N) for m in range(self.M)]
        tf = opti.variable()
        # for m in range(self.M):
        opti.set_initial(tf, 10.0)

        opti.minimize(tf)
        
        dt = tf/self.N
        for k in range(self.N):
            for m in range(self.M):
                opti.subject_to(x[m][:,k+1] == self.dynamics.propagate(x[m][:,k], u[m][:,k],  dt))
        
        for i in range(self.dynamics.x_shape):
            for m in range(self.M):
                opti.subject_to(x[m][i,0] == x0[m,i])
                opti.subject_to(x[m][i,-1] == xf[m,i])

        opti.subject_to(tf > 0.0)

        if u0 is not None:
            for i in range(self.dynamics.u_shape):
                for m in range(self.M):
                    opti.subject_to(u[m][i,0] == u0.item(i))
        if uf is not None:
            for i in range(self.dynamics.u_shape):
                for m in range(self.M):
                    opti.subject_to(u[m][i,-1] == uf.item(i))

        if x_bounds is not None:
            for i in range(self.dynamics.x_shape):
                if np.isinf(x_bounds[i,0]) and np.isinf(x_bounds[i,1]):
                    continue
                for m in range(self.M):
                    opti.subject_to(opti.bounded(x_bounds[i,0], x[m][i,:], x_bounds[i,1]))
        
        if u_bounds is not None:
            for i in range(self.dynamics.u_shape):
                for m in range(self.M):
                    opti.subject_to(opti.bounded(u_bounds[i,0], u[m][i,:], u_bounds[i,1]))
            
        if u_diff_bounds is not None:
            for i in range(self.dynamics.u_shape):
                if np.isinf(u_diff_bounds[i]):
                    continue
                for m in range(self.M):
                    for k in range(self.N - 1):
                        opti.subject_to(((u[m][i,k] - u[m][i,k+1])/dt)**2 <= (u_diff_bounds[i])**2)

        # for ob in self.obstacles:
        #     opti.subject_to((x[0,:] - ob['position'][0])**2+(x[1,:] - ob['position'][1])**2 >= ob['radius']**2)
        for m1 in range(self.M):
            for m2 in range(self.M):
                if m1 == m2:
                    continue
                opti.subject_to((x[m1][0,:] - x[m2][0,:])**2+(x[m1][1,:] - x[m2][1,:])**2 >= self.min_allowable_dist)

        opti.solver('ipopt', self.solver_opts)

        sol = opti.solve()
        return [sol.value(x[m]) for m in range(self.M)], [sol.value(u[m]) for m in range(self.M)], sol.value(tf)
    
    def add_obstacles(self, obstacles):
        for ob in obstacles:
            self.add_obstacle(**ob)

    def add_obstacle(self, position, radius):
        self.obstacles.append({'position': position, 'radius': radius})