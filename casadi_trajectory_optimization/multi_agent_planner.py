import numpy as np
import casadi
from casadi import Opti

class MultiAgentPlanner():
    
    def __init__(self, dynamics, num_agents=1, num_timesteps=100, min_allowable_dist=1.0):
        self.solver_opts = {"ipopt.tol":1e-3, "expand":False,
                            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.dynamics = dynamics
        self.N = num_timesteps
        self.M = num_agents
        self.min_allowable_dist = min_allowable_dist
        self.obstacles = []

    def solve_opt(self):
        self.opti.solver('ipopt', self.solver_opts)
        sol = self.opti.solve()
        self.x_sol = [sol.value(self.x[m]) for m in range(self.M)]
        self.u_sol = [sol.value(self.u[m]) for m in range(self.M)]
        self.tf_sol = sol.value(self.tf)
        return self.x_sol, self.u_sol, self.tf_sol 

    def setup_mpc_opt(self, x0, xf, tf, Qf=None, x_bounds=None, u_bounds=None):
        '''
        x0: nx1 initial state
        xf: nx1 goal state
        tf: end time
        Qf: nxn weighting matrix for xf cost
        '''
        self.opti = Opti()
        self.x = [self.opti.variable(self.dynamics.x_shape, self.N+1) for m in range(self.M)]
        self.u = [self.opti.variable(self.dynamics.u_shape, self.N) for m in range(self.M)]
        self.tf = tf
        if Qf is None:
            Qf = np.eye(self.dynamics.x_shape)
        self.dt = self.tf / self.N

        mpc_cost = 0.
        for m in range(self.M):
            mpc_cost += (self.x[m][:,-1] - xf.reshape((-1,1))).T @ Qf @ (self.x[m][:,-1] - xf.reshape((-1,1)))
        self.opti.minimize(mpc_cost)

        self._add_dynamic_constraints()
        self._add_state_constraint(0, x0)
        self.add_x_bounds(x_bounds)
        self.add_u_bounds(u_bounds)
        self._add_obstacle_constraints()
        self._add_multi_agent_collision_constraints()

    def setup_min_time_opt(self, x0, xf, tf_guess=10.0, x_bounds=None, u_bounds=None):
        '''
        x0: nx1 initial state
        xf: nx1 goal state
        '''
        self.opti = Opti()
        self.x = [self.opti.variable(self.dynamics.x_shape, self.N+1) for m in range(self.M)]
        self.u = [self.opti.variable(self.dynamics.u_shape, self.N) for m in range(self.M)]
        self.tf = self.opti.variable()
        self.opti.set_initial(self.tf, tf_guess)
        self.opti.minimize(self.tf)
        self.dt = self.tf / self.N

        self._add_dynamic_constraints()
        self._add_state_constraint(0, x0)
        self._add_state_constraint(-1, xf)
        self.add_x_bounds(x_bounds)
        self.add_u_bounds(u_bounds)
        self._add_obstacle_constraints()
        self._add_multi_agent_collision_constraints()

    def add_x_bounds(self, x_bounds):
        ''' 
        x_bounds: n x 2 vector of x min and max (can be infinity)
        Constrain x to stay within x_bounds. 
        '''
        if x_bounds is None:
            return
        for i in range(self.dynamics.x_shape):
            if np.isinf(x_bounds[i,0]) and np.isinf(x_bounds[i,1]):
                continue
            for m in range(self.M):
                self.opti.subject_to(self.opti.bounded(x_bounds[i,0], self.x[m][i,:], x_bounds[i,1]))

    def add_u_bounds(self, u_bounds):
        ''' 
        u_bounds: m x 2 vector of u min and max (can be infinity)
        Constrain u to stay within u_bounds. 
        '''
        if u_bounds is None:
            return
        for i in range(self.dynamics.u_shape):
            for m in range(self.M):
                self.opti.subject_to(self.opti.bounded(u_bounds[i,0], self.u[m][i,:], u_bounds[i,1]))

    def add_u_diff_bounds(self, u_diff_bounds):
        ''' 
        u_diff_bounds: m x 2 vector of u min and max (can be infinity)
        Constrain difference of u to not excede this rate.
        Not fully tested.
        '''
        for i in range(self.dynamics.u_shape):
            if np.isinf(u_diff_bounds[i]):
                continue
            for m in range(self.M):
                for k in range(self.N - 1):
                    self.opti.subject_to(((self.u[m][i,k] - self.u[m][i,k+1])/self.dt)**2 <= (u_diff_bounds[i])**2)

    def add_u0_constraint(self, u0):
        ''' u0 = len n vector. Add constraint at k=0 on input. '''
        self._add_input_constraint(0, u0)

    def add_uf_constraint(self, uf):
        ''' uf = len n vector. Add constraint at final timestep on input. '''
        self._add_input_constraint(-1, uf)

    def _add_dynamic_constraints(self):
        ''' Constrain states at n and n+1 to follow dynamics '''
        for k in range(self.N):
            for m in range(self.M):
                self.opti.subject_to(self.x[m][:,k+1] == \
                                     self.dynamics.propagate(self.x[m][:,k], self.u[m][:,k], self.dt))

    def _add_state_constraint(self, k, x):
        ''' Used to constrain initial and final state conditions '''
        for i in range(self.dynamics.x_shape):
            for m in range(self.M):
                self.x[m][i,k]
                x[m,i]
                self.opti.subject_to(self.x[m][i,k] == x[m,i])
        
    def _add_input_constraint(self, k, u):
        ''' Used to constrain initial and final input conditions '''
        for i in range(self.dynamics.u_shape):
            for m in range(self.M):
                self.opti.subject_to(self.u[m][i,k] == u.item(i))

    def _add_obstacle_constraints(self):
        ''' Constraints physical state dimensions to not be within obstacle radii '''
        for ob in self.obstacles:
            # import ipdb; ipdb.set_trace()
            # print(self.dynamics.physical_state_idx)
            # [(x[j,:]) for i, j in enumerate(self.dynamics.physical_state_idx)]
            # [(ob['position'][i])**2 for i, j in enumerate(self.dynamics.physical_state_idx)]
            for m in range(self.M):
            # [(x[j,:] - ob['position'][i])**2 for i, j in enumerate(self.dynamics.physical_state_idx)]
                self.opti.subject_to(sum([(self.x[m][j,:] - ob['position'][i])**2 for i, j in enumerate(self.dynamics.physical_state_idx)]) >= ob['radius']**2)

    def _add_multi_agent_collision_constraints(self):
        for m1 in range(self.M):
            for m2 in range(self.M):
                if m1 == m2:
                    continue
                self.opti.subject_to((self.x[m1][0,:] - self.x[m2][0,:])**2+(self.x[m1][1,:] - self.x[m2][1,:])**2 >= self.min_allowable_dist) 
    
    def add_obstacles(self, obstacles):
        for ob in obstacles:
            self.add_obstacle(**ob)

    def add_obstacle(self, position, radius):
        self.obstacles.append({'position': position, 'radius': radius})

    def draw_path(self):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        colors = ['green', 'blue', 'red', 'orange', 'pink']

        for i in range(self.M):
            ax.plot(self.x_sol[i][self.dynamics.physical_state_idx[0],:], 
                    self.x_sol[i][self.dynamics.physical_state_idx[1],:],
                    color=colors[i])
        
        for ob in self.obstacles:
            ax.add_patch(plt.Circle(ob['position'], ob['radius'], facecolor='brown', edgecolor='k'))

        ax.set_aspect('equal')

        return fig, ax