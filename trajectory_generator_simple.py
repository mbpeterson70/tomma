import numpy as np
import matplotlib.pyplot as plt
import time

from pyomo.environ import ConcreteModel, NonNegativeReals, Var, VarList, \
    Constraint, ConstraintList, TransformationFactory, SolverFactory, Objective, \
    Binary, Integers, Boolean
from pyomo.environ import cos, sin, sqrt
from pyomo.dae import ContinuousSet, DerivativeVar, Integral 

class TrajectoryGenerator():
    
    def __init__(self, x0, xf, R):
        self.x0 = x0
        self.xf = xf
        self.R = R
        
    def solve(self):
        # setup problem inputs
        z0, y0, v0, th0 = self.x0.reshape(-1).tolist()
        zf, yf, vf, thf = self.xf.reshape(-1).tolist()
        
        # create a model object
        m = ConcreteModel()

        # define the independent variable
        m.tf = Var(domain=NonNegativeReals)#, initialize = tf_estimate)
        m.t = ContinuousSet(bounds=(0, 1))

        # define control inputs
        m.u_th = Var(m.t, initialize = 0.1, bounds=(-5,5))
        m.u_v = Var(m.t, initialize = v0, bounds=(-1,1))

        # define the dependent variables
        m.z = Var(m.t, initialize = z0)
        m.y = Var(m.t, initialize = y0)
        m.th = Var(m.t, initialize = th0)
        m.v = Var(m.t, initialize=v0)

        # define derivatives
        m.z_dot = DerivativeVar(m.z)
        m.y_dot = DerivativeVar(m.y)
        m.th_dot = DerivativeVar(m.th)
        m.v_dot = DerivativeVar(m.v)

        # m.b = [Var(domain=Binary) for i in range(4)]
        

        # define the differential equation as constrainta
        m.ode_z = Constraint(m.t, rule=lambda m, t: m.z_dot[t] == m.tf*(m.v[t]*cos(m.th[t])))
        m.ode_y = Constraint(m.t, rule=lambda m, t: m.y_dot[t] == m.tf*(m.v[t]*sin(m.th[t])))
        m.ode_th = Constraint(m.t, rule=lambda m, t: m.th_dot[t] == m.tf*(m.u_th[t]))
        m.ode_v = Constraint(m.t, rule=lambda m, t: m.v_dot[t] == m.tf*(m.u_v[t]))
        
        # Object constraints
        M = 1e5
        m.b0 = Var(m.t, within=Boolean, initialize=1, bounds=(0,1))
        m.b1 = Var(m.t, within=Boolean, initialize=0, bounds=(0,1))
        m.b2 = Var(m.t, within=Boolean, initialize=1, bounds=(0,1))
        m.b3 = Var(m.t, within=Boolean, initialize=1, bounds=(0,1))
        m.box1 = Constraint(m.t, rule=lambda m, t: m.z[t] >= 4 - m.b0[t]*M)
        m.box2 = Constraint(m.t, rule=lambda m, t: m.z[t] <= 2 + m.b1[t]*M)
        m.box3 = Constraint(m.t, rule=lambda m, t: m.y[t] >= 1 - m.b2[t]*M)
        m.box4 = Constraint(m.t, rule=lambda m, t: m.y[t] <= -1 + m.b3[t]*M)
        m.box5 = Constraint(m.t, rule=lambda m, t: m.b0[t] + m.b1[t] + m.b2[t] + m.b3[t] <= 3)
        
        # Norm constraints
        # m.dist_to_obj = Var(m.t, domain=NonNegativeReals, initialize=2.5)
        # m.dist_def = Constraint(m.t, rule=lambda m, t: m.dist_to_obj[t]**2 == (m.z[t] - 2.5)**2 + (m.y[t] - 0.0)**2)
        # m.not_in_obj = Constraint(m.t, rule=lambda m, t: abs(m.z[t] - 3) + abs(m.y[t] - 0.0) >= 0.01)

        # initial conditions
        m.pc = ConstraintList()
        m.pc.add(m.z[0] == z0)
        m.pc.add(m.y[0] == y0)
        m.pc.add(m.th[0] == th0)
        m.pc.add(m.v[0] == v0)

        # final conditions
        m.pc.add(m.z[1] == zf)
        m.pc.add(m.y[1] == yf)
        m.pc.add(m.th[1] == thf)
        m.pc.add(m.v[1] == vf)

        # define the optimization objective
        # m.integral = Integral(m.t, wrt=m.t, rule=lambda m, t: m.tf*(1 + self.R[0,0]*(m.u_v[t])**2 + self.R[1,1]*m.u_th[t]**2 + 100/((m.z[t]-2.5)**2 + m.y[t]**2)
        #                       + 1/((m.z[t]-2.5)**2 + (m.y[t]-1)**2) + 100/((m.z[t]-2.5)**2 + (m.y[t]+1)**2)))
        m.integral = Integral(m.t, wrt=m.t, rule=lambda m, t: m.tf*(1 + self.R[0,0]*(m.u_v[t])**2 + self.R[1,1]*m.u_th[t]**2))
        m.obj = Objective(expr = m.integral)

        # transform and solve
        TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=1000)
        # discretizer = TransformationFactory('dae.collocation')
        # discretizer.apply_to(m,wrt=m.t,nfe=500,ncp=3,scheme='LAGRANGE-RADAU')
        solver = SolverFactory('mindtpy')
        # discretizer.reduce_collocation_points(m,var=m.u,ncp=1,contset=m.t)
        # solver.options['max_iter']= 10000 #number of iterations you wish
        solver.solve(m, mip_solver='glpk', nlp_solver='ipopt', time_limit=60).write()
        self.m = m
        
    def plot_top_view(self):
        fig, ax = plt.subplots()
        t = np.array([t*self.m.tf() for t in self.m.t])
        z = np.array([self.m.z[t]() for t in self.m.t])
        y = np.array([self.m.y[t]() for t in self.m.t])
        ax.fill([2, 4, 4, 2], [1, 1, -1, -1], 'yellow')
        ax.plot(z,y)
        fig.set_dpi(240)
        plt.show()
        
    def plot_states(self):
        t = np.array([t*self.m.tf() for t in self.m.t])
        z = np.array([self.m.z[t]() for t in self.m.t])
        y = np.array([self.m.y[t]() for t in self.m.t])
        v = np.array([self.m.y[t]() for t in self.m.t])
        th = np.array([self.m.y[t]() for t in self.m.t])
        
        fig, (ax) = plt.subplots(4,1)
        fig.set_dpi(240)
        state_vars = [z, y, v, th]
        state_names = ['z', 'y', 'v', r'$\theta$']
        for i, (state_var, name) in enumerate(zip(state_vars, state_names)):
            ax[i].plot(t, state_var)
            ax[i].set_ylabel(name)
        plt.show()

    def plot_bs(self):
        t = np.array([t*self.m.tf() for t in self.m.t])
        b0 = np.array([self.m.b0[t]() for t in self.m.t])
        b1 = np.array([self.m.b1[t]() for t in self.m.t])
        b2 = np.array([self.m.b2[t]() for t in self.m.t])
        b3 = np.array([self.m.b3[t]() for t in self.m.t])
        plt.plot(t, b0)
        plt.plot(t, b1)
        plt.plot(t, b2)
        plt.plot(t, b3)  
        plt.legend(['b0', 'b1', 'b2', 'b3'])  
        plt.ylim([-.1, 1.1])    
        plt.show()
            
        
if __name__ == '__main__':
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    xf = np.array([5.0, 0.0, 0.0, np.pi/2])
    tg = TrajectoryGenerator(x0, xf, np.diag([1, 1]))
    tg.solve()
    tg.plot_top_view()
    tg.plot_states()
    tg.plot_bs()
