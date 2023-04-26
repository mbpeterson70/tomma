import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, NonNegativeReals, Var, VarList, \
    Constraint, ConstraintList, TransformationFactory, SolverFactory, Objective
from pyomo.environ import cos, sin
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

        # define the differential equation as constrainta
        m.ode_z = Constraint(m.t, rule=lambda m, t: m.z_dot[t] == m.tf*(m.v[t]*cos(m.th[t])))
        m.ode_y = Constraint(m.t, rule=lambda m, t: m.y_dot[t] == m.tf*(m.v[t]*sin(m.th[t])))
        m.ode_th = Constraint(m.t, rule=lambda m, t: m.th_dot[t] == m.tf*(m.u_th[t]))
        m.ode_v = Constraint(m.t, rule=lambda m, t: m.v_dot[t] == m.tf*(m.u_v[t]))

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
        m.integral = Integral(m.t, wrt=m.t, rule=lambda m, t: m.tf*(1 + self.R[0,0]*(m.u_v[t])**2 + self.R[1,1]*m.u_th[t]**2))
        m.obj = Objective(expr = m.integral)

        # transform and solve
        TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=1000)
        #discretizer = TransformationFactory('dae.collocation')
        #discretizer.apply_to(m,wrt=m.t,nfe=500,ncp=3,scheme='LAGRANGE-RADAU')
        solver = SolverFactory('ipopt')
        #discretizer.reduce_collocation_points(m,var=m.u,ncp=1,contset=m.t)
        solver.options['max_iter']= 10000 #number of iterations you wish
        solver.solve(m).write()
        self.m = m
        
    def plot_top_view(self):
        fig, ax = plt.subplots()
        t = np.array([t*self.m.tf() for t in self.m.t])
        z = np.array([self.m.z[t]() for t in self.m.t])
        y = np.array([self.m.y[t]() for t in self.m.t])
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
        state_names = ['z', 'y', 'v', r'$theta$']
        for i, (state_var, name) in enumerate(zip(state_vars, state_names)):
            ax[i].plot(t, state_var)
            ax[i].set_ylabel(name)
        plt.show()
            
        
if __name__ == '__main__':
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    xf = np.array([5.0, 0.0, 0.0, np.pi])
    tg = TrajectoryGenerator(x0, xf, np.diag([1, 1]))
    tg.solve()
    tg.plot_top_view()
    tg.plot_states()
