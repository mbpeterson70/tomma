import numpy as np
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, NonNegativeReals, Var, VarList, \
    Constraint, ConstraintList, TransformationFactory, SolverFactory, Objective
from pyomo.environ import cos, sin
from pyomo.dae import ContinuousSet, DerivativeVar, Integral

# magic numbers
# TODO: need to go
V = 5

class TrajectoryGenerator():
    
    def __init__(self, x0, xf, R):
        self.x0 = x0
        self.xf = xf
        self.R = R
        
    def solve(self):
        # setup problem inputs
        z0, y0, th0 = self.x0.reshape(-1).tolist()
        zf, yf, thf = self.xf.reshape(-1).tolist()
        
        # create a model object
        m = ConcreteModel()

        # define the independent variable
        m.tf = Var(domain=NonNegativeReals)#, initialize = tf_estimate)
        m.t = ContinuousSet(bounds=(0, 1))

        # define control inputs
        m.u = Var(m.t, initialize = 0.1, bounds=(-5,5))

        # define the dependent variables
        m.z = Var(m.t, initialize = z0)
        m.y = Var(m.t, initialize = y0)
        m.th = Var(m.t, initialize = th0)

        # define derivatives
        m.z_dot = DerivativeVar(m.z)
        m.y_dot = DerivativeVar(m.y)
        m.th_dot = DerivativeVar(m.th)

        # define the differential equation as constrainta
        m.ode_z = Constraint(m.t, rule=lambda m, t: m.z_dot[t] == m.tf*(V*cos(m.th[t])))
        m.ode_y = Constraint(m.t, rule=lambda m, t: m.y_dot[t] == m.tf*(V*sin(m.th[t])))
        m.ode_th = Constraint(m.t, rule=lambda m, t: m.th_dot[t] == m.tf*(m.u[t]))

        # initial conditions
        m.pc = ConstraintList()
        m.pc.add(m.z[0]==z0)
        m.pc.add(m.y[0]==y0)
        m.pc.add(m.th[0]==th0)

        # final conditions
        m.pc.add(m.z[1]==zf)
        m.pc.add(m.y[1]==yf)
        m.pc.add(m.th[1]==thf)

        # define the optimization objective
        m.integral = Integral(m.t, wrt=m.t, rule=lambda m, t: m.tf*(1 + self.R*(m.u[t])**2))
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
        
    def plot(self):
        t = np.array([t*self.m.tf() for t in self.m.t])
        z = np.array([self.m.z[t]() for t in self.m.t])
        y = np.array([self.m.y[t]() for t in self.m.t])
        plt.plot(z,y)
        plt.show()
        
if __name__ == '__main__':
    x0 = np.array([0.0, 0.0, 0.0])
    xf = np.array([5.0, 0.0, np.pi/2])
    tg = TrajectoryGenerator(x0, xf, 1)
    tg.solve()
    tg.plot()
