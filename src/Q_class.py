import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.integrate import ode
from scipy.optimize import minimize
import sympy as sy
from scipy.integrate import odeint
from scipy.integrate import quad
from numpy import loadtxt
from scipy.integrate import cumulative_trapezoid
from IPython.display import display
from pylab import *

class Q_balls():
    def __init__(self, w0, M, λ, r):
        self.α0, self.x0, self.g0, self.yy0 = w0 
        self.M = M
        self.λ = λ
        self.r0, self.rf = r
    def Num_sol(self):
        def vectorfield(r, funs):
            α, x, g, yy = funs
            f = [x,
                -x*(2*r/(r**2+1e-12))+(1/(1-λ*(g**2)*np.sin(α)**2))*(2*x*yy*λ*g*np.sin(α)**2-(1/2)*(g**2)*(np.sin(2*α))*(1-λ*x**2)+(4*(M**2))*np.sin(α)),
                yy,
                -yy*(2*r/(r**2+1e-12)) + (0.13468/λ)*g*np.sin(α)**2*(1+λ*x**2)]       
            return f
        self.solution = solve_ivp(vectorfield, (self.r0, self.rf), y0=w0, method='BDF')
        self.α, self.x, self.g, self.yy = self.solution.y
        self.rn = self.solution.t
    def ED(self):
        αn, dαn, gn, dgn = self.α, self.x, self.g, self.yy
        ED = (0.4231/λ)*((gn**2)*(np.sin(αn)**2)*(1+λ*dαn**2)+dαn**2 - 8*(M**2)*(np.cos(αn)-1)+7.425*λ*(dgn)**2  )
        return ED  
    def E(self):
        αn, dαn, gn, dgn, rn = self.α, self.x, self.g, self.yy, self.rn
        Eg = (rn**2)*((0.4231/λ)*((gn**2)*(np.sin(αn)**2)*(1+λ*dαn**2)+dαn**2 - 8*(M**2)*(np.cos(αn)-1)+7.425*λ*(dgn)**2  ))
        Int=cumulative_trapezoid(Eg, rn, initial=0)
        return Int
    def QD(self):       
        αn, dαn, gn, dgn = self.α, self.x, self.g, self.yy
        qg = 2*(0.4231/λ)*gn*(np.sin(αn)**2)*(1+λ*dαn**2)
        return qg
    def Q(self):
        αn, dαn, gn, dgn, rn = self.α, self.x, self.g, self.yy, self.rn
        Q =  rn**2*(2*(0.4231/λ)*gn*(np.sin(αn)**2)*(1+λ*dαn**2))
        QInt=cumulative_trapezoid(Q,rn, initial=0)
        return QInt
