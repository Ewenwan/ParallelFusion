# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np

# --- Define your problem
def f(x): return (6*x-2)**2*np.sin(12*x-4)
bounds = [(0,1)]

# --- Solve your problem
myBopt = BayesianOptimization(f=f, bounds=bounds)
myBopt.run_optimization(max_iter=15)
myBopt.plot_acquisition()