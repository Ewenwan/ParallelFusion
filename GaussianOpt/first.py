# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np
import math

# --- Define your problem
def wrapper(d):
  return camel(d[0][0], d[0][1])

def camel(x, y):
  x2 = math.pow(x,2)
  x4 = math.pow(x,4)
  y2 = math.pow(y,2)
  res = (4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + x*y + (-4.0 + 4.0 * y2) * y2

  return res

space = [
  { "domain":(-1,1), "dimensionality":1, "name": "x" },
  { "domain":(-1,1), "dimensionality":1, "name": "y" }
]


# --- Solve your problem
myBopt = BayesianOptimization(f=wrapper, domain=space)
myBopt.run_optimization(max_iter=100)
myBopt.plot_acquisition("camel_plot")