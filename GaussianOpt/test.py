import subprocess, shlex, math, random
from fastnumbers import fast_real, fast_int
import numpy as np
from GPyOpt.methods import BayesianOptimization



last_val = 0
def wrapper(evals):
  print evals
  global last_val
  last_val -= 1
  return np.atleast_2d([last_val])

def main():
  max_num_proposals = 20
  space = [
    {
      "name" : "exchange_amount",
      "type" : "discrete",
      "domain": tuple(range(0, max_num_proposals + 1)),
      "dimensionality": 1
    },
    {
      "name" : "num_proposals",
      "type" : "discrete",
      "domain": tuple(range(1, max_num_proposals + 1)),
      "dimensionality": 1
    }
  ]

  constrains = [
    {
      "name": "cont_1",
      "constrain": "x[:, 0] - x[:, 1] + 0.5"
    }
  ]


  # --- Solve your problem
  myBopt = BayesianOptimization(f=wrapper, domain=space, constrains=constrains)
  myBopt.run_optimization(max_iter=200, evaluations_file="simple-stereo-evals.txt", models_file="simple-stereo-model.txt", batch_size=5, evaluator_type="local_penalization")


  myBopt.plot_acquisition("stereo_plot")


if __name__ == '__main__':
  main()