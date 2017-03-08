import subprocess, shlex, math, random
from fastnumbers import fast_real, fast_int
import numpy as np
from GPyOpt.methods import BayesianOptimization
import re
from bopt_utils import Runner

exe = "/home/erik/Projects/ParallelFusion/build/LayerDepthMap/LayerDepthMap"

data_base = "/home/erik/Projects/ParallelFusion/LayerDepthMap/Inputs"
data_sets = "cse013".split(" ")


searcher = re.compile(".*Final energy:\s*(\d*\.\d*).*")
runner = Runner(exe, searcher, data_base, "{} -scene_name={}/{} -num_proposals_in_total={} -solution_exchange_interval={} -num_proposals_from_others={} -num_threads=")


def layer(exchange_amount, num_proposals, exchange_interval = 2):
  print num_proposals, exchange_interval, exchange_amount
  return runner.run(data_sets, fast_int(num_proposals), fast_int(exchange_interval), fast_int(exchange_amount))

def wrapper(evals):
  print evals
  res = np.array([layer(*e) for e in evals])
  print res
  return np.atleast_2d(res)

def main():
  max_num_proposals = 11
  num_threads = 1
  while num_threads <= 8:
    global runner
    runner = Runner(exe, searcher, data_base, "{} -scene_name={}/{} -num_proposals_in_total={} -solution_exchange_interval={} -num_proposals_from_others={} -num_threads=" + str(num_threads))
    space = [
      {
        "name" : "exchange_amount",
        "type" : "discrete",
        "domain": tuple(range(0, num_threads)),
        "dimensionality": 1
      },
      {
        "name" : "num_proposals",
        "type" : "discrete",
        "domain": tuple(range(1, max_num_proposals + 1)),
        "dimensionality": 1
      },
      {
        "name" : "exchange_interval",
        "type" : "discrete",
        "domain": tuple(range(1, max_num_proposals + 1)),
        "dimensionality": 1
      }
    ]

    constrains = [
      {
        "name": "cont_1",
        "constrain": "x[:, 0] - x[:, 1] - 0.5"
      },
      {
        "name": "cont_2",
        "constrain": "-1*(np.logical_or(x[:, 0] == x[:, 1], x[:, 2] == 1))"
      }
    ]


    # --- Solve your problem
    opt = BayesianOptimization(f=wrapper,
                                  domain=space,
                                  constrains=constrains)

    opt.run_optimization(max_iter=25,
                            evaluations_file="layer-depth-evals-{}.txt".format(num_threads),
                            models_file="layer-depth-model-{}.txt".format(num_threads),
                            batch_size=5,
                            evaluator_type="local_penalization")


    opt.plot_acquisition("layer-depth")
    num_threads *= 2


if __name__ == '__main__':
  main()