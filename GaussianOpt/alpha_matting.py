import subprocess, shlex, math, random
from fastnumbers import fast_real, fast_int
import numpy as np
from GPyOpt.methods import BayesianOptimization
import re
from bopt_utils import Runner

exe = "/home/erik/Projects/ParallelFusion/build/AlphaMatting/AlphaMatting"

data_base = "."
data_sets = "GT01 GT02 GT13 GT24 GT25".split(" ")


searcher = re.compile(".*Final error:\s*(\d*\.\d*).*")
runner = Runner(exe, searcher, data_base, "{} -img_name={}/{} -num_proposals={} -exchange_interval={} -exchange_amount={} -num_threads=")


def layer(exchange_amount, num_proposals, exchange_interval = 4):
  print num_proposals, exchange_interval, exchange_amount
  return runner.run(data_sets, fast_int(num_proposals), fast_int(exchange_interval), fast_int(exchange_amount))

def wrapper(evals):
  print evals
  res = np.array([layer(*e) for e in evals])
  print res
  return np.atleast_2d(res)

def main():
  num_threads = 1
  max_num_proposals = 5
  while num_threads <= 4:
    global runner
    runner = Runner(exe, searcher, data_base, "{} -img_name={}/{} -num_proposals={} -exchange_interval={} -exchange_amount={} -num_threads=" + str(num_threads))
    space = [
      {
        "name" : "exchange_amount",
        "type" : "discrete",
        "domain": tuple(range(0, num_threads + 1)),
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
      },
      {
        "name": "cont_3",
        "constrain": "-1*((x == 1).all(1))"
      }
    ]


    # --- Solve your problem
    opt = BayesianOptimization(f=wrapper,
                                domain=space,
                                constrains=constrains)

    opt.run_optimization(max_iter=20,
                          evaluations_file="alpha-matting-evals-{}.txt".format(num_threads),
                          models_file="alpha-matting-model-{}.txt".format(num_threads),
                          batch_size=5,
                          evaluator_type="local_penalization")


    opt.plot_acquisition("alpha-matting")
    num_threads *= 2


if __name__ == '__main__':
  main()