import subprocess, shlex, math, random
from fastnumbers import fast_real, fast_int
import numpy as np
from GPyOpt.methods import BayesianOptimization
import re
from bopt_utils import Runner, get_contrains, get_space, build_wrapper, utils

exe = "/home/erik/Projects/ParallelFusion/build/AlphaMatting/AlphaMatting"

data_base = "."
data_sets = "GT01 GT02 GT13 GT24 GT25".split(" ")


searcher = re.compile(".*Final error:\s*(\d*\.\d*).*")
runner = Runner(exe, searcher, data_base, "{} -img_name={}/{} -num_proposals={} -exchange_interval={} -exchange_amount={} -num_threads=")


def matting(num_proposals, exchange_amount, exchange_interval = 4):
  print num_proposals, exchange_interval, exchange_amount
  return runner.run(data_sets, fast_int(num_proposals), fast_int(exchange_interval), fast_int(exchange_amount))

def main():
  num_threads = 1
  max_num_proposals = 5
  while num_threads <= 4:
    global runner
    runner = Runner(exe, searcher, data_base, "{} -img_name={}/{} -num_proposals={} -exchange_interval={} -exchange_amount={} -num_threads=" + str(num_threads))

    # --- Solve your problem
    opt = BayesianOptimization(f=build_wrapper(matting),
                                domain=get_space(num_threads, max_num_proposals),
                                constrains=get_contrains())

    opt.run_optimization(max_iter=20,
                          evaluations_file="alpha-matting-evals-{}.txt".format(num_threads),
                          models_file="alpha-matting-model-{}.txt".format(num_threads),
                          batch_size=utils.BATCH_SIZE,
                          evaluator_type="local_penalization")


    opt.plot_acquisition("alpha-matting")
    num_threads *= 2


if __name__ == '__main__':
  main()