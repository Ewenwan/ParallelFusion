import subprocess, shlex, math, random
from fastnumbers import fast_real, fast_int
import numpy as np
from GPyOpt.methods import BayesianOptimization
import re
from bopt_utils import Runner, get_contrains, get_space, build_wrapper, utils

exe = "/home/erik/Projects/ParallelFusion/build/stereo/code/simplestereo/SimpleStereo"

data_base = "/home/erik/Projects/ParallelFusion/stereo/stereo_data"
data_sets = "data_teddy2 data_teddy data_cones data_book data_book2".split(" ")

searcher = re.compile(".*Final energy:\s*(\d*\.\d*).*")
runner = Runner(exe, searcher, data_base, "{} {}/{} -num_proposals={} -exchange_interval={} -exchange_amount={} -num_threads=4")

def simple_stereo(num_proposals, exchange_amount, exchange_interval = 2):
  print num_proposals, exchange_interval, exchange_amount
  return runner.run(data_sets, fast_int(num_proposals), fast_int(exchange_interval), fast_int(exchange_amount))

def main():
  num_threads = 1
  max_num_proposals = 11
  while num_threads <= 8:
    global runner
    runner = Runner(exe, searcher, data_base, "{} {}/{} -num_proposals={} -exchange_interval={} -exchange_amount={} -num_threads=" + str(num_threads))

    # --- Solve your problem
    opt = BayesianOptimization(f=build_wrapper(simple_stereo),
                                  domain=get_space(num_threads, max_num_proposals),
                                  constrains=get_contrains())

    opt.run_optimization(max_iter=utils.MAX_ITERS,
                            evaluations_file="simple-stereo-evals-{}.txt".format(num_threads),
                            models_file="simple-stereo-model-{}.txt".format(num_threads),
                            batch_size=utils.BATCH_SIZE,
                            evaluator_type="local_penalization")


    opt.plot_acquisition("stereo_plot")

    num_threads *= 2


if __name__ == '__main__':
  main()