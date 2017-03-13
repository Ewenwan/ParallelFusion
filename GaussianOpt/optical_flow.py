import subprocess, shlex
from fastnumbers import fast_real, fast_int
import numpy as np
import math
from GPyOpt.methods import BayesianOptimization
import re
from bopt_utils import Runner, get_contrains, get_space, build_wrapper, utils


exe = "/home/vision/erikwijmans/ParallelFusion/build/OpticalFlow/OpticalFlow"

data_base = "/home/vision/erikwijmans/ParallelFusion/OpticalFlow/other-data"
data_sets = "Beanbags DogDance Grove3 MiniCooper Urban2 Venus Dimetrodon Grove2 Hydrangea RubberWhale Urban3 Walking".strip().split(" ")[5:8]

searcher = re.compile(".*Final energy:\s*(\d*\.\d*).*")
runner = Runner(exe, searcher, data_base, "{} -scene_name={}/{} -num_proposals_in_total={} -solution_exchange_interval={} -num_proposals_from_others={} -num_threads=")

def flow(num_proposals, exchange_amount, exchange_interval = 2):
  print num_proposals, exchange_interval, exchange_amount
  return runner.run(data_sets, fast_int(num_proposals), fast_int(exchange_interval), fast_int(exchange_amount))

def main():
  max_num_proposals = 11
  num_threads = 1
  while num_threads <= 8:
    global runner
    runner = Runner(exe, searcher, data_base, "{} -dataset_name={}/{} -num_proposals_in_total={} -solution_exchange_interval={} -num_proposals_from_others={} -num_threads=" + str(num_threads))

  # --- Solve your problem
    opt = BayesianOptimization(f=build_wrapper(flow),
                                  domain=get_space(num_threads, max_num_proposals),
                                  constrains=get_contrains())

    opt.run_optimization(max_iter=utils.MAX_ITERS,
                            evaluations_file="opt-flow-evals-{}.txt".format(num_threads),
                            models_file="opt-flow-model-{}.txt".format(num_threads),
                            batch_size=utils.BATCH_SIZE,
                            evaluator_type="local_penalization")


    opt.plot_acquisition("opt-flow")
    num_threads *= 2


if __name__ == '__main__':
  main()
