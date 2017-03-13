import subprocess, shlex, math, random
from fastnumbers import fast_real, fast_int
import numpy as np
from GPyOpt.methods import BayesianOptimization
import re
from bopt_utils import Runner, get_space, get_contrains, build_wrapper, utils

exe = "/home/erik/Projects/ParallelFusion/build/LayerDepthMap/LayerDepthMap"

data_base = "/home/erik/Projects/ParallelFusion/LayerDepthMap/Inputs"
data_sets = "cse013"


searcher = re.compile(".*Final energy:\s*(\d+\.\d+(e\d+)?).*")
runner = Runner(exe, searcher, data_base, "{} -scene_name={}/{} -num_proposals_in_total={} -solution_exchange_interval={} -num_proposals_from_others={} -num_threads=")


def layer(num_proposals, exchange_amount, exchange_interval = 2):
  print num_proposals, exchange_interval, exchange_amount
  return runner.run(data_sets, fast_int(num_proposals), fast_int(exchange_interval), fast_int(exchange_amount))

def main():
  max_num_proposals = 11
  num_threads = 4
  while num_threads <= 4:
    global runner
    runner = Runner(exe, searcher, data_base, "{} -scene_name={}/{} -num_proposals_in_total={} -solution_exchange_interval={} -num_proposals_from_others={} -num_threads=" + str(num_threads))

    # --- Solve your problem
    opt = BayesianOptimization(f=build_wrapper(layer),
                                  domain=get_space(num_threads, max_num_proposals),
                                  constrains=get_contrains())

    opt.run_optimization(max_iter=utils.MAX_ITERS,
                            evaluations_file="layer-depth-evals-{}.txt".format(num_threads),
                            models_file="layer-depth-model-{}.txt".format(num_threads),
                            batch_size=utils.BATCH_SIZE,
                            evaluator_type="local_penalization")


    opt.plot_acquisition("layer-depth")
    num_threads *= 2


if __name__ == '__main__':
  main()