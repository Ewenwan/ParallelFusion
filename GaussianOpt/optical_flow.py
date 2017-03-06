import subprocess, shlex
from fastnumbers import fast_real, fast_int
import numpy as np
import math
from GPyOpt.methods import BayesianOptimization
import re
from bopt_utils import Runner

exe = "/home/vision/erikwijmans/ParallelFusion/build/OpticalFlow/OpticalFlow"

data_base = "/home/vision/erikwijmans/ParallelFusion/OpticalFlow/other-data"
data_sets = "Beanbags DogDance Grove3 MiniCooper Urban2 Venus Dimetrodon Grove2 Hydrangea RubberWhale Urban3 Walking".strip().split(" ")
searcher = re.compile(".*Final energy:\s*(\d*\.\d*).*")
runner = Runner(exe, searcher, data_base, "{} -scene_name={}/{} -num_proposals_in_total={} -solution_exchange_interval={} -num_proposals_from_others={} -num_threads=")
num_threads = 4

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

def run_command(command):
  num_tries = 2
  while num_tries != 0:
    try:
      process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

      data = process.communicate()
      print data[0]

      res = np.array([math.log(fast_real(data[0].strip().split("\n")[-1].split(",")[0].split(":")[1]))])

      return res
    except:
      num_tries -= 1

  raise RuntimeError("Failure to run: \"{}\"".format(command))

def optical_flow(exchange_amount, num_proposals, exchange_interval = 2):
  print num_proposals, exchange_interval, exchange_amount
  res = np.array([0.0])

  num_successful = len(data_sets)
  for data in data_sets:
    command = "{} -dataset_name={}/{} -num_proposals_in_total={} -solution_exchange_interval={} -num_proposals_from_others={} -num_threads={}".format(exe, data_base, data, fast_int(num_proposals), fast_int(exchange_interval), fast_int(exchange_amount), num_threads)
    print command

    try:
      res += run_command(command)
    except RuntimeError, e:
      print "{}{}{}".format(bcolors.WARNING, e, bcolors.ENDC)
      num_successful -= 1

  res /= num_successful
  print res
  return res


def wrapper(evals):
  print evals
  res = np.array([optical_flow(*e) for e in evals])
  print res
  return res

def main():
  max_num_proposals = 11
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
    }
  ]


  # --- Solve your problem
  opt = BayesianOptimization(f=wrapper,
                                domain=space,
                                constrains=constrains)

  opt.run_optimization(max_iter=50,
                          evaluations_file="Dimetrodon-flow-evals.txt",
                          models_file="Dimetrodon-flow-model.txt",
                          batch_size=5,
                          evaluator_type="local_penalization")

  opt.plot_acquisition("Dimetrodon_plot")


if __name__ == '__main__':
  main()
