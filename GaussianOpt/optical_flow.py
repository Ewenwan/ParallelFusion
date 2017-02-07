import subprocess, shlex
from fastnumbers import fast_real, fast_int
import numpy as np
import math

exe = "/home/vision/erikwijmans/ParallelFusion/build/OpticalFlow/OpticalFlow"

data_base = "/home/vision/erikwijmans/ParallelFusion/OpticalFlow/other-data"
data_sets = "Beanbags DogDance Grove3 MiniCooper Urban2 Venus Dimetrondon Grove2 Hydrangea RubberWhale Urban3 Walking".strip().split(" ")[:3]

def run_command(command):
  num_tries = 2
  while num_tries != 0:
    try:
      process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

      data = process.communicate()
      print data[0]

      res = np.array([math.log(fast_real(data[0].strip().split("\n")[-1].split(",")[0].split(":")[1]))])
      print res
      return res
    except:
      num_tries -= 1

  raise RuntimeError

def optical_flow(exchange_amount, num_proposals, exchange_interval):
  print num_proposals, exchange_interval, exchange_amount
  res = np.array([0.0])

  num_successful = len(data_sets)
  for data in data_sets:
    command = "{} -dataset_name={}/{} -num_proposals_in_total={} -solution_exchange_interval={} -num_proposals_from_others={}".format(exe, data_base, data, fast_int(round(num_proposals[0])), fast_int(round(exchange_interval[0])), fast_int(round(exchange_amount[0])))
    print command

    try:
      res += run_command(command)
    except:
      num_successful -= 1

  res /= num_successful
  print res
  return res


if __name__ == '__main__':
  print optical_flow([0], [1], [1])