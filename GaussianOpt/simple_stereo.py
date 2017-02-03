import subprocess, shlex
from fastnumbers import fast_real, fast_int
import numpy as np
import math

exe = "/home/erik/Projects/ParallelFusion/build/stereo/code/simplestereo/SimpleStereo"

data_base = "/home/erik/Projects/ParallelFusion/build/stereo/code/stereo_data"
data_sets = ["data_teddy2", "data_teddy", "data_book", "data_book2", "data_cones"]

def run_command(command):
  num_tries = 5
  while num_tries != 0:
    try:
      process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

      data = process.communicate()
      print data[0]

      res = np.array([math.log(fast_real(data[0].strip().split("\n")[-2].split(",")[0].split(":")[1]))])
      return res
    except:
      num_tries -= 1

  raise RuntimeError

def simple_stereo(exchange_amount, num_proposals, exchange_interval):
  print num_proposals, exchange_interval, exchange_amount
  res = np.array([0.0])

  num_successful = len(data_sets)
  for data in data_sets:
    command = "{} {}/{} -num_proposals={} -exchange_interval={} -exchange_amount={} -num_threads=8".format(exe, data_base, data, fast_int(round(num_proposals[0])), fast_int(round(exchange_interval[0])), fast_int(round(exchange_amount[0])))
    print command

    try:
      res += run_command(command)
    except:
      num_successful -= 1

  res /= num_successful
  print res
  return res


if __name__ == '__main__':
  print simple_stereo([0], [1], [1])