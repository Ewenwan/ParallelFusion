import subprocess, shlex
from fastnumbers import fast_real, fast_int
import numpy as np

exe = "/home/erik/Projects/ParallelFusion/build/stereo/code/simplestereo/SimpleStereo"

data_base = "/home/erik/Projects/ParallelFusion/build/stereo/code/stereo_data"
data_sets = ["data_teddy2", "data_teddy", "data_book", "data_book2", "data_cones"]

def simple_stereo(exchange_amount, num_proposals, exchange_interval):
  print num_proposals, exchange_interval, exchange_amount
  res = np.array([0.0])
  for data in data_sets:

    command = "{} {}/{} -num_proposals={} -exchange_interval={} -exchange_amount={} -num_threads=2".format(exe, data_base, data, fast_int(round(num_proposals[0])), fast_int(round(exchange_interval[0])), fast_int(round(exchange_amount[0])))
    print command
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

    data = process.communicate()
    print data[0]

    res += np.array([fast_real(data[0].strip().split("\n")[-2].split(",")[0].split(":")[1])])

  res /= float(len(data_sets))
  print res
  return res


if __name__ == '__main__':
  print simple_stereo([0], [1], [1])