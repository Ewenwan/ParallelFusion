import subprocess, shlex, math
from fastnumbers import fast_real, fast_int
import numpy as np
import re

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


class Runner:
  def __init__(self, exe, searcher, data_base, frmt_str):
    self.searcher, self.frmt_str, self.exe, self.data_base = searcher, frmt_str, exe, data_base

  def run(self, data_sets, *frmt_args):
    res = np.array([0.0])
    if isinstance(data_sets, list):
      num_successful = len(data_sets)
      for data in data_sets:
        command = self.frmt_str.format(self.exe, self.data_base, data, *frmt_args)
        print(command)
        try:
          res += self._run_command(command)
        except RuntimeError, e:
          print "{}{}{}".format(bcolors.WARNING, e, bcolors.ENDC)
          num_successful -= 1

      res /= num_successful
    elif isinstance(data_sets, str):
      command = self.frmt_str.format(self.exe, self.data_base, data_sets, *frmt_args)
      print(command)
      res += self._run_command(command)

    return res

  def _run_command(self, command):
    num_tries = 1
    while num_tries != 0:
      try:
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

        data = process.communicate()
        print data[0]

        res = self.searcher.search(data[0])
        return math.log(fast_real(res.group(1)))
      except:
        num_tries -= 1

    raise RuntimeError("Failure to run: \"{}\"".format(command))

constrains = [
  {
    "name": "cont_1",
    "constrain": "x[:, 0] - x[:, 1] + 0.5"
  },
  {
    "name": "cont_2",
    "constrain": "-1*(np.logical_or(x[:, 0] == x[:, 1], x[:, 2] == 1))"
  },
  {
    "name": "cont_3",
    "constrain": "-1*((x[:, :] != 1).any(1))"
  }
]