from numpy import isclose
from fastnumbers import fast_real, isreal, fast_int
from sys import argv

class Result:
  def __init__(self, id, score, exchange_amount, num_proposals, exchange_interval = 1):
    self.id, self.score, self.gamma = id, score, fast_int(exchange_interval)

    self.beta = fast_int(num_proposals)
    if self.gamma != 1:
      self.alpha = fast_int(num_proposals)
    else:
      self.alpha = self.beta - fast_int(exchange_amount)

  def __str__(self):
    return "id: {:<4}  score: {:.4f}  alpha: {:<4}  beta: {:<4}  gamma: {:<4}".format(self.id, self.score, self.alpha, self.beta, self.gamma)

  def __hash__(self):
    return hash((self.alpha, self.beta, self.gamma))

  def __eq__(self, other):
    return self.alpha, self.beta, self.gamma == other.alpha, other.beta, other.gamma

  def __format__(self, code):
    return format(str(self), code)

  def __getitem__(self, code):
    return [self.id, self.score, self.alpha, self.beta, self.gamma][code]


def parse_gpyopt(csv_name, reverse=False, key_idx=1):
  with open(csv_name) as f:
    plain_text = f.read()

  data = [d.split("\t") for d in plain_text.split("\n") if len(d) != 0]

  converted = []
  for d in data:
    if all(isreal(e) for e in d):
      converted.append([fast_real(e) for e in d])

  converted = sorted(converted, key=lambda x: x[key_idx], reverse=reverse)

  return sorted(list(set(Result(*a) for a in converted if isclose(a[key_idx], converted[0][key_idx], rtol=1e-4))),  key=lambda x: x[key_idx], reverse=reverse)



if __name__ == '__main__':
  print "\n".join(["{}".format(e) for e in parse_gpyopt(argv[1])])