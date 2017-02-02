from numpy import isclose
from fastnumbers import fast_real, isreal, fast_int
from sys import argv

class Result:
  def __init__(self, id, score, exchange_amount, num_proposals, exchange_interval):
    self.id, self.score, self.exchange_amount, self.num_proposals, self.exchange_interval = id, score, fast_int(round(exchange_amount)), num_proposals, exchange_interval


  def __str__(self):
    return "id: {:<4}  score: {}  # to Share: {:<4}  # of Proposals: {:<4}  Exchange Interval: {:<4}".format(self.id, self.score, self.exchange_amount, self.num_proposals, self.exchange_interval)

  def __hash__(self):
    return hash(self.exchange_amount) + hash(self.num_proposals) + hash(self.exchange_interval)

  def __eq__(self, other):
    return self.exchange_amount == other.exchange_amount and self.num_proposals == other.num_proposals and self.exchange_interval == other.exchange_interval

def parse_gpyopt(csv_name, reverse=False, key_idx=1):
  with open(csv_name) as f:
    plain_text = f.read()

  data = [d.split("\t") for d in plain_text.split("\n") if len(d) != 0]

  converted = []
  for d in data:
    if all(isreal(e) for e in d):
      converted.append([fast_real(e) for e in d])

  converted = sorted(converted, key=lambda x: x[key_idx], reverse=reverse)

  top = [a for a in converted if isclose(a[key_idx], converted[0][key_idx])]


  return set(Result(*t) for t in top)



if __name__ == '__main__':
  print "\n".join([str(e) for e in parse_gpyopt(argv[1])])