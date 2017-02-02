import numpy as np

def test(x, y):
  print x, y

  x_int = int(x[0])
  y_int = int(y[0])

  res = np.array([x_int*y_int + 1.5])
  print res
  return res