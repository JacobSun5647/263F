import numpy as np
def getExternalForce(m):
  W = np.zeros_like(m)
  for i in range(len(m) // 2): # Go over every node. Size of m is 2 times node number
    W[2 * i] = 0.0 # x-coordinate
    W[2 * i + 1] = -9.81 * m[2 * i + 1] # y-coordinate
  return W