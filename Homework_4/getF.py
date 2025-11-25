import numpy as np
from grad_hessEnergies import gradEs_hessEs
from grad_hessEnergies import gradEb_hessEb
from grad_hessEnergies import gradEt_hessEt

def getFs(q, EA, refLen):
  # STRETCHING
  # Input q is a DOF vector of size 4*nv - 1
  # Output is the stretching force vector (size 4*nv-1) and it's gradient w.r.t. q (square matrix)
  ndof = len(q)
  nv = (ndof + 1) // 4
  ne = nv - 1

  Fs = np.zeros(ndof)
  Js = np.zeros((ndof, ndof))

  # Loop over each stretching spring
  for c in range(ne):
    xa = q[4 * c : 4 * c + 3]
    xb = q[4 * c + 4 : 4 * c + 7]
    ind = np.array([4*c, 4*c+1, 4*c+2, 4*c+4, 4*c+5, 4*c+6])
    dL = refLen[c]

    dF, dJ = gradEs_hessEs(xa, xb, dL, EA)

    Fs[ind] -= dF
    Js[np.ix_(ind, ind)] -= dJ

  return Fs, Js

def getFb(q, m1, m2, kappaBar, EI, voronoiRefLen):
  # BENDING
  # Input q is a DOF vector of size 4*nv - 1
  ndof = len(q)
  nv = (ndof + 1) // 4
  ne = nv - 1

  Fb = np.zeros(ndof)
  Jb = np.zeros((ndof, ndof))

  # Loop over each bending spring
  for c in range(1, ne): # Ignore the terminal nodes (0 and nv)
    node0 = q[4 * (c - 1) : 4 * (c - 1) + 3] # (c-1)-th node
    node1 = q[4 * c : 4*c + 3] # c-th node
    node2 = q[4 * (c + 1) : 4 * (c + 1) + 3] # (c+1)-th node
    # Calculate bending force due to the turning at the c-th node

    m1e = m1[c - 1, 0:3]
    m2e = m2[c - 1, 0:3]
    m1f = m1[c,  0:3]
    m2f = m2[c,  0:3]

    dL = voronoiRefLen[c]
    curvature0 = kappaBar[c, 0:2]

    dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, curvature0, dL, EI)
    ind = np.array([4*c-4, 4*c-3, 4*c-2, 4*c-1,4*c, 4*c+1, 4*c+2, 4*c+3, 4*c+4, 4*c+5, 4*c+6])

    Fb[ind] -= dF
    Jb[np.ix_(ind, ind)] -= dJ

  return Fb, Jb

def getFt(q, refTwist, twistBar, GJ, voronoiRefLen):
  # TWISTING
  # Input q is a DOF vector of size 4*nv - 1
  ndof = len(q)
  nv = (ndof + 1) // 4
  ne = nv - 1

  Ft = np.zeros(ndof)
  Jt = np.zeros((ndof, ndof))

  # Loop over each twisting spring
  for c in range(1, ne): # Ignore the terminal nodes (0 and nv)
    node0 = q[4 * (c - 1) : 4 * (c - 1) + 3] # (c-1)-th node
    node1 = q[4 * c : 4*c + 3] # c-th node
    node2 = q[4 * (c + 1) : 4 * (c + 1) + 3] # (c+1)-th node
    theta_e = q[4*c-1]
    theta_f = q[4*c+3]
    # Calculate twisting force due to the turning at the c-th node

    dL = voronoiRefLen[c]

    dF, dJ = gradEt_hessEt(node0, node1, node2, theta_e, theta_f, refTwist[c], twistBar[c], dL, GJ)
    ind = np.array([4*c-4, 4*c-3, 4*c-2, 4*c-1,4*c, 4*c+1, 4*c+2, 4*c+3, 4*c+4, 4*c+5, 4*c+6])

    Ft[ind] -= dF
    Jt[np.ix_(ind, ind)] -= dJ

  return Ft, Jt