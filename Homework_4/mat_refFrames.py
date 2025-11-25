import numpy as np
from Helper import parallel_transport

def computeTangent(q):
  # q is the DOF vector of size 4*nv - 1 = 3 * nv + ne
  nv = (len(q) + 1) // 4
  ne = nv - 1
  tangent = np.zeros((ne, 3))  # every edge has a tangent
  for c in range(ne):
    node0 = q[4*c:4*c+3]
    node1 = q[4*c+4:4*c+7]
    edge = node1 - node0
    tangent[c, :] = edge / np.linalg.norm(edge)
  return tangent


def computeMaterialDirectors(a1, a2, theta):
  # a1 = matrix of size ne x 3 - First reference director
  # a2 = matrix of size ne x 3
  # theta = vector of size ne (extracted from the DOF vector; every fourth element of the DOF vector)
  ne = len(theta) # Number of edges
  m1 = np.zeros_like(a1) # First material director
  m2 = np.zeros_like(a2) # Second material director
  for c in range(ne): # Loop over every edge
    cs = np.cos(theta[c])
    sn = np.sin(theta[c])
    m1[c, :] = cs * a1[c, :] + sn * a2[c, :]
    m2[c, :] = - sn * a1[c,:] + cs * a2[c, :]
  return m1, m2


def computeSpaceParallel(u1_first, q):
  # u1_first = first reference frame vector (arbitrary but orthonormal adapted) on the first edge
  # q is the DOF vector of size 4*nv - 1
  nv = (len(q)+1) // 4
  ne = nv -1

  tangent = computeTangent(q) # Get the tangent of each edge

  u1 = np.zeros((ne, 3)) # First reference frame director
  u2 = np.zeros((ne, 3)) # Second reference frame director

  u1_first = u1_first / np.linalg.norm(u1_first) # Ensure it is unit

  # First edge
  u1[0,:] = u1_first
  t0 = tangent[0,:]
  u2[0,:] = np.cross(t0, u1_first)
  u2[0,:] = u2[0,:] / np.linalg.norm(u2[0,:]) # Ensure it is unit

  for c in np.arange(1, ne):
    t0 = tangent[c-1,:] # "From" tangent
    t1 = tangent[c,:] # "To" tangent
    u1[c,:] = parallel_transport(u1[c-1,:], t0, t1)
    u1[c, :] = u1[c,:] / np.linalg.norm(u1[c,:]) # Ensure it is unit
    u2[c,:] = np.cross(t1, u1[c,:])
    u2[c, :] = u2[c,:] / np.linalg.norm(u2[c,:]) # Ensure it is unit

  return u1, u2

def computeTimeParallel(a1_old, q0, q):
  # a1_old: First time parallel frame director in "old" configuration
  # q0: "old" shape of the rod or DOF vector
  # q: "new" shape (a1 on this new shape is unknown)

  nv = (len(q)+1) // 4
  ne = nv -1

  tangent0 = computeTangent(q0) # Get the tangents in old configuration
  tangent = computeTangent(q) # Get the tangents in new configuration

  a1 = np.zeros((ne, 3)) # First time parallel frame director
  a2 = np.zeros((ne, 3)) # Second time parallel frame director

  for c in np.arange(ne): # Loop over every edge
    t0 = tangent0[c,:] # old tangent on the c-th edge
    t1 = tangent[c,:] # new tangent on the c-th edge
    a1[c,:] = parallel_transport(a1_old[c,:], t0, t1)
    a1[c,:] = a1[c,:] - np.dot(a1[c,:], t1) * t1 # Ensure it is orthogonal to t1
    a1[c, :] = a1[c,:] / np.linalg.norm(a1[c,:]) # Ensure it is unit
    a2[c, :] = np.cross(t1, a1[c,:])
    a2[c,:] = a2[c,:] - np.dot(a2[c,:], t1) * t1 # Ensure it is orthogonal to t1
    a2[c, :] = a2[c,:] / np.linalg.norm(a2[c,:]) # Ensure it is unit

  return a1, a2