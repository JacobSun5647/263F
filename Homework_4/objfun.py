import numpy as np
from mat_refFrames import computeTimeParallel
from mat_refFrames import computeTangent
from mat_refFrames import computeMaterialDirectors
from refTwist_Curv import getRefTwist
from getF import getFt, getFb, getFs

def objfun(qOld, uOld, a1_old, a2_old,
           freeIndex,
           dt, tol,
           refTwist,
           massVector, massMatrix,
           EA, refLen,
           EI, GJ, voronoiRefLen,
           kappaBar, twistBar,
           Fg):

  q_new = qOld.copy()
  iter = 0
  error = 10 * tol

  while error > tol:
    # Reference frame
    a1_new, a2_new = computeTimeParallel(a1_old, qOld, q_new) # Time parallel reference frame along the rod
    # Reference twist
    tangent = computeTangent(q_new)
    refTwist_new = getRefTwist(a1_new, tangent, refTwist) # Reference twist vector of size nv
    # Material frame
    theta = q_new[3::4]
    m1, m2 = computeMaterialDirectors(a1_new, a2_new, theta) # Material directors of size nv x 3

    # Computer elastic forces
    Fs, Js = getFs(q_new, EA, refLen)
    Fb, Jb = getFb(q_new, m1, m2, kappaBar, EI, voronoiRefLen)
    Ft, Jt = getFt(q_new, refTwist_new, twistBar, GJ, voronoiRefLen)

    Forces = Fs + Fb + Ft + Fg
    JForces = Js + Jb + Jt

    f = massVector / dt * ( (q_new - qOld) / dt - uOld ) - Forces
    J = massMatrix / dt**2 - JForces

    # Extract the free part
    f_free = f[freeIndex]
    J_free = J[np.ix_(freeIndex, freeIndex)]

    dq_free = np.linalg.solve(J_free, f_free) # J \ f

    q_new[freeIndex] -= dq_free
    error = np.sum(np.abs(f_free)) # Correction
    # Keep in mind that "error = np.sum(np.abs(dq_free))" is ok but tol should be computed based on length
    iter += 1

  uNew = (q_new - qOld) / dt
  return q_new, uNew, a1_new, a2_new