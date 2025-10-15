import numpy as np
from getForceJacobian import getForceJacobian
def myInt(t_new, x_old, u_old, free_DOF, stiffness_matrix, index_matrix, m, dt, l_k):
  # t_new is optional for debugging
  # It should calculate x_new and u_new knowing old positions and velocities
  # free_DOF is a vector containing the indices of free variables (not boundary)

  # Guess
  x_new = x_old.copy()

  eps = 1.0e-6 # not a good practice -- tolerance
  err = 10 * eps
  # Newton - Raphson
  while err > eps:
    f, J = getForceJacobian(x_new, x_old, u_old, stiffness_matrix, index_matrix, m, dt, l_k)

    # Extract free DOFs
    f_free = f[free_DOF]
    J_free = J[np.ix_(free_DOF, free_DOF)]

    # Solve for the correction (Delta x)
    deltaX_free = np.linalg.solve(J_free, f_free) # Most time consuming step and can be optimized using the sparsity/banded nature of the jacobian
    # PARDISO Project

    # Full deltaX
    deltaX = np.zeros_like(x_new)
    deltaX[free_DOF] = deltaX_free # Only update the free part

    # Update x_new
    x_new = x_new - deltaX

    # Calculate error
    err = np.linalg.norm(f_free)

  u_new = (x_new - x_old) / dt

  return x_new, u_new