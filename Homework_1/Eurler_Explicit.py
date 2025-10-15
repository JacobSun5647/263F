import numpy as np

def myInt_explicit(t_new, x_old, u_old, free_DOF,
                   stiffness_matrix, index_matrix, m, dt, l_k,
                   fext_fn=None):
    """
    Forward (explicit) Euler using ONLY old-state info.
    Adds optional external force fext_fn(t, x, u) -> (ndof,).
    """
    ndof = x_old.size
    f = np.zeros(ndof)

    # --- internal spring forces at old state ---
    for s in range(stiffness_matrix.shape[0]):
        ind = index_matrix[s].astype(int)
        i_x, i_y, j_x, j_y = ind

        xi, yi = x_old[i_x], x_old[i_y]
        xj, yj = x_old[j_x], x_old[j_y]

        dx = xj - xi
        dy = yj - yi
        L  = np.hypot(dx, dy)
        if L == 0.0:
            continue

        k_s = stiffness_matrix[s]
        L0  = l_k[s]
        fac = k_s * (L - L0) / L   # = k * ΔL / L

        # equal & opposite projections
        fxi =  fac * (xi - xj); fyi =  fac * (yi - yj)
        fxj =  fac * (xj - xi); fyj =  fac * (yj - yi)

        f[i_x] += fxi; f[i_y] += fyi
        f[j_x] += fxj; f[j_y] += fyj

    # --- external forces (e.g., gravity) at t_k ---
    if fext_fn is not None:
        f += fext_fn(t_new - dt, x_old, u_old)  # use old time t_k

    # --- a = M^{-1} f (diagonal mass) ---
    Minv = (1.0 / m) if np.ndim(m) else (1.0 / float(m))
    a = Minv * f

    # --- explicit Euler updates (only FREE dofs) ---
    x_new = x_old.copy()
    u_new = u_old.copy()

    x_new[free_DOF] = x_old[free_DOF] + dt * u_old[free_DOF]
    u_new[free_DOF] = u_old[free_DOF] + dt * a[free_DOF]

    # clamp fixed DOFs
    fixed = np.setdiff1d(np.arange(ndof), free_DOF, assume_unique=True)
    x_new[fixed] = x_old[fixed]
    u_new[fixed] = 0.0

    return x_new, u_new
