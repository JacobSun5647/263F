import numpy as np


def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the derivative of bending energy E_k^b with respect to
    x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0.0])
    node1 = np.array([xk,    yk,    0.0])
    node2 = np.array([xkp1,  ykp1,  0.0])

    # Unit directors (here just z)
    m2e = np.array([0.0, 0.0, 1.0])
    m2f = np.array([0.0, 0.0, 1.0])

    kappaBar = curvature0

    # Initialize gradient of curvature
    gradKappa = np.zeros(6)

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    # If any edge is degenerate, no well-defined curvature → return zero
    eps = 1e-12
    if norm_e < eps or norm_f < eps or l_k < eps:
        return np.zeros(6)

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    # Dot product (clamped) to avoid exactly -1 or +1
    dot_te_tf = np.dot(te, tf)
    dot_clamped = np.clip(dot_te_tf, -1.0 + 1e-6, 1.0 - 1e-6)

    # Curvature binormal using clamped dot
    denom = 1.0 + dot_clamped
    if abs(denom) < 1e-8:
        # Nearly 180° bend → numerically singular, treat curvature as zero
        return np.zeros(6)

    kb = 2.0 * np.cross(te, tf) / denom

    # Use the same clamped dot for chi
    chi = 1.0 + dot_clamped
    if abs(chi) < 1e-8:
        # Same degeneracy as above, just in another form
        return np.zeros(6)

    tilde_t  = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature (only z-component for planar rod)
    kappa1 = kb[2]

    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = ( -kappa1 * tilde_t + np.cross(tf, tilde_d2) ) / norm_e
    Dkappa1Df = ( -kappa1 * tilde_t - np.cross(te, tilde_d2) ) / norm_f

    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] =  Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] =  Dkappa1Df[0:2]

    # Gradient of bending energy
    dkappa = kappa1 - kappaBar
    dF = gradKappa * EI * dkappa / l_k

    return dF
