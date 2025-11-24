import numpy as np
from crossMat import crossMat

def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the Hessian (second derivative) of bending energy E_k^b
    with respect to x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0.0])
    node1 = np.array([xk,    yk,    0.0])
    node2 = np.array([xkp1,  ykp1,  0.0])

    # Unit directors along z
    m2e = np.array([0.0, 0.0, 1.0])
    m2f = np.array([0.0, 0.0, 1.0])

    kappaBar = curvature0

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    # Degenerate geometry -> no well-defined curvature/Hessian
    eps = 1e-12
    if norm_e < eps or norm_f < eps or l_k < eps:
        return np.zeros((6, 6))

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    # Clamp dot product to avoid exactly -1 or +1
    dot_te_tf = np.dot(te, tf)
    dot_clamped = np.clip(dot_te_tf, -1.0 + 1e-6, 1.0 - 1e-6)

    # Curvature binormal using clamped dot
    denom = 1.0 + dot_clamped
    if abs(denom) < 1e-8:
        # Nearly 180° bend -> numerically singular
        return np.zeros((6, 6))

    kb = 2.0 * np.cross(te, tf) / denom

    # Use same clamped dot for chi
    chi = 1.0 + dot_clamped
    if abs(chi) < 1e-8:
        return np.zeros((6, 6))

    tilde_t  = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature (z-component)
    kappa1 = kb[2]

    # Gradient of kappa1 wrt edge vectors
    Dkappa1De = ( -kappa1 * tilde_t + np.cross(tf, tilde_d2) ) / norm_e
    Dkappa1Df = ( -kappa1 * tilde_t - np.cross(te, tilde_d2) ) / norm_f

    # Assemble gradKappa (6-vector)
    gradKappa = np.zeros(6)
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] =  Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] =  Dkappa1Df[0:2]

    # Now the second derivatives of kappa
    DDkappa1 = np.zeros((6, 6))

    norm2_e = norm_e**2
    norm2_f = norm_f**2
    Id3     = np.eye(3)

    tt_o_tt = np.outer(tilde_t, tilde_t)

    tmp = np.cross(tf, tilde_d2)
    tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
    kb_o_d2e      = np.outer(kb, m2e)

    D2kappa1De2 = (
        (2.0 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tf_c_d2t_o_tt.T) / norm2_e
        - kappa1 / (chi * norm2_e) * (Id3 - np.outer(te, te))
        + (kb_o_d2e + kb_o_d2e.T) / (4.0 * norm2_e)
    )

    tmp = np.cross(te, tilde_d2)
    te_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d2t = te_c_d2t_o_tt.T
    kb_o_d2f      = np.outer(kb, m2f)

    D2kappa1Df2 = (
        (2.0 * kappa1 * tt_o_tt + te_c_d2t_o_tt + te_c_d2t_o_tt.T) / norm2_f
        - kappa1 / (chi * norm2_f) * (Id3 - np.outer(tf, tf))
        + (kb_o_d2f + kb_o_d2f.T) / (4.0 * norm2_f)
    )

    D2kappa1DeDf = (
        -kappa1 / (chi * norm_e * norm_f) * (Id3 + np.outer(te, tf))
        + 1.0 / (norm_e * norm_f)
          * (2.0 * kappa1 * tt_o_tt
             - tf_c_d2t_o_tt
             + tt_o_te_c_d2t
             - crossMat(tilde_d2))
    )
    D2kappa1DfDe = D2kappa1DeDf.T

    # Fill DDkappa1 (only 2D blocks needed)
    DDkappa1[0:2, 0:2] = D2kappa1De2[0:2, 0:2]
    DDkappa1[0:2, 2:4] = -D2kappa1De2[0:2, 0:2] + D2kappa1DeDf[0:2, 0:2]
    DDkappa1[0:2, 4:6] = -D2kappa1DeDf[0:2, 0:2]

    DDkappa1[2:4, 0:2] = -D2kappa1De2[0:2, 0:2] + D2kappa1DfDe[0:2, 0:2]
    DDkappa1[2:4, 2:4] = (
        D2kappa1De2[0:2, 0:2]
        - D2kappa1DeDf[0:2, 0:2]
        - D2kappa1DfDe[0:2, 0:2]
        + D2kappa1Df2[0:2, 0:2]
    )
    DDkappa1[2:4, 4:6] = D2kappa1DeDf[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]

    DDkappa1[4:6, 0:2] = -D2kappa1DfDe[0:2, 0:2]
    DDkappa1[4:6, 2:4] = D2kappa1DfDe[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 4:6] = D2kappa1Df2[0:2, 0:2]

    # Hessian of bending energy
    dkappa = kappa1 - kappaBar

    dJ = EI / l_k * np.outer(gradKappa, gradKappa)
    dJ += EI / l_k * dkappa * DDkappa1

    return dJ
