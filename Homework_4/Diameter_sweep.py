import numpy as np
import matplotlib.pyplot as plt

from mat_refFrames import computeTangent, computeSpaceParallel, computeMaterialDirectors
from refTwist_Curv import getKappa
from objfun import objfun
from plot import plotrod_simple

# ============================================================
# Global inputs (D will be varied later)
# ============================================================

# Discretization
nv = 50                  # number of nodes
ne = nv - 1              # number of edges
ndof = 3 * nv + ne       # 3 positional + 1 twist DOF per edge

# Rod cross-section
r0 = 0.001               # radius [m] (d = 0.002 m)
d_rod = 2.0 * r0         # diameter (constant in Part (3))

# Helix / spring parameters (D will be varied)
pitch = 2 * r0           # pitch [m]
N = 5                    # number of turns

# Material parameters
Y = 10e6                 # Young's modulus [Pa]
nu = 0.5                 # Poisson's ratio
G = Y / (2.0 * (1.0 + nu))  # shear modulus [Pa]

# Stiffnesses (depend only on r0, not on D)
EA = Y * np.pi * r0**2             # stretching stiffness
EI = Y * np.pi * r0**4 / 4.0       # bending stiffness
GJ = G * np.pi * r0**4 / 2.0       # torsional stiffness

# Density
rho = 1000.0           # [kg/m^3]

# Time integration
totalTime = 5.0
dt = 0.01
Nsteps = int(round(totalTime / dt))

# Fixed / free DOFs
fixedIndex = np.arange(0, 7)       # clamp first node (x, y, z, theta, next x, next y, next z)
freeIndex = np.arange(7, ndof)

# ------------------------------------------------------------
# Global variables that depend on geometry D
# (will be updated by setup_geometry_for_D)
# ------------------------------------------------------------
nodes = None
refLen = None
voronoiRefLen = None
massVector = None
massMatrix = None
a1 = None
a2 = None
refTwist = None
kappaBar = None
twistBar = None
L = None
axial_l = None
tol = None   # tolerance for Newton iterations inside objfun

# ============================================================
# Helpers to (1) build frames, (2) set up geometry for a given D
# ============================================================

def setup_reference_frames():
    """
    Build reference tangent frame, material frame, and natural curvature
    for the current geometry in 'nodes'.
    Updates global a1, a2, refTwist, kappaBar, twistBar.
    """
    global a1, a2, refTwist, kappaBar, twistBar

    # Build reference generalized coordinates q_ref from nodes
    q_ref = np.zeros(ndof)
    for c in range(nv):
        ind = [4 * c, 4 * c + 1, 4 * c + 2]
        q_ref[ind] = nodes[c, :]

    # Tangent and space-parallel frame
    tangent = computeTangent(q_ref)
    t0 = tangent[0, :]

    arb_v = np.array([0.0, 0.0, -1.0])
    if np.linalg.norm(np.cross(t0, arb_v)) < 1e-3:
        arb_v = np.array([0.0, 1.0, 0.0])

    a1_first = np.cross(t0, arb_v)
    a1_first /= np.linalg.norm(a1_first)

    a1, a2 = computeSpaceParallel(a1_first, q_ref)

    # Material frame (for now, same as space frame at reference)
    theta = q_ref[3::4]  # twist angles (zero at reference)
    m1, m2 = computeMaterialDirectors(a1, a2, theta)

    # Reference twist and natural curvature
    refTwist = np.zeros(nv)
    kappaBar = getKappa(q_ref, m1, m2)
    twistBar = np.zeros(nv)


def setup_geometry_for_D(D_value):
    """
    Set up all geometry- and mass-related quantities for a given helix diameter D_value.
    Updates global nodes, L, axial_l, refLen, voronoiRefLen, massVector, massMatrix, tol,
    and reference frames (a1, a2, refTwist, kappaBar, twistBar).
    """
    global nodes, refLen, voronoiRefLen, massVector, massMatrix
    global L, axial_l, tol

    # Helix parameters for this D
    a = D_value / 2.0                # helix radius
    b = pitch / (2.0 * np.pi)        # parameter in standard helix
    T = 2.0 * np.pi * N              # total angle (N turns)

    # Exact arc length (centerline)
    L = T * np.sqrt((a**2) + b**2)
    axial_l = N * pitch

    print("\n========================================")
    print(f"Setting up geometry for D = {D_value:.3f} m")
    print(f"Arc length L = {L:.6e} m, axial length = {axial_l:.6e} m")

    # Build node coordinates for centerline
    nodes = np.zeros((nv, 3))
    for c in range(nv):
        t = c * T / (nv - 1.0)
        nodes[c, 0] = a * np.cos(t)
        nodes[c, 1] = a * np.sin(t)
        nodes[c, 2] = -b * t

    # Reference edge lengths
    refLen = np.zeros(ne)
    for c in range(ne):
        refLen[c] = np.linalg.norm(nodes[c + 1, :] - nodes[c, :])

    # Voronoi lengths
    voronoiRefLen = np.zeros(nv)
    for c in range(nv):
        if c == 0:
            voronoiRefLen[c] = 0.5 * refLen[c]
        elif c == nv - 1:
            voronoiRefLen[c] = 0.5 * refLen[c - 1]
        else:
            voronoiRefLen[c] = 0.5 * (refLen[c - 1] + refLen[c])

    # Mass and inertia
    totalM = L * np.pi * r0**2 * rho
    dm = totalM / ne

    massVector = np.zeros(ndof)
    for c in range(nv):
        ind = [4 * c, 4 * c + 1, 4 * c + 2]
        if c == 0 or c == nv - 1:
            massVector[ind] = dm / 2.0
        else:
            massVector[ind] = dm

    # Angular mass per edge (solid cylinder approx; could be neglected if tiny)
    for c in range(ne):
        massVector[4 * c + 3] = 0.5 * dm * r0**2

    massMatrix = np.diag(massVector)

    # Tolerance (scaled with geometry)
    tol = EI / (L**2) * 1e-3

    # Build reference frames and natural curvature
    setup_reference_frames()


# ============================================================
# Dynamic stepper for a given external load F_ext
# ============================================================

def run_to_steady(F_ext, do_snapshots=False):
    """
    Run dynamic simulation for current geometry and given external force vector F_ext
    until totalTime, then estimate steady-state end displacement δz*.

    Returns
    -------
    steady_time : float or None
    steady_value : float
    endZ : (Nsteps,) array     # δz(t)
    local_snaps : list of (q, t)
    """
    # initial configuration qOld from current nodes
    qOld = np.zeros(ndof)
    for c in range(nv):
        ind = [4 * c, 4 * c + 1, 4 * c + 2]
        qOld[ind] = nodes[c, :]

    uOld = np.zeros_like(qOld)
    a1_old = a1.copy()
    a2_old = a2.copy()

    # reference z of end node
    endZ_0_local = qOld[-1]

    ctime = 0.0
    endZ = np.zeros(Nsteps)
    local_snaps = []

    for timeStep in range(Nsteps):
        q_new, u_new, a1_new, a2_new = objfun(
            qOld, uOld, a1_old, a2_old,
            freeIndex, dt, tol, refTwist,
            massVector, massMatrix,
            EA, refLen,
            EI, GJ, voronoiRefLen,
            kappaBar, twistBar,
            F_ext
        )

        # end-node Z displacement from reference
        endZ[timeStep] = q_new[-1] - endZ_0_local

        if do_snapshots and (timeStep % 500 == 0 or timeStep == Nsteps - 1):
            local_snaps.append((q_new.copy(), ctime))

        ctime += dt

        # advance
        qOld = q_new.copy()
        uOld = u_new.copy()
        a1_old = a1_new.copy()
        a2_old = a2_new.copy()

    # ---- Steady-state detection on δz(t) ----
    window_duration = 1.0
    window_steps = int(window_duration / dt)

    steady_time = None
    steady_value = endZ[-1]  # fallback

    for i in range(0, Nsteps - window_steps):
        z_start = endZ[i]
        z_end = endZ[i + window_steps]
        denom = max(abs(z_start), 1e-9)
        rel_change = abs(z_end - z_start) / denom

        if rel_change < 0.01:    # <1% change over 1 second
            steady_time = (i + window_steps) * dt
            steady_value = z_end
            break

    return steady_time, steady_value, endZ, local_snaps


# ============================================================
# Force sweep (Part B) for current geometry
# ============================================================

def run_force_sweep(F_char, num_forces=5, collect_time_histories=False, collect_snaps_for_last=False):
    """
    Perform a log-spaced force sweep for the current geometry and
    compute a linear stiffness k from the small-displacement regime.

    Returns
    -------
    k_fit : float              # stiffness from linear fit
    F_vals : (num_forces,)     # force magnitudes used
    dz_star : (num_forces,)    # steady δz* for each force
    steady_times : (num_forces,)  # steady time (or nan) for each force
    endZ_all : list of arrays      # δz(t) histories (one per force) if requested
    snapshots : list of (q, t)     # snapshots for last force if requested
    """

    # Log-spaced forces between 0.01 F_char and 10 F_char
    F_sweep = np.logspace(np.log10(0.01 * F_char),
                          np.log10(10.0 * F_char),
                          num=num_forces)

    F_vals = np.zeros(num_forces)
    dz_star = np.zeros(num_forces)
    steady_times = np.full(num_forces, np.nan)

    endZ_all = []
    snapshots = []

    # index of end node
    c_end = nv - 1
    ind_end = [4 * c_end, 4 * c_end + 1, 4 * c_end + 2]

    for k_idx, F_mag in enumerate(F_sweep):
        print(f"\n===== Force sweep {k_idx + 1}/{num_forces}: |F| = {F_mag:.3e} N =====")

        F_ext = np.zeros(ndof)
        F_ext[ind_end] += np.array([0.0, 0.0, -F_mag])

        do_snaps = collect_snaps_for_last and (k_idx == num_forces - 1)

        steady_time_k, steady_value_k, endZ_k, snaps_k = run_to_steady(F_ext, do_snapshots=do_snaps)

        F_vals[k_idx] = F_mag
        dz_star[k_idx] = steady_value_k
        steady_times[k_idx] = steady_time_k if steady_time_k is not None else np.nan

        if collect_time_histories:
            endZ_all.append(endZ_k.copy())
        else:
            endZ_all.append(None)

        if do_snaps:
            snapshots = snaps_k

    # --------- Linear fit: |F| = k |δz*| + b in small-displacement regime ---------
    F_arr = F_vals
    dz_arr = dz_star
    dz_abs = np.abs(dz_arr)

    N_fit = min(3, len(dz_abs))   # use the 2–3 smallest |δz*|
    idx_sorted = np.argsort(dz_abs)
    idx_fit = idx_sorted[:N_fit]

    dz_small = dz_abs[idx_fit]
    F_small = F_arr[idx_fit]

    print("\nPoints used for linear fit (|δz*|, F):")
    for z, f in zip(dz_small, F_small):
        print(f"  |δz*| = {z:.6e} m,  F = {f:.6e} N")

    k_fit, b_fit = np.polyfit(dz_small, F_small, 1)
    print(f"\nLinear stiffness fit (small-displacement region):")
    print(f"  k ≈ {k_fit:.3e} N/m, intercept b ≈ {b_fit:.3e} N")

    return k_fit, F_vals, dz_star, steady_times, endZ_all, snapshots


# ============================================================
# MAIN: Diameter sweep (Part 3) + comparison with textbook trend
# ============================================================

if __name__ == "__main__":

    # ---- Sweep D from 0.01 m to 0.05 m (at least 10 values) ----
    D_values = np.linspace(0.01, 0.05, 10)

    k_numeric = []
    k_textbook = []

    # Optionally collect detailed time histories for the last diameter only
    endZ_all_lastD = None
    F_vals_lastD = None
    steady_times_lastD = None

    for i, D_val in enumerate(D_values):
        # Set up geometry, mass, frames for this diameter
        setup_geometry_for_D(D_val)

        # Characteristic axial force for this geometry (use axial length)
        F_char = EI / (L**2)

        # For the last diameter, record δz(t) for each force & snapshots
        collect_histories = (i == len(D_values) - 1)
        collect_snaps = (i == len(D_values) - 1)

        k_fit, F_vals, dz_star, steady_times, endZ_all, snapshots = run_force_sweep(
            F_char,
            num_forces=5,
            collect_time_histories=collect_histories,
            collect_snaps_for_last=collect_snaps
        )

        # Store numerical stiffness
        k_numeric.append(k_fit)

        # Textbook stiffness: k_text = G d^4 / (8 N D^3)
        k_text = G * d_rod**4 / (8.0 * N * D_val**3)
        k_textbook.append(k_text)

        # Save details for last D for plotting δz(t)
        if collect_histories:
            endZ_all_lastD = endZ_all
            F_vals_lastD = F_vals
            steady_times_lastD = steady_times

    k_numeric = np.array(k_numeric)
    k_textbook = np.array(k_textbook)

    # ============================================================
    # Plot Part (3): numerical k vs textbook stiffness k_text
    # ============================================================

    plt.figure()
    plt.plot(k_textbook, k_numeric, 'o', label='DER stiffness (simulation)')

    # Reference line y = x (slope 1 through origin)
    x_line = np.linspace(0.0, 1.1 * k_textbook.max(), 200)
    plt.plot(x_line, x_line, '-', label='Textbook: k = k_text')

    plt.xlabel(r'Textbook stiffness $k_{\mathrm{text}} = G d^4 / (8 N D^3)$ [N/m]')
    plt.ylabel(r'Numerical stiffness $k$ from DER [N/m]')
    plt.title('Diameter Sweep: Numerical vs Textbook Axial Stiffness')
    plt.grid(True)
    plt.legend()

    # ============================================================
    # Extra: δz(t) vs t for each force, for the LAST diameter
    # (this is essentially Part B shown for one geometry)
    # ============================================================

    if endZ_all_lastD is not None:
        plt.figure()
        time_array = np.arange(Nsteps) * dt

        for k_idx, endZ_k in enumerate(endZ_all_lastD):
            if endZ_k is None:
                continue
            plt.plot(time_array, endZ_k, label=f"F = {F_vals_lastD[k_idx]:.2e} N")

            st = steady_times_lastD[k_idx]
            if not np.isnan(st):
                # approximate steady δz*: evaluate δz at nearest time index
                idx_st = int(round(st / dt))
                if 0 <= idx_st < len(endZ_k):
                    plt.plot(st, endZ_k[idx_st], 'kx')

        plt.xlabel("Time (s)")
        plt.ylabel("End-node δz(t) [m]")
        plt.title(f"End-node δz(t) for all forces (last D = {D_values[-1]:.3f} m)")
        plt.grid(True)
        plt.legend()

        # Optional: plot geometry snapshots for last D & last force
        for (q_snap, t_snap) in snapshots:
            plotrod_simple(q_snap, t_snap)

    plt.show()
