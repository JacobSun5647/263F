import numpy as np
import matplotlib.pyplot as plt
from mat_refFrames import computeTangent
from mat_refFrames import computeSpaceParallel
from mat_refFrames import computeMaterialDirectors
from objfun import objfun
from refTwist_Curv import getKappa
from objfun import objfun
from plot import plotrod
from plot import plotrod_simple



# Inputs
nv = 50 # number of nodes
ne = nv - 1
ndof = 3*nv + ne

# Helix parameters
r0 = 0.001 # cross-sectional radius of the rod # Given, d = 0.002 m
D = 0.04 # meter: helix diameter
pitch = 2 * r0  # Pitch is the same as the cross-sectional diameter
N = 5 # Number of turns
# a and b are parameters used in standard (wikipedia) definition of helix
a = D/2 # Helix radius
b = pitch / (2.0 * np.pi)
T = 2.0 * np.pi * N # Angle created by the helix (N turns in the center)
L = T * np.sqrt( (2*np.pi*a)**2 + pitch ** 2) # Arc length of the helix
axial_l = N * pitch # Axial length

print('Helix diameter = ', D)
print('Pitch = ', pitch)
print('N = ', N)
print('Arc length = ', L)
Estimated_Arc = np.pi * D * N
print('Estimated arc length = ', Estimated_Arc)
print('axial_l = ', axial_l)

# Create our nodes matrix
nodes = np.zeros((nv, 3))
for c in range(nv):
  t = c * T / (nv - 1.0)
  nodes[c,0] = a * np.cos(t)
  nodes[c,1] = a * np.sin(t)
  nodes[c,2] = - b * t

# Material Parameters
Y = 10e6  # 10 MPa - Young's modulus
nu = 0.5  # Poisson's ration
G = Y / (2 * (1 + nu))  # Shear modulus

# Stiffness variables
EA = Y * np.pi * r0 ** 2  # Stretching stiffness
EI = Y * np.pi * r0 ** 4 / 4.0  # Bending stiffness
GJ = G * np.pi * r0 ** 4 / 2.0  # Twisting stiffness


totalTime = 5.0  # seconds - total time of the simulation
dt = 0.01  # TIme step size -- may need to be adjusted

# Tolerance
tol = EI / L ** 2 * 1e-3

rho = 1000  # kg/m^3 -- density
totalM = L * np.pi * r0 ** 2 * rho  # Total mass of the rod
dm = totalM / ne

massVector = np.zeros(ndof)
for c in range(nv):
    ind = [4 * c, 4 * c + 1, 4 * c + 2]  # x, y, z coordinates of c-th node
    if c == 0 or c == nv - 1:
          massVector[ind] = dm / 2
    else:
          massVector[ind] = dm

for c in range(ne):
    massVector[4 * c + 3] = 0.5 * dm * r0 ** 2  # Equation for a solid cylinder
    # Because r0 is really small, we may get away with just using 0 angular mass

massMatrix = np.diag(massVector)

F_end = EI / L ** 2
vectorLoad = np.array([0, 0, -F_end])  # Point load vector

Fg = np.zeros(ndof)  # Eexternal force vector
c = nv - 1
ind = [4 * c, 4 * c + 1, 4 * c + 2]  # last node
Fg[ind] += vectorLoad




qOld = np.zeros(ndof)
for c in range(nv):
  ind = [4*c, 4*c + 1, 4*c + 2] # c-th node
  qOld[ind] = nodes[c, :]

uOld = np.zeros_like(qOld) # Velocity is zero initially

plotrod_simple(qOld, 0)


# Reference length of each edge
refLen = np.zeros(ne)
for c in range(ne):
    refLen[c] = np.linalg.norm(nodes[c + 1, :] - nodes[c, :])

voronoiRefLen = np.zeros(nv)
for c in range(nv):
    if c == 0:
          voronoiRefLen[c] = 0.5 * refLen[c]
    elif c == nv - 1:
          voronoiRefLen[c] = 0.5 * refLen[c - 1]
    else:
          voronoiRefLen[c] = 0.5 * (refLen[c - 1] + refLen[c])


# Reference frame (At t=0, we initialize it with space parallel reference frame but not mandatory)
tangent = computeTangent(qOld)

t0 = tangent[0, :]
arb_v = np.array([0, 0, -1])
a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))
if np.linalg.norm(np.cross(t0, arb_v)) < 1e-3: # Check if t0 and arb_v are parallel
  arb_v = np.array([0, 1, 0])
  a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))

a1, a2 = computeSpaceParallel(a1_first, qOld)

# Material frame
theta = qOld[3::4] # Extract theta angles
m1, m2 = computeMaterialDirectors(a1, a2, theta)


# Reference twist
refTwist = np.zeros(nv) # Or use the function we computed

# Natural curvature
kappaBar = getKappa(qOld, m1, m2)

# Natural twist
twistBar = np.zeros(nv)


# Fixed and free DOFs
fixedIndex = np.arange(0, 7)
freeIndex = np.arange(7, ndof)
# If we include the x and y coordinates of the last node as FIXED DOFs, we will get better agreement

Nsteps = round(totalTime / dt ) # number of steps
ctime = 0 # Current time
endZ_0 = qOld[-1] # End Z coordinate of the first node
endZ = np.zeros(Nsteps)

a1_old = a1
a2_old = a2

snapshots = []    # Will store (q, time) for plotting geometry (we'll use it for the last force only)


# ======================= Helper: run simulation for a given load =======================

def run_to_steady(F_ext, do_snapshots=False):
    """
    Run the dynamic simulation for a given external force vector F_ext
    until totalTime, then estimate steady-state end displacement δz*.

    Returns
    -------
    steady_time : float or None
        Time at which steady-state criterion is first met (or None).
    steady_value : float
        Steady δz* value (z-displacement of end node).
    endZ : (Nsteps,) array
        Time history of δz(t).
    local_snaps : list of (q, t)
        Snapshots of configurations if do_snapshots=True.
    """

    # --- initial configuration (helix) ---
    qOld = np.zeros(ndof)
    for c in range(nv):
        ind = [4 * c, 4 * c + 1, 4 * c + 2]  # c-th node
        qOld[ind] = nodes[c, :]

    uOld = np.zeros_like(qOld)
    a1_old = a1.copy()
    a2_old = a2.copy()

    ctime = 0.0
    endZ = np.zeros(Nsteps)
    local_snaps = []

    for timeStep in range(Nsteps):
        print('Current time: ', ctime)

        q_new, u_new, a1_new, a2_new = objfun(qOld, uOld, a1_old, a2_old,
                                              freeIndex, dt, tol, refTwist,
                                              massVector, massMatrix,
                                              EA, refLen,
                                              EI, GJ, voronoiRefLen,
                                              kappaBar, twistBar,
                                              F_ext)

        # Save endZ (z coordinate displacement of the last node)
        endZ[timeStep] = q_new[-1] - endZ_0

        if do_snapshots and (timeStep % 500 == 0 or timeStep == Nsteps - 1):
            local_snaps.append((q_new.copy(), ctime))

        ctime += dt  # Current time

        # Old parameters become new
        qOld = q_new.copy()
        uOld = u_new.copy()
        a1_old = a1_new.copy()
        a2_old = a2_new.copy()

    # ---- Detect steady state of end Z for THIS force ----
    window_duration = 1.0                       # seconds
    window_steps = int(window_duration / dt)    # number of steps in 1 second

    steady_time = None
    steady_value = endZ[-1]   # fallback: last value

    for i in range(0, Nsteps - window_steps):
        z_start = endZ[i]
        z_end = endZ[i + window_steps]

        # Avoid division by zero: if z_start is tiny, use absolute change only
        denom = max(abs(z_start), 1e-9)
        rel_change = abs(z_end - z_start) / denom

        if rel_change < 0.01:  # less than 1% change over 1 second
            steady_time = (i + window_steps) * dt
            steady_value = z_end
            break

    if steady_time is not None:
        print(f"Steady state detected at t ≈ {steady_time:.3f} s, δz* ≈ {steady_value:.6e} m")
    else:
        print("No strict steady state found within simulation time (1%/1s criterion); "
              f"using final δz* ≈ {steady_value:.6e} m")

    return steady_time, steady_value, endZ, local_snaps


# ======================= Force sweep: 0.01 F_char to 10 F_char =======================

# Characteristic axial force
F_char = F_end   # same as EI / axial_l**2

# Log-spaced forces from 0.01 F_char to 10 F_char
num_forces = 5
F_sweep = np.logspace(np.log10(0.01 * F_char),
                      np.log10(10.0 * F_char),
                      num=num_forces)

# Arrays to store F and δz* from each run
F_vals = np.zeros(num_forces)
dz_star = np.zeros(num_forces)
steady_times = np.full(num_forces, np.nan)

# index of end node (where axial force is applied)
c_end = nv - 1
ind_end = [4 * c_end, 4 * c_end + 1, 4 * c_end + 2]

endZ_last = None  # to store time history for last force run
endZ_all = []     # will store endZ history for each force


for k, F_mag in enumerate(F_sweep):
    print(f"\n===== Force sweep {k+1}/{num_forces}: |F| = {F_mag:.3e} N =====")

    # Build external force vector for this run (axial downward load at last node).
    F_ext = np.zeros(ndof)
    F_ext[ind_end] += np.array([0.0, 0.0, -F_mag])

    # For the last force level, also collect shape snapshots for plotting
    do_snaps = (k == num_forces - 1)

    steady_time_k, steady_value_k, endZ_k, snaps_k = run_to_steady(F_ext, do_snapshots=do_snaps)

    F_vals[k] = F_mag
    dz_star[k] = steady_value_k
    steady_times[k] = steady_time_k if steady_time_k is not None else np.nan

    # store full time history for this force
    endZ_all.append(endZ_k.copy())

    if do_snaps:
        endZ_last = endZ_k.copy()
        snapshots = snaps_k  # overwrite global snapshots with last run's snapshots

# ======================= Plot geometry snapshots for last force =======================

for i, (q_snap, t_snap) in enumerate(snapshots):
    plotrod_simple(q_snap, t_snap)


# ======================= Plot F vs δz* and fit F = k δz* =======================

# ======================= Plot F vs δz* and fit F = k δz* =======================

F_arr = F_vals
dz_arr = dz_star

# Use absolute displacement for fitting (we care about magnitude)
dz_abs = np.abs(dz_arr)

# ---- Automatically choose "small-displacement" region ----
# Sort points by |δz*| and take the smallest N_fit points
N_fit = min(3, len(dz_abs))   # use 2–3 smallest points depending on data size
idx_sorted = np.argsort(dz_abs)
idx_fit = idx_sorted[:N_fit]

dz_small = dz_abs[idx_fit]
F_small = F_arr[idx_fit]

print("\nPoints used for linear fit (|δz*|, F):")
for z, f in zip(dz_small, F_small):
    print(f"  |δz*| = {z:.6e} m,  F = {f:.6e} N")

# ---- Linear fit: |F| = k |δz*| + b ----
k, b = np.polyfit(dz_small, F_small, 1)
print(f"\nLinear stiffness fit (small-displacement region):")
print(f"  k ≈ {k:.3e} N/m, intercept b ≈ {b:.3e} N")

# ---- Plot F vs |δz*| with fit ----
plt.figure()
plt.plot(dz_abs, F_arr, 'o', label='simulation data')

dz_fit = np.linspace(0.0, dz_small.max(), 200)
F_fit = k * dz_fit + b
plt.plot(dz_fit, F_fit, '-', label=f'fit: |F| = k |δz*|, k={k:.3e} N/m')

plt.xlabel("Steady displacement |δz*| (m)")
plt.ylabel("Applied force F (N)")
plt.title("Force vs steady displacement")
plt.legend()
plt.grid(True)



# ======================= Plot δz(t) for each force level =======================

if endZ_all:
    plt.figure()
    time_array = np.arange(1, Nsteps + 1) * dt

    for k, endZ_k in enumerate(endZ_all):
        # Plot δz(t) curve for this force
        plt.plot(time_array, endZ_k, label=f"F = {F_vals[k]:.2e} N")

        # Mark steady state as a point (if detected)
        st = steady_times[k]
        sv = dz_star[k]
        if not np.isnan(st):
            plt.plot(st, sv, 'kx')  # black 'X' at steady state

    plt.xlabel("Time (s)")
    plt.ylabel("End Z (m)")
    plt.title("End Z vs Time for all force levels")
    plt.legend()
    plt.grid(True)

plt.show()
