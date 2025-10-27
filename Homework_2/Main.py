import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.psLib import endofthingRE

from crossMat import crossMat
from getFb import getFb
from getFs import getFs
from gradEb import gradEb
from gradES import gradEs
from hessEs import hessEs
from hessEb import hessEb
from objfun import objfun

nv = 50 # number of nodes/vertices
ndof = 2 * nv
midNode = nv//2 + 1
ymaxnode = 75

# Time step
dt = 0.01 # second

# Rod length
RodLength = 1 # meter

# Discrete length / reference length
deltaL = RodLength / (nv - 1)

# Radii of spheres (given)
R = np.zeros(nv)
for k in range(nv):
  R[k] = deltaL/10 # meter
R[midNode-1] = 0.025 # meter

# Densities
rho_metal = 2700 # kg/m^3
rho = rho_metal

# Cross sectional radius
ro = 0.013 # meter
ri = 0.011
# Young's modulus
Y = 70e9

# Viscosity
#visc = 1000.0 # Pa-s

# Maximum number of iterations
maximum_iter = 1000

# Total time
totalTime = 5 # second

# Variables related to plotting
saveImage = 0
plotStep = 500 # Every 5-th step will be plotted

# Utility quantites
ne = nv - 1 # number of edges
EI = Y * np.pi * (ro**4 - ri**4) / 4
EA = Y * np.pi * (ro**2 - ri**2)

# Tolerance
tol = EI / RodLength ** 2 * 1e-3

# Geometry
nodes = np.zeros((nv, 2))
for c in range(nv):
  nodes[c, 0] = c * deltaL # x-coordinate
  nodes[c, 1] = 0.0 # y-coordinate

# Mass vector and matrix
m = np.zeros( 2 * nv )
for k in range(0, nv):
  m[2*k] = 4/3 * np.pi * R[k]**3 * rho_metal # mass of k-th node along x
  m[2*k + 1] = 4/3 * np.pi * R[k]**3 * rho_metal # mass of k-th node along y
mMat = np.diag(m)

# Gravity (external force)
W = np.zeros( 2 * nv)
g = np.array([0, -9.8]) # m/s^2


for k in range(0, nv):
  if k == 37:
    W[2 * k] = 4.0 / 3.0 * np.pi * R[k] ** 3 * rho * g[0]
    W[2*k+1] = -2000
  else:
    W[2 * k] = 4.0 / 3.0 * np.pi * R[k] ** 3 * rho * g[0]
    W[2 * k+1] = 4.0 / 3.0 * np.pi * R[k] ** 3 * rho * g[0]

C = np.zeros((2 * nv, 2 * nv))

# Initial conditions
q0 = np.zeros(2 * nv)
for c in range(nv):
  q0[2*c] = nodes[c, 0] # x coordinate
  q0[2*c+1] = nodes[c, 1] # y coordinate

u0 = np.zeros(2 * nv) # old velocity

all_DOFs = np.arange(ndof) # Set of all DOFs
fixed_index = np.array([0, 1, ndof-1]) # Fixed DOFs

# Free index
free_index = np.setdiff1d(all_DOFs, fixed_index) # All the DOFs are free except the fixed ones

# Number of steps
Nsteps = round( totalTime / dt )

ctime = 0 # Current time

# Store the y-coordinate of the middle node, its velocity, and the angle
all_pos = np.zeros(Nsteps)
all_vel = np.zeros(Nsteps)
mid_angle = np.zeros(Nsteps)

all_pos[0] = 0
all_vel[0] = 0
mid_angle[0] = 0

# --- containers for maximum vertical displacement ---

Nsteps = int(totalTime/dt) + 1
y_min_series = np.zeros(Nsteps)
x_at_ymin    = np.zeros(Nsteps)

# Loop over the time steps
# --- containers for maximum vertical displacement ---
Nsteps = int(totalTime / dt) + 1
y_min_series = np.zeros(Nsteps)
x_at_ymin = np.zeros(Nsteps)

ctime = 0.0

# --- time integration loop ---
for timeStep in range(1, Nsteps):
    q_new, error = objfun(q0, u0, dt, tol, maximum_iter, m, mMat, EI, EA, W, C,
                          deltaL, free_index)
    if error < 0:
        print('Could not converge.')
        break

    u_new = (q_new - q0) / dt
    ctime += dt

    # --- find node with maximum downward displacement among FREE y-DOFs ---
    x_arr = q_new[::2]
    y_arr = q_new[1::2]

    y_mask = (free_index % 2 == 1)
    y_free_nodes = (free_index[y_mask] // 2)
    y_free_vals = y_arr[y_free_nodes]

    imin = np.argmin(y_free_vals)
    node_ymin = y_free_nodes[imin]

    y_min_series[timeStep] = y_free_vals[imin]
    x_at_ymin[timeStep] = x_arr[node_ymin]

    # optional: plot the rod shape every few steps
    if timeStep % plotStep == 0:
        plt.figure(1)
        plt.clf()
        plt.plot(x_arr, y_arr, 'ko-')
        plt.plot(x_arr[node_ymin], y_arr[node_ymin], 'ro', markersize=8)
        plt.title(f't={ctime:.2f}s')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.xlim(0.0, RodLength)
        #plt.show()

    # update for next step
    q0 = q_new.copy()
    u0 = u_new.copy()

# --- plot full-time results once after the loop ---
t_arr = np.linspace(0, totalTime, Nsteps)

plt.figure(2)
plt.clf()
plt.plot(t_arr, y_min_series, 'ko-')
plt.xlabel('Time [s]')
plt.ylabel('Maximum downward deflection (m)')
plt.title('y_min (maximum downward displacement) vs. time')
#plt.show()

plt.figure(3)
plt.clf()
plt.plot(t_arr, x_at_ymin, 'ko-')
plt.xlabel('Time [s]')
plt.ylabel('x-location of max deflection (m)')
plt.title('Position of maximum deflection vs. time')
#plt.show()


# ================== P vs y_min over time ==================

a_target = 0.75
force_node = int(np.argmin(np.abs(nodes[:, 0] - a_target)))
print(f"[dynamic sweep] force_node={force_node}, x={nodes[force_node,0]:.6f} m")


P_vals = np.linspace(20.0, 20000.0, 21)
y_min_vs_P = np.zeros_like(P_vals)


q_init = np.zeros(ndof)
for cnode in range(nv):
    q_init[2*cnode]   = nodes[cnode, 0]
    q_init[2*cnode+1] = nodes[cnode, 1]
u_init = np.zeros(ndof)

for i, P in enumerate(P_vals):
    # Reset state for EACH P so results are comparable
    qk = q_init.copy()
    uk = u_init.copy()
    ctime_local = 0.0

    # External load vector for this P:
    #   - zero “gravity” everywhere
    #   - ONLY a downward (negative Y) point load at the selected node
    W_load = np.zeros(ndof)
    W_load[2*force_node + 1] = -P

    # Track the minimum Y among FREE nodes over the whole time history
    min_over_time = 1.0e30

    # Progress
    print(f"[{i+1}/{len(P_vals)}] P = {P:.1f} N : starting...", flush=True)

    # Time integration using your objfun
    for timeStep in range(1, Nsteps):
        q_new, error = objfun(qk, uk, dt, tol, maximum_iter, m, mMat, EI, EA, W_load, C,
                              deltaL, free_index)
        if error < 0:
            print(f"  step {timeStep}: Could not converge at P={P:.1f} N", flush=True)
            break

        u_new = (q_new - qk) / dt
        ctime_local += dt

        # Extract coordinates
        x_arr = q_new[::2]
        y_arr = q_new[1::2]

        # Among FREE y-DOFs, get current minimum (most negative)
        y_mask = (free_index % 2 == 1)
        y_free_nodes = (free_index[y_mask] // 2)
        y_free_vals = y_arr[y_free_nodes]

        y_curr_min = y_free_vals.min()
        if y_curr_min < min_over_time:
            min_over_time = y_curr_min

        # Optional progress every ~200 steps
        if (timeStep % 200 == 0) or (timeStep == Nsteps - 1):
            print(f"  step {timeStep}/{Nsteps-1}: y_min_so_far = {min_over_time:.6e} m", flush=True)

        # Advance state (local only)
        qk = q_new.copy()
        uk = u_new.copy()

    # Store one value per P (length matches P_vals)
    y_min_vs_P[i] = min_over_time if np.isfinite(min_over_time) else np.nan
    print(f"--> P = {P:.1f} N : min_Y_over_time = {y_min_vs_P[i]:.6e} m", flush=True)

# Plot: max downward deflection (negative) vs Force
plt.figure()
plt.plot(P_vals, y_min_vs_P, 'ko-')
plt.xlabel('Load P [N]')
plt.ylabel('Maximum downward deflection over time, y_min [m]')
plt.title(' 20 - 20000N  vs y_min ')
plt.grid(True)
plt.tight_layout()
# ================== P vs y_min over time  ==================



# ===================== BEAM THEORY PREDICTION  =====================
L = RodLength

# load position d (m) from the left support
d = 0.75
c = min(d, L - d)
I_geom = np.pi/4 * (ro**4 - ri**4)

def y_max_beam_general(P, L, E, I, c):
    # negative sign => downward (−y) deflection
    return -(P * c * (L**2 - c**2)**1.5) / (9*np.sqrt(3) * E * I * L)

# Use the same P grid as your sweep
# (If P_vals already exists, this just reuses it.)
P_vals = np.linspace(20.0, 20000.0, 21)
ymax_beam = np.array([y_max_beam_general(P, RodLength, Y, I_geom, c) for P in P_vals])

# Optional standalone beam-theory figure (now with a label so legend works)
plt.figure()
plt.plot(P_vals, ymax_beam, 'b--o', label='Euler–Bernoulli (theory)')
plt.xlabel('Load P [N]')
plt.ylabel('Maximum deflection y_max [m]')
plt.title('Beam Theory: P vs. Maximum Deflection')
plt.axhline(0, linewidth=0.8)  # reference line at y=0
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


ymax_static = y_min_vs_P.copy()  # rename for plotting; values are ≤ 0 (downward)

plt.figure()
plt.plot(P_vals, ymax_static, 'ko-', label='Implicit Euler simulation')
plt.plot(P_vals, ymax_beam,  'b--o', label='Euler–Bernoulli (theory)')
plt.xlabel('Load P [N]')
plt.ylabel('Maximum (downward) deflection y [m]')
plt.title('P vs y_max: Simulation vs Beam Theory')
plt.axhline(0, linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

