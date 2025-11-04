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

nv = 51 # number of nodes/vertices
ndof = 2 * nv
midNode = nv//2 + 1

# Time step
dt = 1 # second

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
totalTime = 1001 # second

# Variables related to plotting
saveImage = 0
plotStep = 100

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
for k in range(nv):
  m[2*k] = 4/3 * np.pi * R[k]**3 * rho_metal # mass of k-th node along x
  m[2*k + 1] = 4/3 * np.pi * R[k]**3 * rho_metal # mass of k-th node along y
mMat = np.diag(m)

# Gravity (external force)
W = np.zeros( 2 * nv)
g = np.array([0, -9.8]) # m/s^2


for k in range(0, nv):
    W[2 * k] = 4.0 / 3.0 * np.pi * R[k] ** 3 * rho * g[0]
    W[2 * k+1] = 4.0 / 3.0 * np.pi * R[k] ** 3 * rho * g[1]

C = np.zeros((2 * nv, 2 * nv))

# Initial conditions
q0 = np.zeros(2 * nv)
for c in range(nv):
  q0[2*c] = nodes[c, 0] # x coordinate
  q0[2*c+1] = nodes[c, 1] # y coordinate

u0 = np.zeros(2 * nv) # old velocity

all_DOFs = np.arange(ndof) # Set of all DOFs
fixed_index = np.array([0, 1]) # Fixed DOFs

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
y_min_series = np.zeros(Nsteps)
x_at_ymin    = np.zeros(Nsteps)

# Loop over the time steps
# ---- indices for the last two nodes (0-based) ----
iN  = nv - 1       # last node index in "nodes" space
iNm = nv - 2       # penultimate node index

ixN,  iyN  = 2*iN,  2*iN+1
ixNm, iyNm = 2*iNm, 2*iNm+1

def end_eff_traj(t):
    """Quarter-circle from (1,0) to (0,-1) over t in [0, 1000]."""
    phi = 0.5*np.pi * (t / 1000.0)          # 0 -> pi/2
    x_c = np.cos(phi)
    y_c = -np.sin(phi)
    # Use the radial angle for orientation (simple, works well here)
    theta_c = np.arctan2(y_c, x_c)          # equals -phi for this path
    return x_c, y_c, theta_c



xc_series = np.zeros(Nsteps)
yc_series = np.zeros(Nsteps)
th_series = np.zeros(Nsteps)

# --- control input series include t=0 ---
xc0, yc0, th0 = end_eff_traj(0.0)
xc_series[0] = xc0
yc_series[0] = yc0
th_series[0] = th0


snapshot_times = set(range(0, int(totalTime) + 1, 100))
snapshots = {}

# middle node index (0-based)
mid_idx = nv // 2

# --- time integration loop ---
for timeStep in range(1, Nsteps):
    # 1) End-effector reference for this time
    xc, yc, th = end_eff_traj(ctime)

    # 2) Enforce Dirichlet values on fixed DOFs in the state vector q0
    q0[ixN]  = xc
    q0[iyN]  = yc
    q0[ixNm] = xc - deltaL*np.cos(th)
    q0[iyNm] = yc - deltaL*np.sin(th)

    # 3) build "fixed_index" to include (x1,y1), (x_{N-1},y_{N-1}), (x_N,y_N)
    fixed_index = np.array([0, 1, ixNm, iyNm, ixN, iyN], dtype=int)
    free_index  = np.setdiff1d(all_DOFs, fixed_index)

    # 4) Log control inputs (for deliverable plots)
    xc_series[timeStep] = xc
    yc_series[timeStep] = yc
    th_series[timeStep] = th

    # 5) Advance one implicit step with your solver (forces include gravity as you already set)
    q_new, error = objfun(q0, u0, dt, tol, maximum_iter, m, mMat, EI, EA, W, C,
                          deltaL, free_index)
    if error < 0:
        print('Could not converge.')
        break

    u_new = (q_new - q0) / dt
    ctime += dt

    x_arr = q_new[::2]
    y_arr = q_new[1::2]

    # --- NEW: snapshot every 100 s  ---

    if int(ctime) in snapshot_times and int(ctime) not in snapshots:
        snapshots[int(ctime)] = (x_arr.copy(), y_arr.copy())



    if timeStep % plotStep == 0:
        plt.figure(1);
        plt.clf()
        plt.plot(x_arr, y_arr, 'ko-')
        plt.plot(x_arr[mid_idx], y_arr[mid_idx], 'ro', markersize=8)
        plt.title(f't={ctime:.2f}s')
        plt.xlabel('x [m]');
        plt.ylabel('y [m]')
        plt.axis('equal');
        #plt.show()
        #plt.xlim(0.0, RodLength)
    plt.show

    q0 = q_new.copy()
    u0 = u_new.copy()

t_arr = np.linspace(0, totalTime, Nsteps)

# ---- (1) Control inputs over time: xc(t), yc(t), theta_c(t) ----
plt.figure()
plt.plot(t_arr, xc_series)
plt.xlabel('t [s]'); plt.ylabel('x_c(t) [m]')
plt.title('x_c over time'); plt.tight_layout()

plt.figure()
plt.plot(t_arr, yc_series)
plt.xlabel('t [s]'); plt.ylabel('y_c(t) [m]')
plt.title('y_c over time'); plt.tight_layout()

plt.figure()
plt.plot(t_arr, th_series)
plt.xlabel('t [s]'); plt.ylabel(r'$\theta_c(t)$ [rad]')
plt.title(r'$\theta_c$ over time'); plt.tight_layout()


# ---- (3) Rod shape snapshots every 10 s, middle node marked red ----
# One figure per snapshot to keep it simple & readable.
for t_snap in sorted(snapshots.keys()):
    xa, ya = snapshots[t_snap]
    plt.figure()
    plt.plot(xa, ya, 'k.-', linewidth=1)
    # mark middle node red
    plt.plot(xa[mid_idx], ya[mid_idx], 'ro', markersize=6, label='middle node')
    plt.axis('equal')
    plt.xlim(0.0, RodLength)
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.title(f'Rod shape at t = {t_snap} s')
    plt.legend(loc='best')
    plt.tight_layout()

plt.show()
