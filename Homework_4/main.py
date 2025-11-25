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
dt = 0.002  # TIme step size -- may need to be adjusted

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

snapshots = []    # Will store (q, time)



for timeStep in range(Nsteps):
    print('Current time: ', ctime)

    q_new, u_new, a1_new, a2_new = objfun(qOld, uOld, a1_old, a2_old,
                                        freeIndex, dt, tol, refTwist,
                                        massVector, massMatrix,
                                        EA, refLen,
                                        EI, GJ, voronoiRefLen,
                                        kappaBar, twistBar,
                                        Fg)

    # Save endZ (z coordinate of the last node)
    endZ[timeStep] = q_new[-1] - endZ_0

    if timeStep % 500 == 0 or timeStep == Nsteps - 1:
        snapshots.append((q_new.copy(), ctime))

    ctime += dt # Current time
     # Old parameters become new
    qOld = q_new.copy()
    uOld = u_new.copy()
    a1_old = a1_new.copy()
    a2_old = a2_new.copy()

# ---- Detect steady state of end Z ----
window_duration = 1.0                       # seconds
window_steps = int(window_duration / dt)    # number of steps in 1 second

steady_time = None
steady_value = None

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
    print(f"Steady state detected at t ≈ {steady_time:.3f} s, δz ≈ {steady_value:.6e} m")
else:
    print("No steady state found within simulation time (1%/1s criterion).")


# Plot snapshots in separate windows
for i, (q_snap, t_snap) in enumerate(snapshots):
    plotrod_simple(q_snap, t_snap)   # <- just call it, no fig/ax here

# Now show everything
plt.figure()
label = (
    "δz(t)\n"
    f"steady time = {steady_time:.2f}s\n"
    f"steady δz  = {steady_value:.6e}m")

time_array = np.arange(1, Nsteps+1, 1) * dt
plt.plot(time_array, endZ, 'ro-', label= label)

# If a steady state was found, mark it
if steady_time is not None:
    plt.axvline(steady_time, linestyle='--', label='steady-state time')
    plt.axhline(steady_value, linestyle=':', label='steady δz')
    plt.legend()

plt.xlabel(f"Time (s)")
plt.ylabel(f"End Z (m)")
plt.title('End Z vs Time')
plt.show()



