import numpy as np
from numpy.ma.core import zeros_like

nv = 50
ndof = 2 * nv
midNode = nv//2 + 1
ymaxnode = 75
# Gravity (external force)
W = np.zeros( 2 * nv)
g = np.array([0, -9.8]) # m/s^2
rho_metal = 2700 # kg/m^3
rho = rho_metal
nv = 50

# Rod length
RodLength = 1 # meter
# Discrete length / reference length
deltaL = RodLength / (nv - 1)

# Cross sectional radius
ro = 0.013 # meter
ri = 0.011
# Young's modulus
Y = 70e9

# Radii of spheres (given)
R = np.zeros(nv)
for k in range(nv):
  R[k] = deltaL/10 # meter
R[midNode-1] = 0.025 # meter

P_force = np.linspace(20,20000,21)
print(P_force)
P_val = zeros_like(P_force)
for i in range(len(P_force)):
    P_val[i] = P_force[i]

    for k in range(0, nv):
        if k == 37:
            W[2 * k] = 4.0 / 3.0 * np.pi * R[k] ** 3 * rho * g[0]
            W[2 * k + 1] = -P_val
        else:
            W[2 * k] = 4.0 / 3.0 * np.pi * R[k] ** 3 * rho * g[0]
            W[2 * k + 1] = 4.0 / 3.0 * np.pi * R[k] ** 3 * rho * g[0]
