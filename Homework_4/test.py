import numpy as np

# Inputs
nv = 50 # number of nodes
ne = nv - 1
ndof = 3*nv + ne

# Material Parameters
Y = 10e6  # 10 MPa - Young's modulus
nu = 0.5  # Poisson's ration
G = Y / (2 * (1 + nu))  # Shear modulus


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

EA = Y * np.pi * r0 ** 2  # Stretching stiffness
EI = Y * np.pi * r0 ** 4 / 4.0  # Bending stiffness
GJ = G * np.pi * r0 ** 4 / 2.0  # Twisting stiffness

F_end = EI / axial_l ** 2
vectorLoad = np.array([0, 0, -F_end])  # Point load vector

F_sweep = np.logspace(0.01*F_end,10*F_end,num = 5,base = 10)
n = len(F_sweep)
Fg = np.zeros(ndof)  # Eexternal force vector
c = nv - 1
ind = [4 * c, 4 * c + 1, 4 * c + 2]  # last node
F0 = np.zeros(ndof)
F0[ind] += np.array([0, 0, -F_sweep[0]])

F1 = np.zeros(ndof)
F1[ind] += np.array([0, 0, -F_sweep[1]])

F2 = np.zeros(ndof)
F2[ind] += np.array([0, 0, -F_sweep[2]])

F3 = np.zeros(ndof)
F3[ind] += np.array([0, 0, -F_sweep[3]])

F4 = np.zeros(ndof)
F4[ind] += np.array([0, 0, -F_sweep[4]])

F_Sweep = [F0,F1,F2,F3,F4]
for i in range(len(F_Sweep)):
    F_load = F_Sweep[i]
    print(F_load)


