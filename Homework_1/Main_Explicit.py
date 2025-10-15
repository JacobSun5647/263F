import numpy as np
import matplotlib.pyplot as plt
from Eurler_Explicit import myInt_explicit
from plot import plot

# ---------------------------------------------------------------------
# (1) Example input files (comment out if you already have your own)
with open("nodes.txt", "w") as f:
    f.write("0, 0\n")
    f.write("1, 0\n")
    f.write("2, 0\n")
    f.write("1, -1\n")

with open("springs.txt","w") as f:
    f.write("0,1,10\n")
    f.write("1,2,20\n")
    f.write("2,3,5\n")
    f.write("3,0,5\n")
# ---------------------------------------------------------------------

# (2) Load nodes.txt  -> node_matrix (N,2)
nodes_file_path = 'nodes.txt'
node_coordinates = []
with open(nodes_file_path, 'r') as f:
    for line in f:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 2:
            x = float(parts[0]); y = float(parts[1])
            node_coordinates.append([x, y])
        else:
            print(f"Skipping line (bad format): {line.strip()}")
node_matrix = np.array(node_coordinates)
print("Node coordinates:\n", node_matrix)

# (3) Load springs.txt -> index_matrix (ns,4), stiffness_matrix (ns,)
springs_file_path = 'springs.txt'
index_info, stiffness_info = [], []
with open(springs_file_path, 'r') as f:
    for line in f:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 3:
            i = int(parts[0]); j = int(parts[1]); k = float(parts[2])
            index_info.append([2*i, 2*i+1, 2*j, 2*j+1])
            stiffness_info.append(k)
        else:
            print(f"Skipping line (bad format): {line.strip()}")
index_matrix     = np.array(index_info, dtype=int)
stiffness_matrix = np.array(stiffness_info, dtype=float)
print("Index matrix:\n", index_matrix)
print("Stiffnesses:\n", stiffness_matrix)

# (4) Build initial state
N    = node_matrix.shape[0]
ndof = 2 * N
x_old = np.zeros(ndof)
u_old = np.zeros(ndof)
for n in range(N):
    x_old[2*n]   = node_matrix[n,0]
    x_old[2*n+1] = node_matrix[n,1]

# (5) Rest lengths l_k from initial geometry
l_k = np.zeros(stiffness_matrix.shape[0])
for s in range(stiffness_matrix.shape[0]):
    ind = index_matrix[s]
    xi, yi = x_old[ind[0]], x_old[ind[1]]
    xj, yj = x_old[ind[2]], x_old[ind[3]]
    l_k[s] = np.hypot(xj - xi, yj - yi)

# (6) Masses (diagonal)
m = np.ones(ndof)  # 1 kg per DOF

# (7) Time
dt = 0.000001
maxTime = 1.0
t = np.arange(0.0, maxTime + dt, dt)

# (8) Choose FREE DOFs (edit to your case)
# free_DOF = np.arange(2, ndof-2)     # example: clamp first/last nodes
free_DOF = np.array([2,3,6,7], dtype=int)  # your earlier choice
print("Free DOF:", free_DOF)

# (9) For demo: track y of node 1 (2nd node)
y_trace = np.zeros_like(t)
y_trace[0] = x_old[3]

# (10) Plot at selected times (only those within time span)
plot_times = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
plot_times = plot_times[(plot_times >= 0.0) & (plot_times <= maxTime)]

g = 9.81  # m/s^2  (downwards)

def gravity_fext(tk, xk, vk):
    F = np.zeros_like(xk)
    # apply weight on y-DOFs of all *free* nodes
    # every (x,y) pair -> y index is 2*i + 1
    for dof in free_DOF:
        if dof % 2 == 1:  # y DOF
            F[dof] = F[dof] - m[dof] * g
    return F


# ---- Plot t=0 BEFORE the loop ----
plot(x_old, index_matrix, t[0])

# (11) Time stepping
for k in range(len(t) - 1):
    t_new = t[k+1]

    x_new, u_new = myInt_explicit(
        t_new, x_old, u_old, free_DOF,
        stiffness_matrix, index_matrix, m, dt, l_k,
        fext_fn=gravity_fext
    )

    if np.isclose(plot_times, t_new, atol=1e-12).any():
        plot(x_new, index_matrix, t_new)

    # store, advance
    y_trace[k+1] = x_new[3]   # y of node 1 (2nd node)
    x_old = x_new
    u_old = u_new

# (12) Final history plot (this one can block)
plt.figure()
plt.plot(t, y_trace, 'ro-')
plt.xlabel('Time [second], ')
plt.ylabel('y-coordinate of the second node [meter]')
plt.grid(True, alpha=0.3)
plt.show()
