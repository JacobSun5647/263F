import numpy as np
import matplotlib.pyplot as plt
from myInt import myInt
from getForceJacobian import getForceJacobian
from getExternalForce import getExternalForce
from gradEs import gradEs
from hessEs import hessEs
from Eurler_Explicit import myInt_explicit
from plot import plot
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

nodes_file_path = 'nodes.txt'
node_coordinates = []

try:
    with open(nodes_file_path, 'r') as f:
        for line in f:
            # Split each line by comma and remove leading/trailing whitespace
            parts = [part.strip() for part in line.split(',')]
            # Assuming the format is node number, x, y
            # We only need x and y, which are the second and third elements (index 1 and 2)
            if len(parts) == 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    node_coordinates.append([x, y])
                except ValueError:
                    print(f"Skipping line due to non-numeric coordinates: {line.strip()}")
            else:
                print(f"Skipping line due to incorrect format: {line.strip()}")

    # Convert the list of coordinates to a NumPy array
    node_matrix = np.array(node_coordinates)

    print("Node coordinates successfully loaded into a numpy matrix.")
    print(node_matrix)

except FileNotFoundError:
    print(f"Error: The file '{nodes_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

springs_file_path = 'springs.txt'
index_info = []
stiffness_info = []

try:
    with open(springs_file_path, 'r') as f:
        for line in f:
            # Split each line by comma and remove leading/trailing whitespace
            parts = [part.strip() for part in line.split(',')]
            # Assuming the format is spring number, first node, second node, stiffness
            if len(parts) == 3:
                try:
                    first_node_index = int(parts[0])
                    second_node_index = int(parts[1])
                    stiffness = float(parts[2])
                    index_info.append([2*first_node_index, 2*first_node_index+1, 2*second_node_index, 2*second_node_index+1])
                    stiffness_info.append(stiffness)
                except ValueError:
                    print(f"Skipping line due to non-numeric coordinates: {line.strip()}")
            else:
                print(f"Skipping line due to incorrect format: {line.strip()}")

    # Convert the list of coordinates to a NumPy array
    index_matrix = np.array(index_info)
    stiffness_matrix = np.array(stiffness_info)

    print("Spring indices successfully loaded into a numpy matrix.")
    print(index_matrix)

    print("Spring stiffnesses successfully loaded into a numpy matrix.")
    print(stiffness_matrix)

except FileNotFoundError:
    print(f"Error: The file '{springs_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

N = node_matrix.shape[0] # Number of nodes
ndof = 2 * N # Number of degrees of freedom

# Initialize my positions and velocities (x_old and u_old; potentially a_old if using Newmark-Beta)
x_old = np.zeros(ndof)
u_old = np.zeros(ndof) # No need to update

# Build x_old vector from the nodes.txt file (node_matrix)
for i in range(N):
  x_old[2 * i] = node_matrix[i][0] # x coordinate
  x_old[2 * i + 1] = node_matrix[i][1] # y coordinate

# Every spring has a rest length
l_k = np.zeros_like(stiffness_matrix)
for i in range(stiffness_matrix.shape[0]):
    ind = index_matrix[i].astype(int)
    xi = x_old[ind[0]]
    yi = x_old[ind[1]]
    xj = x_old[ind[2]]
    yj = x_old[ind[3]]
    l_k[i] = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)

# Mass
m = np.zeros(ndof)
for i in range(ndof):
  m[i] = 1.0 # In this problem, every point mass is 1 kg

dt = 0.1
maxTime = 100
t = np.arange(0, maxTime + dt, dt)

# exact times to show
plot_times = np.array([0.0, 0.1, 1.0, 10.0, 100.0])

# free DOFs
#free_DOF = np.arange(2, ndof-2)
free_DOF = np.array([2,3,6,7])
print(free_DOF)

# store middle-node y
y_middle = np.zeros(len(t))
y_middle[0] = x_old[3]

# store the y coordinate of node 1, and 3
y_node1 = np.zeros(len(t))
y_node1[0] = x_old[3]

y_node3 = np.zeros(len(t))
y_node3[0] = x_old[7]

# ---- plot t=0 BEFORE the loop ----
plot(x_old, index_matrix, t[0])

for k in range(len(t)-1):
    t_new = t[k+1]

    # step integrator
    x_new, u_new = myInt(t_new, x_old, u_old, free_DOF, stiffness_matrix, index_matrix, m, dt, l_k)

    #x_new, u_new = myInt_explicit(t_new, x_old, u_old, free_DOF, stiffness_matrix, index_matrix, m, dt, l_k)

    # plot only at desired times (0.1, 1, 10, 100)
    if np.isclose(t_new, plot_times, atol=1e-12).any():
        plot(x_new, index_matrix, t_new)

    y_middle[k+1] = x_new[3]
    y_node1[k+1] = x_new[3]
    y_node3[k+1] = x_new[7]
    x_old = x_new
    u_old = u_new

# final time history (this one can be blocking)
#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(t, y_middle, 'ro-')
#plt.xlabel('Time [second]')
#plt.ylabel('y-coordinate of the second node [meter]')
#plt.show()


import matplotlib.pyplot as plt

plt.figure()
plt.plot(t, y_node1, 'bo-', label='Node 1')   # blue line with circles
plt.plot(t, y_node3, 'go-', label='Node 3')   # green line with circles
plt.xlabel('Time [second]')
plt.ylabel('y-coordinate [meter]')
plt.title('Y-coordinates of Node 1 and Node 3 Over Time')
plt.legend()  # shows which line is which
plt.show()