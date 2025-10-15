import numpy as np
import matplotlib.pyplot as plt

def plot(x, index_matrix, t):
    plt.figure()
    plt.title(f'Time: {t:.2f} second, dt = 10^-6')
    for i in range(index_matrix.shape[0]):  # All springs
        ind = index_matrix[i].astype(int)
        xi = x[ind[0]]; yi = x[ind[1]]
        xj = x[ind[2]]; yj = x[ind[3]]
        plt.plot([xi, xj], [yi, yj], 'bo-')
    plt.axis('equal')
    plt.xlabel('x [meter]')
    plt.ylabel('y [meter]')
    # Non-blocking so your loop keeps running
    plt.show()