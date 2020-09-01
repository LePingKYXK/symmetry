#!/usr/bin/env python3


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def visualization(CoM_coord, eig_vec, new_coord):
    """ This function visualizes the moment of inertia as vectors and
    molecule (using the shifted coordinates) as points.
    
    ========================
    parameters:
    CoM_coord:    the coordinates of the center of mass
    eig_vec:      the eigen vectors
    new_coord:    the shifted coordinates of molecule

    ------------------------
        ax.quiver method draws the 3D vectors, the 6 arguments in this
    method are X,Y,Z for the starting point and U, V, W for the ending
    point.
        ax.scatter method draws the scatter points in 3D space.
    """
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    vlen = np.linalg.norm(eig_vec)
    X, Y, Z = CoM_coord
    U, V, W = eig_vec
    ax.quiver(X, Y, Z, U, V, W, pivot='tail',
              length=vlen, arrow_length_ratio=0.2/vlen)

    ax.scatter(new_coord[:,0], new_coord[:,1], new_coord[:,2],
               color="r", marker="o", s=50)
    
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.75, 0.75, 1, 1]))
    ax.set_xlim([new_coord.min(),new_coord.max()])
    ax.set_ylim([new_coord.min(),new_coord.max()])
    ax.set_zlim([new_coord.min(),new_coord.max()])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()



if __name__ == "__main__":
    visualization(CoM_coord, eig_vec, new_coord)
