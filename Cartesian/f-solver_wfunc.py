#
# author:         Luca Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# created:        28.10.2020
# last modified:  28.10.2020
# MIT license
#

#
# Transverse Electromagnetics mode 2D 
#          • General coordinates
#          • Energy conserving
#          • Mimetic operator
#

import math
import numpy as np
import matplotlib.pyplot as plt
import time

# Time steps
Nt = 500 
dt = 0.001
# Nodes
Nx1 = 100
Nx2 = 100
Nx3 = 1

# Computational domain 
x1min, x1max = 0, 1    
x2min, x2max = 0, 1    
x3min, x3max = 0, .25  
Lx1 = int(abs(x1max - x1min))
Lx2 = int(abs(x2max - x2min)) 
Lx3 = abs(x2max - x2min)
dx1 = Lx1/Nx1 
dx2 = Lx2/Nx2
dx3 = Lx3/Nx3

# Geometry: Tensor and Jacobian
J = np.ones([Nx1, Nx2, Nx3], dtype=float) 
gx1x1 = np.ones([Nx1, Nx2, Nx3], dtype=float)
gx1x2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
gx1x3 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
gx2x1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
gx2x2 = np.ones([Nx1, Nx2, Nx3], dtype=float)
gx2x3 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
gx3x1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
gx3x2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
gx3x3 = np.ones([Nx1, Nx2, Nx3], dtype=float)
# Grid matrix
x1 = np.linspace(x1min, Lx1 - dx1, Nx1, dtype=float)
x2 = np.linspace(x2min, Lx2 - dx2, Nx2, dtype=float)
x3 = np.linspace(x3min, Lx3 - dx3, Nx3, dtype=float)
x1v, x2v, x3v = np.meshgrid(x1, x2, x3, indexing='ij')
# Field matrix
Ex1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Ex2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Ex3 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx2 = np.zeros([Nx1, Nx2, Nx3], dtype=float) 
# Field matrix (old)
Ex1_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Ex2_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Ex3_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx1_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx2_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx2_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
# Fields increments
dEx1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
dEx2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
dEx3 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
dBx1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
dBx2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
dBx3 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
# Fields increments (old)
dEx1_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
dEx2_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
dEx3_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
dBx1_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
dBx2_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
dBx3_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)

#Perturbation
Bx3[int((Nx1-1)/2), int((Nx2-1)/2), :] = 1.
# Total energy
U = np.zeros(Nt, dtype=float) 
# Divergence of E
divB = np.zeros(Nt, dtype=float) 


# Matrix xyz-position shifter with periodic boundary conditions
def shift(mat, x, y, z):
    result = np.roll(mat, -x, 0)
    result = np.roll(result, -y, 1)
    result = np.roll(result, -z, 2)
    return result

# Derivative in x1
def ddx1(A, s):
	'''
	s = 0 -> Forward deriv
	s = 1 -> Backward derivative
	'''
    return (shift(A, 1-s, 0, 0) - shift(A, 0-s, 0, 0))/dx1

# Derivative in x2
def ddx2(A, s):
	'''
	s = 0 -> Forward deriv
	s = 1 -> Backward derivative
	'''
    return (shift(A, 0, 1-s, 0) - shift(A, 0, 0-s, 0))/dx2

# Derivative in x3
def ddx3(A, s):
	'''
	s = 0 -> Forward deriv
	s = 1 -> Backward derivative
	'''
    return (shift(A, 0, 0, 1-s) - shift(A, 0, 0, 0-s))/dx3

# Curl operator
def curl(Ax, Ay, Az, s):
    rx = ddx2(Az, s) - ddx3(Ay, s)
    ry = ddx3(Ax, s) - ddx1(Az, s)
    rz = ddx1(Ay, s) - ddx2(Ax, s)
    return rx, ry, rz

# Divergence operator
def div(Ax, Ay, Az):
    return ddx1(Ax, 0) + ddx2(Ay, 0) + ddx3(Az, 0)


print('START TIME EVOLUTION...')
start = time.time()
for t in range(time_iter):
    Bx1_old[:, :, :] = Bx1[:, :, :]
    Bx2_old[:, :, :] = Bx2[:, :, :]
    Bx3_old[:, :, :] = Bx2[:, :, :]

    dEx1, dEx2, dEx3 = curl(Bx1, Bx2, Bx3, 0)

    Ex1 += dt*dEx1
    Ex2 += dt*dEx2
    Ex3 += dt*dEx3

    dBx1, dBx2, dBx3 = curl(Ex1, Ex2, Ex3, 1)

    Bx1 -= dt*dBx1
    Bx2 -= dt*dBx2
    Bx3 -= dt*dBx3

    U[t] = 0.5*np.sum(Ex1**2 + Ex2**2 + Ex3**2 \
    	             + Bx1*Bx1_old + Bx2*Bx2_old + Bx3*Bx3_old)
    divB[t] = np.sum(np.abs(div(Bx, By, Bz)))

    plt.subplot(2, 2, 1)
    plt.pcolor(x1v, x2v, Bx3)
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.plot(divB)
    plt.subplot(2, 2, 3)
    plt.plot(U)
    plt.subplot(2, 2, 4)
    plt.plot(U)
    plt.pause(0.000000000001)

stop = time.time()
print('DONE!')
print('time = %1.2f s' % (stop - start))


















