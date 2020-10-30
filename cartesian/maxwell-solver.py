#
# author:         Luca Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# created:        28.10.2020
# last modified:  31.10.2020
# MIT license
#

#
# Maxwell Solver 3D
#       • General coordinates
#       • Energy conserving
#       • Mimetic operator
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

# Grid beguin & end point
# Note we don't start from 0 cause 0 and Nx1-1 are the same node
ib = 1
ie = Nx1 - 1
jb = 1
je = Nx1 - 1
kb = 1
ke = Nx1 - 1


def myplot(values, name):
    plt.figure(name)
    plt.imshow(values.T, origin='lower', extent=[0, Lx, 0, Ly], aspect='equal', vmin=-0.1, vmax=0.1, cmap='plasma')
    plt.colorbar()

def myplot2(values, name):
    plt.figure(name)
    plt.plot(values)

def curl(Ax, Ay, Az, s):
    '''
    s = 0 -> Forward deriv
    s = 1 -> Backward derivative
    '''
    if s = 0:
        rx = (gx3x1[ib+1:ie+1, jb:je, kb:ke]*Ax1[ib+1:ie+1, jb:je, kb:ke] - gx3x1[ib+1:ie+1, jb-1:je-1, kb:ke]*Ax1[ib+1:ie+1, jb-1:je-1, kb:ke]\
           +  gx3x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]         - gx3x1[ib:ie, jb-1:je-1, kb:ke]*Ax1[ib:ie, jb-1:je-1, kb:ke]\
           +  gx3x2[ib:ie, jb+1:ie+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke] - gx3x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke]\
           +  gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]         - gx3x3[ib:ie, jb-1:je-1, kb:ke]*Ax3[ib:ie, jb-1:je-1, kb:ke]\
           +  gx3x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1] - gx3x3[ib:ie, jb-1:je-1, kb-1:ke-1]*Ax3[ib:ie, jb-1:je-1, kb-1:ke-1])/(2.*dx2*J[:,:,:])
           - (gx2x3[ib+1:ie+1, jb:je, kb:ke]*Ax3[ib+1:ie+1, jb:je, kb:ke] - gx2x3[ib+1:ie+1, jb:je, kb-1:ke-1]*Ax3[ib+1:ie+1, jb:ij, kb-1:ke-1]\
           +  gx2x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]         - gx2x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1]\
           +  gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]         - gx2x2[ib:ie, jb:je, kb-1:ke-1]*Ax2[ib:ie, jb:je, kb-1:ke-1]\
           +  gx2x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke] - gx2x2[ib:ie, jb-1:je-1, kb-1:ke-1]*Ax2[ib:ie, jb-1:je-1, kb-1:ke-1]\
           +  gx2x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1] - gx2x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1])/(2.*dx3*J[:,:,:])

        ry = (gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]         - gx1x1[ib:ie, jb:je, kb-1:ke-1]*Ax1[ib:ie, jb:je, kb-1:ke-1]\
           +  gx1x1[ib-1:ie-1, jb:je, kb:ke]*Ax1[ib-1:ie-1, jb:je, kb:ke] - gx1x1[ib-1:ie-1, jb:je, kb-1:ke-1]*Ax1[ib-1:ie-1, jb:je, kb-1:ke-1]\
           +  gx1x2[ib:ie, jb+1:je+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke] - gx1x2[ib:ie, jb+1:je+1, kb-1:ke-1]*Ax2[ib:ie, jb+1:je+1, kb-1:ke-1]\
           +  gx1x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]         - gx1x3[ib:ie, jb:je, kb-1:ke-1]*Ax2[ib:ie, jb:je, kb-1:ke-1]\
           +  gx1x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1] - gx1x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1])/(2.*dx3*J[:,:,:])
           - (gx3x1[ib+1:ie+1, jb:je, kb:ke]*Ax1[ib+1:ie+1, jb:je, kb:ke] - gx3x1[ib-1:ie-1, jb:je, kb:ke]*Ax3[ib-1:ie-1, jb:je, kb:ke]\
           +  gx3x2[ib:ie, jb+1:je+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke] - gx3x3[ib-1:ie-1, jb+1:je+1, kb:ke]*Ax2[ib-1:ie-1, jb+1:je+1, kb:ke]\
           +  gx3x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]         - gx3x2[ib-1:ie-1, jb:je, kb:ke]*Ax2[ib-1:ie-1, jb:je, kb:ke]\
           +  gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]         - gx3x3[ib-1:ie-1, jb:je, kb:ke]*Ax3[ib-1:ie-1, jb:je, kb:ke]\
           +  gx3x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1] - gx3x3[ib-1:ie-1, jb:je, kb-1:ke-1]*Ax3[ib-1:ie-1, jb:je, kb-1:ke-1])/(2.*dx1*J[:,:,:])

        rz = (gx2x1[ib+1:ie+1, jb:je, kb:ke]*Ax1[ib+1:ie+1, jb:je, kb:ke] - gx2x1[ib-1:ie-1, jb:je, ib:ke]*Ax1[ib-1:ie-1, jb:je, kb:ke]\
           +  gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]         - gx2x2[ib-1:ie-1, jb:je, kb:ke]*Ax2[ib-1:ie-1, jb:je, kb:ke]\
           +  gx2x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke] - gx2x2[ib-1:ie-1, jb-1:je-1, kb:ke]*Ax2[ib-1:ie-1, jb-1:je-1, kb:ke]\
           +  gx2x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1] - gx2x3[ib-1:ie-1, jb:je, kb+1:ke+1]*Ax3[ib-1:ie-1, jb:je, kb+1:ke+1]\
           +  gx2x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]         - gx2x3[ib-1:ie-1, jb:je, kb:ke]*Ax3[ib-1:ie-1, jb:je, kb:ke])/(2.*dx1*J[:,:,:])
           - (gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]         - gx1x1[ib:ie, jb-1:je-1, kb:ke]*Ax1[ib:ie, jb-1:je-1, kb:ke]\
           +  gx1x1[ib-1:ie-1, jb:je, kb:ke]*Ax1[ib-1:ie-1, jb:je, kb:ke] - gx1x1[ib-1:ie-1, jb-1:je-1, kb:ke]*Ax1[ib-1:ie-1, jb-1:je-1, kb:ke]\
           +  gx1x2[ib:ie, jb+1:je+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke] - gx1x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke]\
           +  gx1x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1] - gx1x3[ib:ie, jb-1:je-1, kb+1:ke+1]*Ax3[ib:ie, jb-1:je-1, kb+1:ke+1]\
           +  gx1x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]         - gx1x3[ib:ie, jb-1:je-1, kb:ke]*Ax3[ib:ie, jb-1:je-1, kb:ke])/(2.*dx2*J[:,:,:])
    else:
        rx = (gx3x1[ib:ie, jb+1:je+1, kb:ke]*Ax1[ib:ie, jb+1:je+1, kb:ke]         - gx3x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]\
           +  gx3x1[ib-1:ie-1, jb+1:je+1, kb:ke]*Ax1[ib-1:ie-1, jb+1:je+1, kb:ke] - gx3x1[ib-1:ie-1, jb:je, kb:ke]*Ax1[ib-1:ie-1, jb:je, kb:ke]\
           +  gx3x2[ib:ie, jb+1:je+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke]         - gx3x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke]\
           +  gx3x3[ib:ie, jb+1:je+1, kb+1:ke+1]*Ax3[ib:ie, jb+1:je+1, kb+1:ke+1] - gx3x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1]\
           +  gx3x3[ib:ie, jb+1:je+1, kb:ke]*Ax3[ib:ie, jb+1:je+1, kb:ke]         - gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke])/(2.*dx2*J[:,:,:])
           - (gx2x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1]         - gx2x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]\
           +  gx2x3[ib-1:ie-1, jb:je, kb+1:ke+1]*Ax3[ib-1:ie-1, jb:je, kb+1:ke+1] - gx2x3[ib-1:ie-1, jb:je, kb:ke]*Ax3[ib-1:ie-1, jb:je, kb:ke]\
           +  gx2x2[ib:ie, jb+1:je+1, kb+1:ke+1]*Ax2[ib:ie, jb+1:je+1, kb+1:ke+1] - gx2x2[ib:ie, jb+1:je+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke]\
           +  gx2x2[ib:ie, jb:je, kb+1:ke+1]*Ax2[ib:ie, jb:je, kb+1:ke+1]         - gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]\
           +  gx2x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1]         - gx2x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke])/(2.*dx3*J[:,:,:])

        ry = (gx1x1[ib+1:ie+1, jb:je, kb+1:ke+1]*Ax1[ib+1:ie+1, jb:je, kb+1:ke+1] - gx1x1[ib+1:ie+1, jb:je, kb:ke]*Ax1[ib+1:ie+1, jb:je, kb:ke]\
           +  gx1x1[ib:ie, jb:je, kb+1:ke+1]*Ax1[ib:ie, jb:je, kb+1:ke+1] - gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]\
           +  gx1x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke] - gx1x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]\
           +  gx1x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke] - gx1x3[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]\
           +  gx1x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke] - gx1x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke])/(2.*dx3*J[:,:,:])
           - (gx3x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke] - gx3x1[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]\
           +  gx3x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke] - gx3x3[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]\
           +  gx3x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke] - gx3x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]\
           +  gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke] - gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]\
           +  gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke] - gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke])/(2.*dx1*J[:,:,:])

        rz = (gx2x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke] - gx2x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]\
           +  gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke] - gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]\
           +  gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke] - gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]\
           +  gx2x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke] - gx2x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]\
           +  gx2x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke] - gx2x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke])/(2.*dx1*J[:,:,:])
           - (gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke] - gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]\
           +  gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke] - gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]\
           +  gx1x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke] - gx1x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]\
           +  gx1x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke] - gx1x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]\
           +  gx1x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke] - gx1x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke])/(2.*dx2*J[:,:,:])
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
    '''
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
    '''

myplot(Ex1[:,:,0], 'Ex1')
myplot(Ex2[:,:,0], 'Ex2')
myplot(Bx3[:,:,0], 'Bx3')
myplot2(divB, 'divB vs t')
myplot2(U, 'U vs t')

plt.show()
