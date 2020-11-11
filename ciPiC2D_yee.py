#
# author:         G. Lapenta, L. Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# date:           01.11.2020
# copyright:      2020 KU Leuven (c)
# MIT license
#

#
# Fully Implicit Particle in Cell 2D
#       • General coordinates
#       • Charge conserving
#       • Energy conserving
#       • Mimetic operator
#

import numpy as np
from scipy.optimize import newton_krylov
from numpy import cosh, zeros_like, mgrid, zeros
import matplotlib.pyplot as plt

# parameters
nx, ny = 75, 75
Lx, Ly = 1.,1.
dx, dy = Lx/(nx-1), Ly/(ny-1)
dt = 0.1
nt= 4
npart = 1000

P_left, P_right = 0, 0
P_top, P_bottom = 0, 0
xg, yg = mgrid[0:Lx:(nx*1j), 0:Ly:(ny*1j)]

WP = 1. # Plasma frequency
QM = -1. # Charge/mass ratio
V0x = 0 # Stream velocity
V0y = 0 # Stream velocity
VT = .1 # thermal velocity
# Time steps
Nt = nt
# Nodes
Nx1 = nx
Nx2 = ny
Nx3 = 3 # two layers are used for the BC

# Computational domain 
Lx3 = 0.25
dx1 = dx
dx2 = dy
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
# Field matrix
Ex1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Ex2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Ex3 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx3 = np.zeros([Nx1, Nx2, Nx3], dtype=float) 
# Field matrix (old)
Bx1_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx2_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
Bx3_old = np.zeros([Nx1, Nx2, Nx3], dtype=float)
# Curl of fields
curl_Ex1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
curl_Ex2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
curl_Ex3 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
curl_Bx1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
curl_Bx2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
curl_Bx3 = np.zeros([Nx1, Nx2, Nx3], dtype=float)

#Perturbation
Bx3[int((Nx1-1)/2), int((Nx2-1)/2), :] = 1.
# Total energy
U = np.zeros(Nt, dtype=float) 
# Divergence of E
divB = np.zeros(Nt, dtype=float) 

# Grid begin & end point
# Note we don't start from 0 cause 0 and Nx1-1 are the same node
ib = 1
ie = Nx1 - 1
jb = 1
je = Nx2 - 1
kb = 1
ke = Nx3 - 1

def curl(Ax1, Ax2, Ax3, s):   
    curl_x1 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
    curl_x2 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
    curl_x3 = np.zeros([Nx1, Nx2, Nx3], dtype=float)
    if s == 0:
        curl_x1[ib:ie, jb:je, kb:ke] =\
             (gx3x1[ib+1:ie+1, jb:je, kb:ke]*Ax1[ib+1:ie+1, jb:je, kb:ke] - gx3x1[ib+1:ie+1, jb-1:je-1, kb:ke]*Ax1[ib+1:ie+1, jb-1:je-1, kb:ke]\
        +     gx3x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]         - gx3x1[ib:ie, jb-1:je-1, kb:ke]*Ax1[ib:ie, jb-1:je-1, kb:ke]\
        +     gx3x2[ib:ie, jb+1:ie+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke] - gx3x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke]\
        + 2.*(gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]         - gx3x3[ib:ie, jb-1:je-1, kb:ke]*Ax3[ib:ie, jb-1:je-1, kb:ke]))/(2.*dx2*J[ib:ie, jb:je, kb:ke])\
        -    (gx2x1[ib+1:ie+1, jb:je, kb:ke]*Ax1[ib+1:ie+1, jb:je, kb:ke] - gx2x1[ib+1:ie+1, jb:je, kb-1:ke-1]*Ax1[ib+1:ie+1, jb:je, kb-1:ke-1]\
        +     gx2x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]         - gx2x1[ib:ie, jb:je, kb-1:ke-1]*Ax1[ib:ie, jb:je, kb-1:ke-1]\
        + 2.*(gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]         - gx2x2[ib:ie, jb:je, kb-1:ke-1]*Ax2[ib:ie, jb:je, kb-1:ke-1])\
        +     gx2x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1] - gx2x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1])/(2.*dx3*J[ib:ie, jb:je, kb:ke])
        curl_x2[ib:ie, jb:je, kb:ke] =\
        (2.*(gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]          - gx1x1[ib:ie, jb:je, kb-1:ke-1]*Ax1[ib:ie, jb:je, kb-1:ke-1])\
        +    gx1x2[ib:ie, jb+1:je+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke]  - gx1x2[ib:ie, jb+1:je+1, kb-1:ke-1]*Ax2[ib:ie, jb+1:je+1, kb-1:ke-1]\
        +    gx1x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]          - gx1x2[ib:ie, jb:je, kb-1:ke-1]*Ax2[ib:ie, jb:je, kb-1:ke-1]\
        +    gx1x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1]  - gx1x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1])/(2.*dx3*J[ib:ie, jb:je, kb:ke])\
        -   (gx3x1[ib+1:ie+1, jb:je, kb:ke]*Ax1[ib+1:ie+1, jb:je, kb:ke]  - gx3x1[ib-1:ie-1, jb:je, kb:ke]*Ax1[ib-1:ie-1, jb:je, kb:ke]\
        +    gx3x2[ib:ie, jb+1:je+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke]  - gx3x2[ib-1:ie-1, jb+1:je+1, kb:ke]*Ax2[ib-1:ie-1, jb+1:je+1, kb:ke]\
        +    gx3x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]          - gx3x2[ib-1:ie-1, jb:je, kb:ke]*Ax2[ib-1:ie-1, jb:je, kb:ke]\
        +2.*(gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]          - gx3x3[ib-1:ie-1, jb:je, kb:ke]*Ax3[ib-1:ie-1, jb:je, kb:ke]))/(2.*dx1*J[ib:ie, jb:je, kb:ke])
        curl_x3[ib:ie, jb:je, kb:ke] =\
             (gx2x1[ib+1:ie+1, jb:je, kb:ke]*Ax1[ib+1:ie+1, jb:je, kb:ke] - gx2x1[ib-1:ie-1, jb:je, ib:ke]*Ax1[ib-1:ie-1, jb:je, kb:ke]\
        + 2.*(gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]         - gx2x2[ib-1:ie-1, jb:je, kb:ke]*Ax2[ib-1:ie-1, jb:je, kb:ke])\
        +     gx2x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1] - gx2x3[ib-1:ie-1, jb:je, kb+1:ke+1]*Ax3[ib-1:ie-1, jb:je, kb+1:ke+1]\
        +     gx2x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]         - gx2x3[ib-1:ie-1, jb:je, kb:ke]*Ax3[ib-1:ie-1, jb:je, kb:ke])/(2.*dx1*J[ib:ie, jb:je, kb:ke])\
        -(2.*(gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]         - gx1x1[ib:ie, jb-1:je-1, kb:ke]*Ax1[ib:ie, jb-1:je-1, kb:ke])\
        +     gx1x2[ib:ie, jb+1:je+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke] - gx1x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke]\
        +     gx1x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1] - gx1x3[ib:ie, jb-1:je-1, kb+1:ke+1]*Ax3[ib:ie, jb-1:je-1, kb+1:ke+1]\
        +     gx1x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]         - gx1x3[ib:ie, jb-1:je-1, kb:ke]*Ax3[ib:ie, jb-1:je-1, kb:ke])/(2.*dx2*J[ib:ie, jb:je, kb:ke])
    elif s == 1:
        curl_x1[ib:ie, jb:je, kb:ke] =\
             (gx3x1[ib:ie, jb+1:je+1, kb:ke]*Ax1[ib:ie, jb+1:je+1, kb:ke]         - gx3x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]\
        +     gx3x1[ib-1:ie-1, jb+1:je+1, kb:ke]*Ax1[ib-1:ie-1, jb+1:je+1, kb:ke] - gx3x1[ib-1:ie-1, jb:je, kb:ke]*Ax1[ib-1:ie-1, jb:je, kb:ke]\
        +     gx3x2[ib:ie, jb+1:je+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke]         - gx3x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke]\
        + 2.*(gx3x3[ib:ie, jb+1:je+1, kb:ke]*Ax3[ib:ie, jb+1:je+1, kb:ke]         - gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]))/(2.*dx2*J[ib:ie, jb:je, kb:ke])\
        -    (gx2x1[ib:ie, jb:je, kb+1:ke+1]*Ax1[ib:ie, jb:je, kb+1:ke+1]         - gx2x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke]\
        +     gx2x1[ib-1:ie-1, jb:je, kb+1:ke+1]*Ax1[ib-1:ie-1, jb:je, kb+1:ke+1] - gx2x1[ib-1:ie-1, jb:je, kb:ke]*Ax1[ib-1:ie-1, jb:je, kb:ke]\
        + 2.*(gx2x2[ib:ie, jb:je, kb+1:ke+1]*Ax2[ib:ie, jb:je, kb+1:ke+1]         - gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke])\
        +     gx2x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1]         - gx2x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1])/(2.*dx3*J[ib:ie, jb:je, kb:ke])
        curl_x2[ib:ie, jb:je, kb:ke] =\
         (2.*(gx1x1[ib:ie, jb:je, kb+1:ke+1]*Ax1[ib:ie, jb:je, kb+1:ke+1]         - gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke])\
        +     gx1x2[ib:ie, jb:je, kb+1:ke+1]*Ax2[ib:ie, jb:je, kb+1:ke+1]         - gx1x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]\
        +     gx1x2[ib:ie, jb-1:je-1, kb+1:ke+1]*Ax2[ib:ie, jb-1:je-1, kb+1:ke+1] - gx1x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke]\
        +     gx1x3[ib:ie, jb:je, kb+1:ke+1]*Ax3[ib:ie, jb:je, kb+1:ke+1]         - gx1x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1])/(2.*dx3*J[ib:ie, jb:je, kb:ke])\
        -    (gx3x1[ib+1:ie+1, jb:je, kb:ke]*Ax1[ib+1:ie+1, jb:je, kb:ke]         - gx3x1[ib-1:ie-1, jb:je, kb:ke]*Ax1[ib-1:ie-1, jb:je, kb:ke]\
        +     gx3x2[ib+1:ie+1, jb:je, kb:ke]*Ax2[ib+1:ie+1, jb:je, kb:ke]         - gx3x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke]\
        +     gx3x2[ib+1:ie+1, jb-1:je-1, kb:ke]*Ax2[ib+1:ie+1, jb-1:je-1, kb:ke] - gx3x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke]\
        + 2.*(gx3x3[ib+1:ie+1, jb:je, kb:ke]*Ax3[ib+1:ie+1, jb:je, kb:ke]         - gx3x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]))/(2.*dx1*J[ib:ie, jb:je, kb:ke])
        curl_x3[ib:ie, jb:je, kb:ke] =\
             (gx2x1[ib+1:ie+1, jb:je, kb:ke]*Ax1[ib+1:ie+1, jb:je, kb:ke]         - gx2x1[ib-1:ie-1, jb:je, kb:ke]*Ax1[ib-1:ie-1, jb:je, kb:ke]\
        + 2.*(gx2x2[ib+1:ie+1, jb:je, kb:ke]*Ax2[ib+1:ie+1, jb:je, kb:ke]         - gx2x2[ib:ie, jb:je, kb:ke]*Ax2[ib:ie, jb:je, kb:ke])\
        +     gx2x3[ib+1:ie+1, jb:je, kb:ke]*Ax3[ib+1:ie+1, jb:je, kb:ke]         - gx2x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]\
        +     gx2x3[ib+1:ie+1, jb:je, kb-1:ke-1]*Ax3[ib+1:ie+1, jb:je, kb-1:ke-1] - gx2x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1])/(2.*dx1*J[ib:ie, jb:je, kb:ke])\
        -(2.*(gx1x1[ib:ie, jb+1:je+1, kb:ke]*Ax1[ib:ie, jb+1:je+1, kb:ke]         - gx1x1[ib:ie, jb:je, kb:ke]*Ax1[ib:ie, jb:je, kb:ke])\
        +     gx1x2[ib:ie, jb+1:je+1, kb:ke]*Ax2[ib:ie, jb+1:je+1, kb:ke]         - gx1x2[ib:ie, jb-1:je-1, kb:ke]*Ax2[ib:ie, jb-1:je-1, kb:ke]\
        +     gx1x3[ib:ie, jb+1:je+1, kb:ke]*Ax3[ib:ie, jb+1:je+1, kb:ke]         - gx1x3[ib:ie, jb:je, kb:ke]*Ax3[ib:ie, jb:je, kb:ke]\
        +     gx1x3[ib:ie, jb+1:je+1, kb-1:ke-1]*Ax3[ib:ie, jb+1:je+1, kb-1:ke-1] - gx1x3[ib:ie, jb:je, kb-1:ke-1]*Ax3[ib:ie, jb:je, kb-1:ke-1])/(2.*dx2*J[ib:ie, jb:je, kb:ke])
    return curl_x1, curl_x2, curl_x3

def periodicBC(A):
    A_w = np.zeros([Nx1, Nx2, Nx3], dtype=float)
    A_s = np.zeros([Nx1, Nx2, Nx3], dtype=float)
    A_b = np.zeros([Nx1, Nx2, Nx3], dtype=float)
    # swop var.
    A_w[0, :, :] = A[0, :, :]
    A_s[:, 0, :] = A[:, 0, :]
    A_b[:, :, 0] = A[:, :, 0]
    # Reflective BC for A field
    A[0, :, :]  = A[-1, :, :]   # west = est
    A[-1, :, :] = A_w[0, :, :]  # est = west
    A[:, 0, :]  = A[:, -1, :]   # south = north
    A[:, -1, :] = A_s[:, 0, :]  # north = south
    A[:, :, 0]  = A[:, :, -1]   # bottom = top
    A[:, :, -1] = A_b[:, :, 0]  # top = bottom  

def phys_to_krylov(phi,u,v):
    ''' To populate the Krylov vector using physiscs vectors
    Phi is a 2D array of dimension (nx,ny)
    u of dimensions npart
    v of dimension npart
    '''
    ykrylov = zeros(nx*ny+2*npart)
    ykrylov[0:nx*ny] = phi.reshape(nx*ny)
    ykrylov[nx*ny:nx*ny+npart] = u
    ykrylov[nx*ny+npart:] = v
    return ykrylov

def krylov_to_phys(xkrylov):
    ''' To populate the physiscs vectors using the Krylov space vector
    Phi is a 2D array of dimension (nx,ny)
    ubar of dimensions npart
    vbar of dimension npart
    '''
    P = np.reshape(xkrylov[0:nx*ny],(nx,ny))
#    ubar = zeros(npart,float)
#    vbar = zeros(npart,float)
    ubar = xkrylov[nx*ny:nx*ny+npart]
    vbar = xkrylov[nx*ny+npart:]
    return P, ubar, vbar

def e_field(P):
    ''' To compute the electric field from the potential
    '''
    Ex = zeros((nx, ny), float)
    Ey = zeros((nx, ny), float)
    Ex[1:-1] = (P[2:] - P[:-2])/dx/2
    Ex[0] = (P[1] - P[-1])/dx/2
    Ex[-1] =(P[0] - P[-2])/dx/2
    Ey[:,1:-1] = (P[:,2:] -  P[:,:-2])/dy/2
    Ey[:,0]    = (P[:,1]  -  P[:,-1])/dy/2
    Ey[:,-1]   = (P[:,0]  -  P[:,-2])/dy/2
    Ex = -Ex
    Ey = -Ey
    return Ex, Ey

def laplacian(P):
    ''' To compute the Laplacian in uniform grid
    '''
    d2x = zeros((nx, ny), float)
    d2y = zeros((nx, ny), float)
    d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) / dx/dx
    #d2x[0]    = (P[1]    - 2*P[0]    + P_left)/dx/dx
    d2x[0]    = (P[1] - 2*P[0]    + P[-1])/dx/dx 
    #d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/dx/dx
    d2x[-1]   = (P[0] - 2*P[-1]   + P[-2])/dx/dx
    

    d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/dy/dy
    #d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/dy/dy
    #d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/dy/dy
    d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P[:,-1])/dy/dy
    d2y[:,-1]   = (P[:,0]  - 2*P[:,-1]   + P[:,-2])/dy/dy
    return d2x + d2y


def residual(xkrylov):
    ''' Calculation of the residual of the equations
    This is the most important part: the definion of the problem
    '''
    global Ex1, Ex2, Ex3, Bx1, Bx2, Bx3
    P, ubar, vbar = krylov_to_phys(xkrylov)
    
    xbar = x + ubar*dt/2
    ybar = y + vbar*dt/2
    xbar = xbar%Lx
    ybar = ybar%Ly
    
    source = particle_to_grid(xbar,ybar,q)
    source = source - np.mean(source) 
    
    lap = laplacian(P)

    res = lap + source
    res[0,0] = P[0,0]
    
    #Ex, Ey = e_field(P)
    Bx1_old[:, :, :] = Bx1[:, :, :]
    Bx2_old[:, :, :] = Bx2[:, :, :]
    Bx3_old[:, :, :] = Bx2[:, :, :]

    curl_Bx1, curl_Bx2, curl_Bx3 = curl(Bx1, Bx2, Bx3, 0)
    Ex1 += dt*curl_Bx1
    Ex2 += dt*curl_Bx2
    Ex3 += dt*curl_Bx3
    periodicBC(Ex1)
    periodicBC(Ex2)
    periodicBC(Ex3)

    curl_Ex1, curl_Ex2, curl_Ex3 = curl(Ex1, Ex2, Ex3, 1)
    Bx1 -= dt*curl_Ex1
    Bx2 -= dt*curl_Ex2
    Bx3 -= dt*curl_Ex3
    periodicBC(Bx1)
    periodicBC(Bx2)
    periodicBC(Bx3)

    Ex, Ey = Ex1, Ex2

    Exp = grid_to_particle(xbar,ybar,Ex)
    Eyp = grid_to_particle(xbar,ybar,Ey)
    
    resu = ubar - u - QM* Exp *dt/2
    resv = vbar - v - QM* Eyp *dt/2
    
    ykrylov = phys_to_krylov(res,resu,resv)
    return  ykrylov

def grid_to_particle(x,y,E):  
    ''' Interpolation grid to particle
    '''
    global dx, dy, nx, ny, npart
    
    Ep = zeros_like(x)

    for i in range(npart):

      #  interpolate field Ex from grid to particle */
      xa = x[i]/dx
      ya = y[i]/dy
      i1 = int(xa)
      i2 = i1 + 1
      j1 = int(ya)
      j2 = j1 + 1      
      wx2 = xa - i1
      wx1 = 1.0 - wx2
      wy2 = ya - j1
      wy1 = 1.0 - wy2
      Ep[i] = wx1* wy1 * E[i1,j1] + wx2* wy1 * E[i2,j1] + wx1* wy2 * E[i1,j2] + wx2* wy2 * E[i2,j2]  
    
    return Ep

def particle_to_grid(x,y,q): 
    ''' Interpolation particle to grid
    ''' 
    global dx, dy, nx, ny, npart
    
    r =  zeros((nx, ny), float)
    for i in range(npart):

      #  interpolate field Ex from grid to particle */
      xa = x[i]/dx
      ya = y[i]/dy
      i1 = int(xa)
      i2 = i1 + 1
      if(i2==nx):
          i2=0 
      j1 = int(ya)
      j2 = j1 + 1  
      if(j2==ny):
          j2=0    
      wx2 = xa - i1
      wx1 = 1.0 - wx2
      wy2 = ya - j1
      wy1 = 1.0 - wy2
      r[i1,j1] += wx1* wy1 * q[i] 
      r[i2,j1] += wx2* wy1 * q[i] 
      r[i1,j2] += wx1* wy2 * q[i] 
      r[i2,j2] += wx2* wy2 * q[i] 
    
    return r

# initialize particles
x = Lx*np.random.rand(npart)
y = Ly*np.random.rand(npart)
u = V0x+VT*np.random.randn(npart)
v = V0y+VT*np.random.randn(npart)
q = np.ones(npart) * WP**2 / (QM*npart/Lx/Ly) 

# main cycle
histEnergy=[]
for it in range(nt):
    plt.clf()
    guess = zeros(nx* ny+2*npart, float)
    sol = newton_krylov(residual, guess, method='lgmres', verbose=1)
    print('Residual: %g' % abs(residual(sol)).max())
    phi, ubar, vbar = krylov_to_phys(sol)
    u = 2*ubar - u
    v = 2*vbar - v
    x += ubar*dt
    y += vbar*dt
    x = x%Lx
    y = y%Ly

    #Ex, Ey = e_field(phi)
    Bx1_old[:, :, :] = Bx1[:, :, :]
    Bx2_old[:, :, :] = Bx2[:, :, :]
    Bx3_old[:, :, :] = Bx2[:, :, :]

    curl_Bx1, curl_Bx2, curl_Bx3 = curl(Bx1, Bx2, Bx3, 0)
    Ex1 += dt*curl_Bx1
    Ex2 += dt*curl_Bx2
    Ex3 += dt*curl_Bx3
    periodicBC(Ex1)
    periodicBC(Ex2)
    periodicBC(Ex3)

    curl_Ex1, curl_Ex2, curl_Ex3 = curl(Ex1, Ex2, Ex3, 1)
    Bx1 -= dt*curl_Ex1
    Bx2 -= dt*curl_Ex2
    Bx3 -= dt*curl_Ex3
    periodicBC(Bx1)
    periodicBC(Bx2)
    periodicBC(Bx3)

    Ex, Ey = Ex1, Ex2
    print('Ex, Ey ='+ Ex +','+ Ey)

    energy = np.sum(u**2) + np.sum(v**2) + np.sum(Ex**2) + np.sum(Ey**2)
    histEnergy.append(energy)
    print('energy =', energy)
    plt.subplot(2, 2, 1)
    plt.pcolor(xg, yg, phi)
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.plot(x,y,'.')
    plt.subplot(2,2,3)
    plt.plot(histEnergy)
    plt.subplot(2,2,4)
    plt.plot((histEnergy-histEnergy[0])/histEnergy[0])
    #plt.show()
    plt.pause(0.000000000000001)
    
# visualize
