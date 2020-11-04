#
# author:         G. Lapenta, L. Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# date:           30.10.2020
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

x1min, x1max = 0, 1 
x2min, x2max = 0, 1 
Lx, Ly = int(abs(x1max - x1min)), int(abs(x2max - x2min)) 
Nx, Ny = 75, 75
dx, dy = Lx/(Nx-1), Ly/(Ny-1)
dt = 0.1
Nt = 10
npart = 1000

WP  =  1.   # Plasma frequency
QM  = -1.   # Charge to mass ratio
V0x =  0    # Stream velocity
V0y =  0    # Stream velocity
VT  =   .1  # Thermal velocity

P_left, P_right = 0, 0
P_top, P_bottom = 0, 0
xg, yg = mgrid[0:Lx:(Nx*1j), 0:Ly:(Ny*1j)]

#---------------------------- Field Solver -------------------------
Nx1, Nx2 = Nx, Ny  
dx1, dx2 = dx, dy 
# Grid generation
#x1 = np.linspace(x1min, x1max, Nx1, dtype=float)
#x2 = np.linspace(x2min, x2max, Nx2, dtype=float)
#x1v, x2v = np.meshgrid(x1, x2, indexing='ij')

# Initialization of field matrices
Ex1 = np.zeros([Nx1, Nx2], dtype=float)
Ex2 = np.zeros([Nx1, Nx2], dtype=float)
Bx3 = np.zeros([Nx1, Nx2], dtype=float)
Bx3old = np.zeros([Nx1, Nx2], dtype=float) # Bx3 at time n-1/2
Bx3[int((Nx1-1)/2),int((Nx2-1)/2)] = 0.1 # Initial perturbation

# Set Geometry
J = np.ones([Nx1, Nx2], dtype=float) 
gx1x1 = np.ones([Nx1, Nx2], dtype=float)
gx1x2 = np.zeros([Nx1, Nx2], dtype=float)
gx1x3 = np.zeros([Nx1, Nx2], dtype=float)
gx2x1 = np.zeros([Nx1, Nx2], dtype=float)
gx2x2 = np.ones([Nx1, Nx2], dtype=float)
gx2x3 = np.zeros([Nx1, Nx2], dtype=float)
gx3x1 = np.zeros([Nx1, Nx2], dtype=float)
gx3x2 = np.zeros([Nx1, Nx2], dtype=float)
gx3x3 = np.ones([Nx1, Nx2], dtype=float)

U = np.zeros([Nt], dtype=float) # Total energy
divE = np.zeros([Nt], dtype=float) # Divergence of E

# Start & End
ib = 1
jb = 1
ie = Nx1 - 1
je = Nx2 - 1

def periodicBC(A):
    A_w = np.zeros([Nx1, Nx2], dtype=float)
    A_s = np.zeros([Nx1, Nx2], dtype=float)
    A_b = np.zeros([Nx1, Nx2], dtype=float)
    # swop var.
    A_w[0, :] = A[0, :]
    A_s[:, 0] = A[:, 0]
    # Reflective BC for A field
    A[0, :]  = A[-1, :]   # west = est
    A[-1, :] = A_w[0, :]  # est = west
    A[:, 0]  = A[:, -1]   # south = north
    A[:, -1] = A_s[:, 0]  # north = south

def e_field(Ex1, Ex2, Bx3):
    Bx3old[:, :] = Bx3[:, :]
    Ex1[ib:ie, jb:je] += dt * (1./(dx2*J[ib:ie, jb:je]))\
                                * (gx3x3[ib:ie, jb+1:je+1]*Bx3[ib:ie, jb+1:je+1] - gx3x3[ib:ie, jb:je]*Bx3[ib:ie, jb:je])
    Ex2[ib:ie, jb:je] -= dt * (1./(dx1*J[ib:ie, jb:je]))\
                                * (gx3x3[ib+1:ie+1, jb:je]*Bx3[ib+1:ie+1, jb:je] - gx3x3[ib:ie, jb:je]*Bx3[ib:ie, jb:je])
    periodicBC(Ex1)
    periodicBC(Ex2) 
    Bx3[ib:ie, jb:je] -= dt * ((1./(2.*dx1*J[ib:ie, jb:je]))\
                                *    (gx2x1[ib+1:ie+1, jb:je]*Ex1[ib+1:ie+1, jb:je] - gx2x1[ib-1:ie-1, jb:je]*Ex1[ib-1:ie-1, jb:je]\
                                + 2.*(gx2x2[ib:ie, jb:je]*Ex2[ib:ie, jb:je] - gx2x2[ib-1:ie-1, jb:je]*Ex2[ib-1:ie-1, jb:je]))\
                                - (1./(2.*dx2*J[ib:ie, jb:je]))\
                                * (2.*(gx1x1[ib:ie, jb:je]*Ex1[ib:ie, jb:je] - gx1x1[ib:ie, jb-1:je-1]*Ex1[jb:je, ib-1:ie-1])\
                                +      gx1x2[ib:ie, jb+1:je+1]*Ex2[ib:ie, jb+1:je+1] - gx1x2[ib:ie, jb-1:je-1]*Ex2[ib:ie, jb-1:je-1]))
    periodicBC(Bx3)
    return Ex1, Ex2
    
#---------------------------- Field Solver -------------------------

def phys_to_krylov(phi,u,v):
    ''' To populate the Krylov vector using physiscs vectors
    Phi is a 2D array of dimension (Nx,Ny)
    u of dimensions npart
    v of dimension npart
    '''
    ykrylov = zeros(Nx*Ny+2*npart)
    ykrylov[0:Nx*Ny] = phi.reshape(Nx*Ny)
    ykrylov[Nx*Ny:Nx*Ny+npart] = u
    ykrylov[Nx*Ny+npart:] = v
    return ykrylov

def krylov_to_phys(xkrylov):
    # To populate the physiscs vectors using the Krylov space vector
    # Phi is a 2D array of dimension (Nx,Ny)
    # ubar of dimensions npart
    # vbar of dimension npart
   
    P = np.reshape(xkrylov[0:Nx*Ny],(Nx,Ny))
#    ubar = zeros(npart,float)
#    vbar = zeros(npart,float)
    ubar = xkrylov[Nx*Ny:Nx*Ny+npart]
    vbar = xkrylov[Nx*Ny+npart:]
    return P, ubar, vbar
'''
def e_field(P):
    # To compute the electric field from the potential
    Ex = zeros((Nx, Ny), float)
    Ey = zeros((Nx, Ny), float)
    Ex[1:-1] = (P[2:] - P[:-2])/dx/2
    Ex[0] = (P[1] - P[-1])/dx/2
    Ex[-1] =(P[0] - P[-2])/dx/2
    Ey[:,1:-1] = (P[:,2:] -  P[:,:-2])/dy/2
    Ey[:,0]    = (P[:,1]  -  P[:,-1])/dy/2
    Ey[:,-1]   = (P[:,0]  -  P[:,-2])/dy/2
    Ex = -Ex
    Ey = -Ey
    return Ex, Ey
'''
def laplacian(P):
    # To compute the Laplacian in uniform grid
    
    d2x = zeros((Nx, Ny), float)
    d2y = zeros((Nx, Ny), float)
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
    # Calculation of the residual of the equations
    # This is the most important part: the definion of the problem
    
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
    Ex, Ey = e_field(Ex1, Ex2, Bx3)
    Exp = grid_to_particle(xbar,ybar,Ex)
    Eyp = grid_to_particle(xbar,ybar,Ey)
    
    resu = ubar - u - QM* Exp *dt/2
    resv = vbar - v - QM* Eyp *dt/2
    

    ykrylov = phys_to_krylov(res,resu,resv)
    return  ykrylov

def grid_to_particle(x,y,E):  
    # Interpolation grid to particle
    
    global dx, dy, Nx, Ny, npart
    
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
    # Interpolation particle to grid
     
    global dx, dy, Nx, Ny, npart
    
    r =  zeros((Nx, Ny), float)
    for i in range(npart):

      # interpolate field Ex from grid to particle */
      xa = x[i]/dx
      ya = y[i]/dy
      i1 = int(xa)
      i2 = i1 + 1
      if(i2==Nx):
          i2=0 
      j1 = int(ya)
      j2 = j1 + 1  
      if(j2==Ny):
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

# PARTICLES INIT
x = Lx*np.random.rand(npart)
y = Ly*np.random.rand(npart)
u = V0x+VT*np.random.randn(npart)
v = V0y+VT*np.random.randn(npart)
q = np.ones(npart) * WP**2 / (QM*npart/Lx/Ly) 

# main cycle
histEnergy=[]
for it in range(Nt):
    plt.clf()
    guess = zeros(Nx* Ny+2*npart, float)
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
    Ex, Ey = e_field(Ex1, Ex2, Bx3)

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