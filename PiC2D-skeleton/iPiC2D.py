"""
Fully Implicit Particle in Cell 2D, Charge conserving but NOT Energy conserving
Author: G. Lapenta
Date: 20 Oct 2020
Copyright 2020 KULeuven
MIT License.
"""

import numpy as np
from scipy.optimize import newton_krylov
from numpy import cosh, zeros_like, mgrid, zeros
import matplotlib.pyplot as plt

# parameters
nx, ny = 75, 75
Lx, Ly = 1.,1.
dx, dy = Lx/(nx-1), Ly/(ny-1)
dt = 0.1
nt= 10
npart = 1000

P_left, P_right = 0, 0
P_top, P_bottom = 0, 0
xg, yg = mgrid[0:Lx:(nx*1j), 0:Ly:(ny*1j)]



WP = 1. # Plasma frequency
QM = -1. # Charge/mass ratio
V0x = 0 # Stream velocity
V0y = 0 # Stream velocity
VT = .1 # thermal velocity

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
    
    Ex, Ey = e_field(P)
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
    Ex, Ey = e_field(phi)
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



