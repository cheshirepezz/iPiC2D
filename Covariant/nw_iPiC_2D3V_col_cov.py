#
# authors:        G. Lapenta, L. Pezzini
# e-mail :        luca.pezzini@edu.unito.it
# date:           23.01.2021
# copyright:      2020 KU Leuven (c)
# MIT license
#

#
# Fully Implicit Particle in Cell 2D
#       • General coordinates
#       • Energy conserving
#       • Divergence free
#

import numpy as np
from scipy.optimize import newton_krylov
from numpy import cosh, zeros_like, mgrid, zeros
import matplotlib.pyplot as plt
import time

'''
Choose the tipe of visualization:
flag_plt_et  -> create a database
flag_plt_et  -> each time step
flag_plt_end -> final time step
'''
#flag_plt = True
flag_data    = False
flag_plt_et  = True
flag_plt_end = False

# parameters
nx, ny = 75, 75
Lx, Ly = 1., 1.
dx, dy = Lx/(nx - 1), Ly/(ny - 1)
dt = 0.01
nt = 20

P_left, P_right = 0, 0
P_top, P_bottom = 0, 0
xg, yg = np.mgrid[0:Lx:(nx*1j), 0:Ly:(ny*1j)]

# Species 1
npart1 = 1024
WP1 = 1.  # Plasma frequency
QM1 = -1.  # Charge/mass ratio
V0x1 = 1.  # Stream velocity
V0y1 = 1.  # Stream velocity
V0z1 = 1.  # Stream velocity
VT1 = 0.01  # thermal velocity
# Species 2
npart2 = npart1
WP2 = 1.  # Plasma frequency
QM2 = -1.  # Charge/mass ratio
V0x2 = 1.  # Stream velocity
V0y2 = 1.  # Stream velocity
V0z2 = 1.  # Stream velocity
VT2 = 0.01  # thermal velocity

npart = npart1+npart2
QM = zeros(npart, np.float64)
QM[0:npart1] = QM1
QM[npart1:npart] = QM2

# Nodes
nx1 = nx
nx2 = ny

# Computational domain
Lx1 = Lx
Lx2 = Ly
dx1 = dx
dx2 = dy

# Geometry: Tensor and Jacobian
''''
 To avoid an 8-point stencil for the metrics
 is introduced one more index to define the metrics location:
 - s = 0: gij is colocated with the E field
 - s = 1: gij is colocated with the B field
'''
J = np.ones([nx1, nx2], dtype=float)
g11 = np.ones([nx1, nx2, 2], dtype=float)
g12 = np.zeros([nx1, nx2, 2], dtype=float)
g13 = np.zeros([nx1, nx2, 2], dtype=float)
g21 = np.zeros([nx1, nx2, 2], dtype=float)
g22 = np.ones([nx1, nx2, 2], dtype=float)
g23 = np.zeros([nx1, nx2, 2], dtype=float)
g31 = np.zeros([nx1, nx2, 2], dtype=float)
g32 = np.zeros([nx1, nx2, 2], dtype=float)
g33 = np.ones([nx1, nx2, 2], dtype=float)

# Grid matrix
x1 = np.linspace(0, Lx1 - dx1, nx1, dtype=float)
x2 = np.linspace(0, Lx2 - dx2, nx2, dtype=float)
x1v, x2v = np.meshgrid(x1, x2, indexing='ij')
x1g, x2g = np.mgrid[0:Lx1:(nx1*1j), 0:Lx2:(nx2*1j)]

# Field matrix
Ex1 = np.zeros([nx1, nx2], dtype=float)
Ex2 = np.zeros([nx1, nx2], dtype=float)
Ex3 = np.zeros([nx1, nx2], dtype=float)

Bx1 = np.zeros([nx1, nx2], dtype=float)
Bx2 = np.zeros([nx1, nx2], dtype=float)
Bx3 = np.zeros([nx1, nx2], dtype=float)

Ex1new = np.zeros([nx1, nx2], dtype=float)
Ex2new = np.zeros([nx1, nx2], dtype=float)
Ex3new = np.zeros([nx1, nx2], dtype=float)

Exbar = np.zeros([nx1, nx2], dtype=float)
Eybar = np.zeros([nx1, nx2], dtype=float)
Ezbar = np.zeros([nx1, nx2], dtype=float)

Jx1 = np.zeros([nx1, nx2], dtype=float)
Jx2 = np.zeros([nx1, nx2], dtype=float)
Jx3 = np.zeros([nx1, nx2], dtype=float)

divE = np.zeros(nt, dtype=float)
U_field = np.zeros(nt, dtype=float)
U_part = np.zeros(nt, dtype=float)

# Grid begin & end point
ib = 1
ie = nx1 - 1
jb = 1
je = nx2 - 1


def myplot(values, name):
    '''
    To plot the Map of a vector fied over a grid.
    '''
    plt.figure(name)
    plt.imshow(values.T, origin='lower', extent=[
               0, Lx1, 0, Lx2], aspect='equal', vmin=-0.01, vmax=0.01)  # ,cmap='plasma')
    plt.colorbar()


def myplot2(values, name):
    '''
    To plot the behavior of a scalar fied in time.
    '''
    plt.figure(name)
    plt.plot(values)


def avg1(A, s):
    '''
    To compute the average over i index.
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je] = (A[ib+1-s:ie+1-s, jb:je] + A[ib-s:ie-s, jb:je])/2
    return res


def avg2(A, s):
    '''
    To compute the average over j index.
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je] = (A[ib:ie, jb+1-s:je+1-s] + A[ib:ie, jb-s:je-s])/2
    return res


def derx1(A, s):
    '''
    To compute the derivative along the direction x1.
    s = 0 -> Forward derivative
    s = 1 -> Beckward derivative
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je] = (A[ib+1-s:ie+1-s, jb:je] - A[ib-s:ie-s, jb:je])/dx1
    return res


def derx2(A, s):
    '''
    To compute the derivative along the direction x2.
    s = 0 -> Forward derivative
    s = 1 -> Beckward derivative
    '''
    res = np.zeros_like(A)
    res[ib:ie, jb:je] = (A[ib:ie, jb+1-s:je+1-s] - A[ib:ie, jb-s:je-s])/dx2
    return res


def curl(A1, A2, A3, s):
    '''
    To compute the Curl in covariant coordinate.
    '''
    curlx1 = np.zeros([nx1, nx2], dtype=float)
    curlx2 = np.zeros([nx1, nx2], dtype=float)
    curlx3 = np.zeros([nx1, nx2], dtype=float)

    curlx1 = (derx2(avg1(g31[:, :, s] * A1, s), s) + derx2(
        avg1(g32[:, :, s] * A2, s), s) + derx2(avg1(g33[:, :, s] * A3, s), s))/J
    curlx2 = - (derx1(avg2(g31[:, :, s] * A1, s), s) + derx1(
        avg2(g32[:, :, s] * A2, s), s) + derx1(avg2(g33[:, :, s] * A3, s), s))/J
    curlx3 = ((derx1(avg2(g21[:, :, s] * A1, s), s) + derx1(avg2(g22[:, :, s] * A2, s), s) + derx1(avg2(g23[:, :, s] * A3, s), s))
              - (derx2(avg1(g11[:, :, s] * A1, s), s) + derx2(avg1(g12[:, :, s] * A2, s), s) + derx2(avg1(g13[:, :, s] * A3, s), s)))/J
    return curlx1, curlx2, curlx3


def div(A1, A2, A3):
    '''
    To compute the Divergence in covariant coordinate.
    '''
    res = np.zeros([nx1, nx2], dtype=float)
    res = (derx1(J[:, :]*A1[:, :], 0) + derx2(J[:, :]*A2[:, :], 0))/J
    return res


def periodicBC(A):
    '''
    To impose periodic BC. The border of the grid domain is used
    to impose periodicity in each direction, while the center of the box
    contains the sensitive informations.
    '''
    A_w = np.zeros([nx1, nx2], dtype=float)
    A_s = np.zeros([nx1, nx2], dtype=float)
    # swop var.
    A_w[0, :] = A[0, :]
    A_s[:, 0] = A[:, 0]
    # Reflective BC for A field
    A[0, :] = A[-1, :]   # west = est
    A[-1, :] = A_w[0, :]  # est = west
    A[:, 0] = A[:, -1]   # south = north
    A[:, -1] = A_s[:, 0]  # north = south

def phys_to_krylov(Exk, Eyk, Ezk, uk, vk, wk):
    '''
    To populate the Krylov vector using physiscs vectors
    Ex,Ey,Ez are 2D arrays of dimension (nx,ny)
    u,v,w of dimensions npart
    '''
    global nx, ny, npart
    ykrylov = np.zeros(3*nx*ny+3*npart, dtype=float)
    ykrylov[0:nx*ny] = Exk.reshape(nx*ny)
    ykrylov[nx*ny:2*nx*ny] = Eyk.reshape(nx*ny)
    ykrylov[2*nx*ny:3*nx*ny] = Ezk.reshape(nx*ny)
    ykrylov[3*nx*ny:3*nx*ny+npart] = uk
    ykrylov[3*nx*ny+npart:3*nx*ny+2*npart] = vk
    ykrylov[3*nx*ny+2*npart:3*nx*ny+3*npart] = wk
    return ykrylov

def krylov_to_phys(xkrylov):
    '''
    To populate the physiscs vectors using the Krylov space vector
    Ex,Ey,Ez are 2D arrays of dimension (nx,ny)
    unew,vnew,wnew of dimensions npart
    '''
    Exk = np.reshape(xkrylov[0:nx*ny],(nx,ny))
    Eyk = np.reshape(xkrylov[nx*ny:2*nx*ny],(nx,ny))
    Ezk = np.reshape(xkrylov[2*nx*ny:3*nx*ny],(nx,ny))
    uk = xkrylov[3*nx*ny:3*nx*ny+npart]
    vk = xkrylov[3*nx*ny+npart:3*nx*ny+2*npart]
    wk = xkrylov[3*nx*ny+2*npart:3*nx*ny+3*npart]
    return Exk, Eyk, Ezk, uk, vk, wk

def grid_to_particle(x,y,E):
    '''
    Interpolation grid to particle
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

def particle_to_grid_J(xk,yk,uk,vk,wk,qk):
    '''
    Interpolation particle to grid
    '''
    global dx, dy, nx, ny, npart
    Jx = zeros((nx, ny), dtype=float)
    Jy = zeros((nx, ny), dtype=float)
    Jz = zeros((nx, ny), dtype=float)
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

      Jx[i1,j1] += wx1* wy1 * qk[i] * uk[i]/dx/dy
      Jx[i2,j1] += wx2* wy1 * qk[i] * uk[i]/dx/dy
      Jx[i1,j2] += wx1* wy2 * qk[i] * uk[i]/dx/dy
      Jx[i2,j2] += wx2* wy2 * qk[i] * uk[i]/dx/dy

      Jy[i1,j1] += wx1* wy1 * qk[i] * vk[i]/dx/dy
      Jy[i2,j1] += wx2* wy1 * qk[i] * vk[i]/dx/dy
      Jy[i1,j2] += wx1* wy2 * qk[i] * vk[i]/dx/dy
      Jy[i2,j2] += wx2* wy2 * qk[i] * vk[i]/dx/dy

      Jz[i1,j1] += wx1* wy1 * qk[i] * wk[i]/dx/dy
      Jz[i2,j1] += wx2* wy1 * qk[i] * wk[i]/dx/dy
      Jz[i1,j2] += wx1* wy2 * qk[i] * wk[i]/dx/dy
      Jz[i2,j2] += wx2* wy2 * qk[i] * wk[i]/dx/dy

    return Jx, Jy, Jz

def residual(xkrylov):
    ''' Calculation of the residual of the equations
    This is the most important part: the definition of the problem
    '''

    global Ex1,Ex2,Ex3,Bx1,Bx2,Bx3,u,v,w,QM,q,npart,dt
    xbar = zeros(npart,np.float64)
    ybar = zeros(npart,np.float64)

    Ex1new, Ex2new, Ex3new, unew, vnew, wnew = krylov_to_phys(xkrylov)

    ubar = (u+unew)/2.
    vbar = (v+vnew)/2.
    wbar = (w+wnew)/2.

    xbar = x + ubar*dt/2.
    ybar = y + vbar*dt/2.
    xbar = xbar%Lx1
    ybar = ybar%Lx2

    Jx1, Jx2, Jx3 = particle_to_grid_J(xbar,ybar,ubar,vbar,wbar,q)

    Ex1bar = (Ex1new+Ex1)/2.
    Ex2bar = (Ex2new+Ex2)/2.
    Ex3bar = (Ex3new+Ex3)/2.

    periodicBC(Ex1bar)
    periodicBC(Ex2bar)
    periodicBC(Ex3bar)

    curl_Ex1, curl_Ex2, curl_Ex3 = curl(Ex1bar, Ex2bar, Ex3bar, 1)

    Bx1bar = Bx1 - dt/2.*curl_Ex1
    Bx2bar = Bx2 - dt/2.*curl_Ex2
    Bx3bar = Bx3 - dt/2.*curl_Ex3

    periodicBC(Bx1bar)
    periodicBC(Bx2bar)
    periodicBC(Bx3bar)

    curl_Bx1, curl_Bx2, curl_Bx3 = curl(Bx1bar, Bx2bar, Bx3bar, 0)

    resEx1 = Ex1new - Ex1 - dt*curl_Bx1 + dt*Jx1
    resEx2 = Ex2new - Ex2 - dt*curl_Bx2 + dt*Jx2
    resEx3 = Ex3new - Ex3 - dt*curl_Bx3 + dt*Jx3

    Ex1p = grid_to_particle(xbar,ybar,Ex1bar)
    Ex2p = grid_to_particle(xbar,ybar,Ex2bar)
    Ex3p = grid_to_particle(xbar,ybar,Ex3bar)
    Bx1p = grid_to_particle(xbar,ybar,Bx1bar)
    Bx2p = grid_to_particle(xbar,ybar,Bx2bar)
    Bx3p = grid_to_particle(xbar,ybar,Bx3bar)

    resu = unew - u - QM * (Ex1p + vbar*Bx3p - wbar*Bx2p)*dt
    resv = vnew - v - QM * (Ex2p - ubar*Bx3p + wbar*Bx1p)*dt
    resw = wnew - w - QM * (Ex3p + ubar*Bx2p - vbar*Bx1p)*dt

    ykrylov = phys_to_krylov(resEx1,resEx2,resEx3,resu,resv,resw)
    return  ykrylov

# initialize particles
np.random.seed(1)

dxp = Lx/np.sqrt(npart1)
dyp = Ly/np.sqrt(npart1)
xp,yp = mgrid[dxp/2.:Lx-dxp/2.:(np.sqrt(npart1)*1j), dyp/2.:Ly-dyp/2.:(np.sqrt(npart1)*1j)]

x = zeros(npart,np.float64)
x[0:npart1] = xp.reshape(npart1)
x[0:npart1] = Lx*np.random.rand(npart1)
x[npart1:npart] = x[0:npart1]

y = zeros(npart,np.float64)
y[0:npart1] = yp.reshape(npart1)
y[0:npart1] = Ly*np.random.rand(npart1)
y[npart1:npart] = y[0:npart1]

u = zeros(npart,np.float64)
u[0:npart1] = V0x1+VT1*np.random.randn(npart1)
u[npart1:npart] = V0x2+VT2*np.random.randn(npart2)

v = zeros(npart,np.float64)
v[0:npart1] = V0y1+VT1*np.random.randn(npart1)
v[npart1:npart] = V0y2+VT2*np.random.randn(npart2)

w = zeros(npart,np.float64)
w[0:npart1] = V0z1+VT1*np.random.randn(npart1)
w[npart1:npart] = V0z2+VT2*np.random.randn(npart2)

q = zeros(npart,np.float64)
q[0:npart1] = np.ones(npart1) * WP1**2 / (QM1*npart1/Lx/Ly)
q[npart1:npart] = np.ones(npart2) * WP2**2 / (QM2*npart2/Lx/Ly)

#Bx3[int((nx1-2)/2):int((nx1+2)/2), int((nx2-2)/2):int((nx2+2)/2)] = 1.
#Bx3[int((nx1-1)/2), int((nx2-1)/2)] = 1.

# main cycle
histEnergy=[]
for t in range(nt):
    start = time.time()
    plt.clf()

    guess = zeros(3*nx*ny+3*npart,np.float64)
    sol = newton_krylov(residual, guess, method='lgmres', verbose=1)
    print('Residual: %g' % abs(residual(sol)).max())
    Ex1new, Ex2new, Ex3new, unew, vnew, wnew = krylov_to_phys(sol)

    ubar = (unew + u)/2.
    vbar = (vnew + v)/2.
    x += ubar*dt
    y += vbar*dt
    x = x%Lx
    y = y%Ly
    u = unew
    v = vnew
    w = wnew

    Ex1bar = (Ex1new+Ex1)/2.
    Ex2bar = (Ex2new+Ex2)/2.
    Ex3bar = (Ex3new+Ex3)/2.

    curl_Ex1, curl_Ex2, curl_Ex3 = curl(Ex1bar,Ex2bar,Ex3bar, 0)
    Bx1 -= dt*curl_Ex1
    Bx2 -= dt*curl_Ex2
    Bx3 -= dt*curl_Ex3
    periodicBC(Bx1)
    periodicBC(Bx2)
    periodicBC(Bx3)

    Ex1 = Ex1new
    Ex2 = Ex2new
    Ex3 = Ex3new

    U_part[t] = np.sum(u**2 + v**2 + w**2)
    U_field[t] = 0.5 * np.sum(Bx1[ib:ie, jb:je]**2 + Bx2[ib:ie, jb:je]**2 + Bx3[ib:ie, jb:je]**2\
                      + Ex1[ib:ie, jb:je]*Ex1new[ib:ie, jb:je] + Ex2[ib:ie, jb:je]*Ex2new[ib:ie, jb:je]\
                      + Ex3[ib:ie, jb:je]*Ex3new[ib:ie, jb:je])
    divE[t] = np.sum(div(Ex1, Ex2, Ex3))

    energy = U_part[t] + U_field[t]
    histEnergy.append(energy)

    print('E field         :', np.sum(Ex1), np.sum(Ex2), np.sum(Ex3))
    print('B field         :', np.sum(Bx1), np.sum(Bx2), np.sum(Bx3))
    print('Field Energy    :', U_field[t])
    print('Particles Energy:', U_part[t])
    print('div(E)          :', divE[t])

    if flag_data == True:
        f = open("iPiC_col_cov_2D3V.dat", "a")
        print(t, np.sum(Ex1), np.sum(Ex2), np.sum(Ex3), np.sum(Bx1), np.sum(Bx2), np.sum(Bx3),\
              U_field[t], U_part[t], energy, divE[t], divE[t], file=f)
        f.close()

    stop = time.time()
    print('TIME LEFT [min]: ', (stop - start)*(nt - 1 - t)/60.)

    if flag_plt_et == True:
        histEnergy.append(energy)
        print('energy =', energy)
        plt.subplot(2, 2, 1)
        plt.pcolor(xg, yg, Bx3)
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.plot(x,y,'.')
        plt.subplot(2,2,3)
        plt.plot(histEnergy)
        plt.subplot(2,2,4)
        plt.plot((histEnergy-histEnergy[0])/histEnergy[0])
        #plt.show()
        plt.pause(0.0001)

if flag_plt_end == True:
    #myplot(Ex1[:,:], 'Ex1')
    #myplot(Ex2[:,:], 'Ex2')
    myplot(Bx3[:,:], 'Bx3')
    myplot2(divE, 'divE vs t')
    myplot2(U_field, 'U_field vs t')
    #myplot2(U_part, 'U_part vs t')
    #myplot2(histEnergy, 'U_system vs t')
    plt.show()
