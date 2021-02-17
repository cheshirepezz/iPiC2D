"""
Fully Implicit, Relativistic, Covariant Particle-in-Cell - 2D3V Electromagnetic - 2 species
Authors: G. Lapenta, F. Bacchini, L. Pezzini
Date: 23 Jan 2021
Copyright 2020 KULeuven
MIT License.
"""

import numpy as np
from scipy.optimize import newton_krylov
from numpy import cosh, zeros_like, mgrid, zeros, ones
import matplotlib.pyplot as plt
import sys

PATH1 = '/Users/luca_pezzini/Documents/Code/cov_pic-2d/figures/'

# physics flags
electron_and_ion = False    # background of ions when QM1=QM2=-1
stable_plasma = True       # stable plasma set up
couter_stream_inst = False  # counterstream inst. set up
landau_damping = False       # landau damping set up
relativistic = False        # relativisitc  set up
# plot flags   
log_file = False            # to save the log file in PATH1
plot_dir = False          # to save the plots in PATH1
plot_each_step = True       # to visualise each time step (memory consuming)
plot_data = False           # to plot data in PATH1

# number of dpi per img (stay low 100 for monitoring purpose!)
ndpi = 100
# how often to plot
every = 10

# parameters
nx, ny = 20, 20
nxc, nyc = nx, ny
nxn, nyn = nxc+1, nyc+1
Lx, Ly = 10., 10.
dx, dy = Lx/nxc, Ly/nyc
dt = 0.05
nt = 71

# Constaint: nppc must be a squerable number (4, 16, 64) bacause particles are 
#            spread over a squared grid
nppc = 4     # per species
V0 = 1.      # stream velocity magnitude 
alpha = 0.1  # fractional reduction for VT thermal velocity

# Species 1
npart1 = nx * ny * nppc
WP1 = 1.   # Plasma frequency
QM1 = -1. # Charge/mass ratio
V0x1 = V0  # Stream velocity
V0y1 = V0  # Stream velocity
V0z1 = V0 # Stream velocity
VT1 = alpha*V0  # thermal velocity

# Species 2
npart2 = npart1
WP2 = 1.   # Plasma frequency
QM2 = 1.  # Charge/mass ratio
V0x2 = V0  # Stream velocity
V0y2 = V0  # Stream velocity
V0z2 = V0  # Stream velocity
VT2 = alpha*V0  # thermal velocity

npart = npart1 + npart2
QM = zeros(npart, np.float64)
QM[0:npart1] = QM1
QM[npart1:npart] = QM2

# INIT PARTICLES
np.random.seed(1)

dxp = Lx/np.sqrt(npart1)
dyp = Ly/np.sqrt(npart1)
xp, yp = mgrid[dxp/2.:Lx-dxp/2.:(np.sqrt(npart1)*1j), dyp/2.:Ly-dyp/2.:(np.sqrt(npart1)*1j)]

x = zeros(npart, np.float64)
x[0:npart1] = xp.reshape(npart1)
x[0:npart1] = Lx*np.random.rand(npart1)
x[npart1:npart] = x[0:npart1]

y = zeros(npart, np.float64)
y[0:npart1] = yp.reshape(npart1)
y[0:npart1] = Ly*np.random.rand(npart1)
y[npart1:npart] = y[0:npart1]

u = zeros(npart, np.float64)
if stable_plasma: 
    u[0:npart1] = V0x1+VT1*np.random.randn(npart1)
    u[npart1:npart] = V0x2+VT2*np.random.randn(npart2)
if couter_stream_inst:
    u[0:npart1] = V0x1+VT1*np.random.randn(npart1)
    u[npart1:npart] = V0x2+VT2*np.random.randn(npart2)
    # velocity in the odd position are negative 
    u[1:npart:2] = - u[1:npart:2]
    # to guarantee 50% of +u0 to e- and the other 50% to e+ and same fo -u0
    np.random.shuffle(u)
#if landau_damping == True:
#    u[0:npart1] = V0x1+VT1*np.sin(npart1)
#    u[npart1:npart] = V0x2+VT2*np.sin(npart2)

v = zeros(npart, np.float64)
v[0:npart1] = V0y1+VT1*np.random.randn(npart1)
v[npart1:npart] = V0y2+VT2*np.random.randn(npart2)
#if stable_plasma == True:
#    v[0:npart1] = VT1*np.random.randn(npart1)
#    v[npart1:npart] = VT2*np.random.randn(npart2)
#if couter_stream_inst == True:
#    v[0:npart1] = V0y1+VT1*np.random.randn(npart1)
#    v[npart1:npart] = V0y2+VT2*np.random.randn(npart2)
#    v[0:npart1:2] = - v[0:npart1:2]
#    v[npart1:npart:2] = - v[npart1:npart:2]
if landau_damping:
    v[0:npart1] = V0y1+VT1*np.sin(x[0:npart1]/Lx)
    v[npart1:npart] = V0y2+VT2*np.sin(x[npart1:npart]/Lx)

w = zeros(npart, np.float64)
w[0:npart1] = V0z1+VT1*np.random.randn(npart1)
w[npart1:npart] = V0z2+VT2*np.random.randn(npart2)
#if stable_plasma == True:
#    w[0:npart1] = VT1*np.random.randn(npart1)
#    w[npart1:npart] = VT2*np.random.randn(npart2)
#if couter_stream_inst == True:
#    w[0:npart1] = V0z1+VT1*np.random.randn(npart1)
#    w[npart1:npart] = V0z2+VT2*np.random.randn(npart2)
#    w[0:npart1:2] = - w[0:npart1:2]
#    w[npart1:npart:2] = - w[npart1:npart:2]
#if landau_damping == True:
#    w[0:npart1] = V0x1+VT1*np.sin(npart1)
#    w[npart1:npart] = V0x2+VT2*np.sin(npart2)

q = zeros(npart, np.float64)
#q[0:npart1] =  np.ones(npart1)
#q[npart1:npart] = - np.ones(npart2) 
q[0:npart1] = np.ones(npart1)*WP1**2/(QM1*npart1/Lx/Ly) 
q[npart1:npart] = np.ones(npart2)*WP2**2/(QM2*npart2/Lx/Ly)

if relativistic:
    g = 1./np.sqrt(1.-(u**2+v**2+w**2))
    u = u*g
    v = v*g
    w = w*g

# INIT GRID
# grid of centres c
xc, yc = mgrid[dx/2.:Lx-dx/2.:(nxc*1j), dy/2.:Ly-dy/2.:(nyc*1j)]
# grid of left-right faces LR
xLR, yLR = mgrid[0.:Lx:(nxn*1j), dy/2.:Ly-dy/2.:(nyc*1j)]
# grid of up-down faces UD
xUD, yUD = mgrid[dx/2.:Lx-dx/2.:(nxc*1j), 0.:Ly:(nyn*1j)]
# grid of corners n
xn, yn = mgrid[0.:Lx:(nxn*1j), 0.:Ly:(nyn*1j)]

# INIT FIELDS
# defined on grid LR:        Ex, Jx, By
# defined on grid UD:        Ey, Jy, Bx
# defined on grid centres c: Ez, Jz, rho
# defined on grid corners n: Bz

Ex = zeros(np.shape(xLR),np.float64)
Ey = zeros(np.shape(xUD),np.float64)
Ez = zeros(np.shape(xc),np.float64)
Bx = zeros(np.shape(xUD),np.float64)
By = zeros(np.shape(xLR),np.float64)
Bz = zeros(np.shape(xn),np.float64)
rho = zeros(np.shape(xc), np.float64)

# INIT METRIC TENSOR
# defined on grid LR:        (g11e, g21e, g31e)E1
#                            (g11b, g21b, g31b)B1
# defined on grid UD:        (g12e, g22e, g32e)E2
#                            (g12b, g22b, g32b)B2
# defined on grid centres c: (g12e, g23e, g33e)E3
# defined on grid corners n: (g12b, g23b, g33b)B3

g11e = np.ones(np.shape(xLR), np.float64)
g12e = np.zeros(np.shape(xUD), np.float64)
g13e = np.zeros(np.shape(xc), np.float64)
g21e = np.zeros(np.shape(xLR), np.float64)
g22e = np.ones(np.shape(xUD), np.float64)
g23e = np.zeros(np.shape(xc), np.float64)
g31e = np.zeros(np.shape(xLR), np.float64)
g32e = np.zeros(np.shape(xUD), np.float64)
g33e = np.ones(np.shape(xc), np.float64)

g11b = np.ones(np.shape(xUD), np.float64)
g12b = np.zeros(np.shape(xLR), np.float64)
g13b = np.zeros(np.shape(xn), np.float64)
g21b = np.zeros(np.shape(xUD), np.float64)
g22b = np.ones(np.shape(xLR), np.float64)
g23b = np.zeros(np.shape(xn), np.float64)
g31b = np.zeros(np.shape(xUD), np.float64)
g32b = np.zeros(np.shape(xLR), np.float64)
g33b = np.ones(np.shape(xn), np.float64)

# INIT JACOBIAN DETERMINANT

J_UD = np.ones(np.shape(xUD), np.float64)
J_LR = np.ones(np.shape(xLR), np.float64)
J_C = np.ones(np.shape(xc), np.float64)
J_N = np.ones(np.shape(xn), np.float64)

# INIT JACOBIAN MATRIX
# defined on grid LR:        (j11e, j21e, j31e)E1, J1
#                            (j11b, j21b, j31b)B1
# defined on grid UD:        (J12e, J22e, J32e)E2, J2
#                            (J12b, J22b, J32b)B2
# defined on grid centres c: (J13e, J23e, J33e)E3, J3
#                            (J13b, J23b, J33b)B3

J11e = np.ones(np.shape(xLR), np.float64)
J12e = np.zeros(np.shape(xUD), np.float64)
J13e = np.zeros(np.shape(xc), np.float64)
J21e = np.zeros(np.shape(xLR), np.float64)
J22e = np.ones(np.shape(xUD), np.float64)
J23e = np.zeros(np.shape(xc), np.float64)
J31e = np.zeros(np.shape(xLR), np.float64)
J32e = np.zeros(np.shape(xUD), np.float64)
J33e = np.ones(np.shape(xc), np.float64)

J11b = np.ones(np.shape(xUD), np.float64)
J12b = np.zeros(np.shape(xLR), np.float64)
J13b = np.zeros(np.shape(xn), np.float64)
J21b = np.zeros(np.shape(xUD), np.float64)
J22b = np.ones(np.shape(xLR), np.float64)
J23b = np.zeros(np.shape(xn), np.float64)
J31b = np.zeros(np.shape(xUD), np.float64)
J32b = np.zeros(np.shape(xLR), np.float64)
J33b = np.ones(np.shape(xn), np.float64)

# INIT INVERSE JACOBIAN MATRIX
# defined on grid LR:        (j11e, j21e, j31e)E1, J1
#                            (j11b, j21b, j31b)B1
# defined on grid UD:        (j12e, j22e, j32e)E2, J2
#                            (j12b, j22b, j32b)B2
# defined on grid centres c: (j12e, j23e, j33e)E3, J3
#                            (j12b, j23b, j33b)B3

j11e = np.ones(np.shape(xLR), np.float64)
j12e = np.zeros(np.shape(xUD), np.float64)
j13e = np.zeros(np.shape(xc), np.float64)
j21e = np.zeros(np.shape(xLR), np.float64)
j22e = np.ones(np.shape(xUD), np.float64)
j23e = np.zeros(np.shape(xc), np.float64)
j31e = np.zeros(np.shape(xLR), np.float64)
j32e = np.zeros(np.shape(xUD), np.float64)
j33e = np.ones(np.shape(xc), np.float64)

j11b = np.ones(np.shape(xUD), np.float64)
j12b = np.zeros(np.shape(xLR), np.float64)
j13b = np.zeros(np.shape(xn), np.float64)
j21b = np.zeros(np.shape(xUD), np.float64)
j22b = np.ones(np.shape(xLR), np.float64)
j23b = np.zeros(np.shape(xn), np.float64)
j31b = np.zeros(np.shape(xUD), np.float64)
j32b = np.zeros(np.shape(xLR), np.float64)
j33b = np.ones(np.shape(xn), np.float64)

# Divergence
# defined on grid c: div(E)
# defined on grid n: div(E)
divE = zeros(nt, np.float64)
divB = zeros(nt, np.float64)

# Energy
energyP = zeros(nt, np.float64) # particles
energyE = zeros(nt, np.float64) # E field 
energyB = zeros(nt, np.float64) # B field

if log_file == True:
    f = open(PATH1 + 'log_file.txt', 'w')
    print('* PHYSICS:', file=f)
    print('stable plasma: ', stable_plasma, file=f)
    print('electrons & ions: ', electron_and_ion, file=f)
    print('counter stream inst.: ', couter_stream_inst, file=f)
    print('landau damping: ', landau_damping, file=f)
    print('relativistic: ', relativistic, file=f)
    print('* PARAMETER:', file=f)
    print('number nodes (x-axes): ', nx, file=f)
    print('number nodes (y-axes): ', ny, file=f)
    print('length of the domain (x-axes): ', Lx, file=f)
    print('length of the domain (y-axes): ', Ly, file=f)
    print('time steps: ', dt, file=f)
    print('number of time steps: ', nt, file=f)
    print('number of part. per cell: ', nppc, file=f)
    print('* SPECIES 1:', file=f)
    print('number of particles : ', npart1, file=f)
    print('plasma frequency : ', WP1, file=f)
    print('charge to mass : ', QM1, file=f)
    print('velocity field: ', '(', V0x1, ',', V0y1, ',', V0z1, ')', file=f)
    print('thermal velocity: ', VT1, file=f)
    print('* SPECIES 2:', file=f)
    print('number of particles : ', npart2, file=f)
    print('plasma frequency : ', WP2, file=f)
    print('charge to mass : ', QM2, file=f)
    print('velocity field: ', '(', V0x2, ',', V0y2, ',', V0z2, ')', file=f)
    print('thermal velocity: ', VT2, file=f)
    f.close()

def myplot_map(xgrid, ygrid, field, title = 'a', xlabel= 'b', ylabel= 'c'):
    '''
    To plot the Map of a vector fied over a grid.
    '''
    plt.figure()
    plt.pcolor(xgrid, ygrid, field)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()

def myplot_func(field, title= 'a', xlabel= 'b', ylabel= 'c'):
    '''
    To plot the behavior of a scalar fied in time.
    '''
    plt.figure()
    plt.plot(field)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def myplot_particle_map(posx, posy):
    plt.figure()
    plt.plot(posx[0:npart1],posy[0:npart1],'b.')
    plt.plot(posx[npart1:npart],posy[npart1:npart],'r.')
    plt.xlim((0,Lx))
    plt.ylim((0,Ly))
    plt.title('Particles map')
    plt.xlabel('x')
    plt.ylabel('y')

def myplot_phase_space(pos, vel, limx=(0, 0), limy=(0, 0), xlabel='b', ylabel='c'):
    plt.figure()
    plt.plot(pos[0:npart1], vel[0:npart1], 'b.')
    plt.plot(pos[npart1:npart], vel[npart1:npart], 'r.')
    plt.xlim(limx)
    plt.ylim(limy)
    plt.title('Particles map')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def dirder(field, dertype):
    ''' To take the directional derivative of a quantity
        dertype defines input/output grid type and direction
    '''
    global nxn, nyn, nxc, nyc, dx, dy

    if dertype == 'C2UD':  # centres to UD faces, y-derivative
      derfield = zeros((nxc, nyn), np.float64)

      derfield[0:nxc, 1:nyn-1] = (field[0:nxc, 1:nyc]-field[0:nxc, 0:nyc-1])/dy
      derfield[0:nxc, 0] = (field[0:nxc, 0]-field[0:nxc, nyc-1])/dy
      derfield[0:nxc, nyn-1] = derfield[0:nxc, 0]

    elif dertype == 'C2LR':  # centres to LR faces, x-derivative
      derfield = zeros((nxn, nyc), np.float64)

      derfield[1:nxn-1, 0:nyc] = (field[1:nxc, 0:nyc]-field[0:nxc-1, 0:nyc])/dx
      derfield[0, 0:nyc] = (field[0, 0:nyc]-field[nxc-1, 0:nyc])/dx
      derfield[nxn-1, 0:nyc] = derfield[0, 0:nyc]

    elif dertype == 'UD2N':  # UD faces to nodes, x-derivative
      derfield = zeros((nxn, nyn), np.float64)

      derfield[1:nxn-1, 0:nyn] = (field[1:nxc, 0:nyn]-field[0:nxc-1, 0:nyn])/dx
      derfield[0, 0:nyn] = (field[0, 0:nyn]-field[nxc-1, 0:nyn])/dx
      derfield[nxn-1, 0:nyn] = derfield[0, 0:nyn]

    elif dertype == 'LR2N':  # LR faces to nodes, y-derivative
      derfield = zeros((nxn, nyn), np.float64)

      derfield[0:nxn, 1:nyn-1] = (field[0:nxn, 1:nyc]-field[0:nxn, 0:nyc-1])/dy
      derfield[0:nxn, 0] = (field[0:nxn, 0]-field[0:nxn, nyc-1])/dy
      derfield[0:nxn, nyn-1] = derfield[0:nxn, 0]

    elif dertype == 'N2LR':  # nodes to LR faces, y-derivative
      derfield = zeros((nxn, nyc), np.float64)

      derfield[0:nxn, 0:nyc] = (field[0:nxn, 1:nyn]-field[0:nxn, 0:nyn-1])/dy

    elif dertype == 'N2UD':  # nodes to UD faces, x-derivative
      derfield = zeros((nxc, nyn), np.float64)

      derfield[0:nxc, 0:nyn] = (field[1:nxn, 0:nyn]-field[0:nxn-1, 0:nyn])/dx

    elif dertype == 'LR2C':  # LR faces to centres, x-derivative
      derfield = zeros((nxc, nyc), np.float64)

      derfield[0:nxc, 0:nyc] = (field[1:nxn, 0:nyc]-field[0:nxn-1, 0:nyc])/dx

    elif dertype == 'UD2C':  # UD faces to centres, y-derivative
      derfield = zeros((nxc, nyc), np.float64)

      derfield[0:nxc, 0:nyc] = (field[0:nxc, 1:nyn]-field[0:nxc, 0:nyn-1])/dy

    return derfield

def avgC2N(fieldC):
    ''' To average a 2D field defined on centres to the nodes
    '''
    global nx,ny

    fieldN = zeros((nx,ny),np.float64)

    fieldN[1:nx-1,1:ny-1] = (fieldC[0:nx-2,0:ny-2]+fieldC[1:nx-1,0:ny-2]
                             +fieldC[0:nx-2,1:ny-1]+fieldC[1:nx-1,1:ny-1])/4.
    fieldN[0,1:ny-1] = (fieldC[0,0:ny-2]+fieldC[0,1:ny-1]+fieldC[nx-2,0:ny-2]+fieldC[nx-2,1:ny-1])/4.
    fieldN[nx-1,1:ny-1] = fieldN[0,1:ny-1]
    fieldN[1:nx-1,0] = (fieldC[0:nx-2,0]+fieldC[1:nx-1,0]+fieldC[0:nx-2,ny-2]+fieldC[1:nx-1,ny-2])/4.
    fieldN[1:nx-1,ny-1] = fieldN[1:nx-1,0]
    fieldN[0,0] = (fieldC[0,0]+fieldC[0,ny-2]+fieldC[nx-2,0]+fieldC[nx-2,ny-2])/4.
    fieldN[0,ny-1] = fieldN[0,0]
    fieldN[nx-1,0] = fieldN[0,0]
    fieldN[nx-1,ny-1] = fieldN[0,0]
    return fieldN

def avg(field, avgtype):
    ''' To take the average of a quantity
        avgtype defines input/output grid type and direction
    '''
    global nxn, nyn, nxc, nyc, dx, dy

    if avgtype == 'C2UD':  # centres to UD faces, y-average
      avgfield = zeros((nxc, nyn), np.float64)

      avgfield[0:nxc, 1:nyn-1] = (field[0:nxc, 1:nyc]+field[0:nxc, 0:nyc-1])/2.
      avgfield[0:nxc, 0] = (field[0:nxc, 0]+field[0:nxc, nyc-1])/2.
      avgfield[0:nxc, nyn-1] = avgfield[0:nxc, 0]

    elif avgtype == 'C2LR':  # centres to LR faces, x-average
      avgfield = zeros((nxn, nyc), np.float64)

      avgfield[1:nxn-1, 0:nyc] = (field[1:nxc, 0:nyc]+field[0:nxc-1, 0:nyc])/2.
      avgfield[0, 0:nyc] = (field[0, 0:nyc]+field[nxc-1, 0:nyc])/2.
      avgfield[nxn-1, 0:nyc] = avgfield[0, 0:nyc]

    elif avgtype == 'UD2N':  # UD faces to nodes, x-average
      avgfield = zeros((nxn, nyn), np.float64)

      avgfield[1:nxn-1, 0:nyn] = (field[1:nxc, 0:nyn]+field[0:nxc-1, 0:nyn])/2.
      avgfield[0, 0:nyn] = (field[0, 0:nyn]+field[nxc-1, 0:nyn])/2.
      avgfield[nxn-1, 0:nyn] = avgfield[0, 0:nyn]

    elif avgtype == 'LR2N':  # LR faces to nodes, y-average
      avgfield = zeros((nxn, nyn), np.float64)

      avgfield[0:nxn, 1:nyn-1] = (field[0:nxn, 1:nyc]+field[0:nxn, 0:nyc-1])/2.
      avgfield[0:nxn, 0] = (field[0:nxn, 0]+field[0:nxn, nyc-1])/2.
      avgfield[0:nxn, nyn-1] = avgfield[0:nxn, 0]

    elif avgtype == 'N2LR':  # nodes to LR faces, y-average
      avgfield = zeros((nxn, nyc), np.float64)

      avgfield[0:nxn, 0:nyc] = (field[0:nxn, 1:nyn]+field[0:nxn, 0:nyn-1])/2.

    elif avgtype == 'N2UD':  # nodes to UD faces, x-average
      avgfield = zeros((nxc, nyn), np.float64)

      avgfield[0:nxc, 0:nyn] = (field[1:nxn, 0:nyn]+field[0:nxn-1, 0:nyn])/2.

    elif avgtype == 'LR2C':  # LR faces to centres, x-average
      avgfield = zeros((nxc, nyc), np.float64)

      avgfield[0:nxc, 0:nyc] = (field[1:nxn, 0:nyc]+field[0:nxn-1, 0:nyc])/2.

    elif avgtype == 'UD2C':  # UD faces to centres, y-average
      avgfield = zeros((nxc, nyc), np.float64)

      avgfield[0:nxc, 0:nyc] = (field[0:nxc, 1:nyn]+field[0:nxc, 0:nyn-1])/2.

    return avgfield

def curl(fieldx, fieldy, fieldz, fieldtype):
    ''' To take the curl of either E or B
    fieltype=='E': input -> LR,UD,c, output -> UD,LR,n
    fieltype=='B': input -> UD,LR,n, output -> LR,UD,c
    '''
    if fieldtype=='E':
      curl_x =   (dirder(avg(g31e * fieldx, 'LR2C'), 'C2UD') + dirder(avg(g32e * fieldy, 'UD2C'), 'C2UD') + dirder(g33e * fieldz, 'C2UD'))/J_UD
      curl_y = - (dirder(avg(g31e * fieldx, 'LR2C'), 'C2LR') + dirder(avg(g32e * fieldy, 'UD2C'), 'C2LR') + dirder(g33e * fieldz, 'C2LR'))/J_LR
      curl_z = (dirder(avg(avg(g21e * fieldx, 'LR2C'),'C2UD'), 'UD2N') + dirder(g22e * fieldy, 'UD2N') + dirder(avg(g13e * fieldz, 'C2UD'), 'UD2N')
              - dirder(g11e * fieldx, 'LR2N') + dirder(avg(avg(g12e * fieldy, 'UD2C'), 'C2LR'), 'LR2N') + dirder(avg(g13e * fieldz, 'C2LR'), 'LR2N'))/J_N

    elif fieldtype == 'B':
      curl_x=   (dirder(avg(g31b * fieldx, 'UD2N'), 'N2LR') + dirder(avg(g32b * fieldy, 'LR2N'), 'N2LR') + dirder(g33b * fieldz, 'N2LR'))/J_LR
      curl_y= - (dirder(avg(g31b * fieldx, 'UD2N'), 'N2UD') + dirder(avg(g32b * fieldy, 'LR2N'), 'N2UD') + dirder(g33b * fieldz, 'N2UD'))/J_UD
      curl_z=   (dirder(avg(avg(g21b * fieldx, 'UD2N'), 'N2LR'), 'LR2C') + dirder(g22b * fieldy, 'LR2C') + dirder(avg(g13b * fieldz, 'N2LR'), 'LR2C')
              - dirder(g11b * fieldx, 'UD2C') + dirder(avg(avg(g12b * fieldy, 'LR2N'), 'N2UD'), 'UD2C') + dirder(avg(g13b * fieldz, 'N2UD'), 'UD2C'))/J_C

    return curl_x, curl_y, curl_z

def div(fieldx, fieldy, fieldz, fieldtype):
    ''' To take the divergence of either E or B
    fieltype=='E': input -> LR,UD,c, output -> c,c,c
    fieltype=='B': input -> UD,LR,n, output -> n,n,n
    '''
    if fieldtype == 'E':
        div = (dirder(fieldx, 'LR2C') + dirder(fieldy, 'UD2C'))/J_C

    elif fieldtype == 'B':
        div = (dirder(fieldx, 'UD2N') + dirder(fieldy, 'LR2N'))/J_N

    return div

def phys_to_krylov(Exk, Eyk, Ezk, uk, vk, wk):
    ''' To populate the Krylov vector using physiscs vectors
    Ex,Ey,Ez are 2D arrays
    u,v,w of dimensions npart
    '''
    global nxc,nyc,nxn,nyn,npart

    ykrylov = zeros(nxn*nyc+nxc*nyn+nxc*nyc+3*npart,np.float64)
    ykrylov[0:nxn*nyc] = Exk.reshape(nxn*nyc)
    ykrylov[nxn*nyc:nxn*nyc+nxc*nyn] = Eyk.reshape(nxc*nyn)
    ykrylov[nxn*nyc+nxc*nyn:nxn*nyc+nxc*nyn+nxc*nyc] = Ezk.reshape(nxc*nyc)
    ykrylov[nxn*nyc+nxc*nyn+nxc*nyc:nxn*nyc+nxc*nyn+nxc*nyc+npart] = uk
    ykrylov[nxn*nyc+nxc*nyn+nxc*nyc+npart:nxn*nyc+nxc*nyn+nxc*nyc+2*npart] = vk
    ykrylov[nxn*nyc+nxc*nyn+nxc*nyc+2*npart:nxn*nyc+nxc*nyn+nxc*nyc+3*npart] = wk
    return ykrylov

def krylov_to_phys(xkrylov):
    ''' To populate the physiscs vectors using the Krylov space vector
    Ex,Ey,Ez are 2D arrays of dimension (nx,ny)
    unew,vnew,wnew of dimensions npart1+npart2
    '''
    global nx,ny,npart

    Exk = np.reshape(xkrylov[0:nxn*nyc],(nxn,nyc))
    Eyk = np.reshape(xkrylov[nxn*nyc:nxn*nyc+nxc*nyn],(nxc,nyn))
    Ezk = np.reshape(xkrylov[nxn*nyc+nxc*nyn:nxn*nyc+nxc*nyn+nxc*nyc],(nxc,nyc))
    uk = xkrylov[nxn*nyc+nxc*nyn+nxc*nyc:nxn*nyc+nxc*nyn+nxc*nyc+npart]
    vk = xkrylov[nxn*nyc+nxc*nyn+nxc*nyc+npart:nxn*nyc+nxc*nyn+nxc*nyc+2*npart]
    wk = xkrylov[nxn*nyc+nxc*nyn+nxc*nyc+2*npart:nxn*nyc+nxc*nyn+nxc*nyc+3*npart]
    return Exk, Eyk, Ezk, uk, vk, wk

def residual(xkrylov):
    ''' Calculation of the residual of the equations
    This is the most important part: the definition of the problem
    '''

    global Ex,Ey,Ez,Bx,By,Bz,u,v,w,QM,q,npart,dt

    Exnew, Eynew, Eznew, unew, vnew, wnew = krylov_to_phys(xkrylov)
    
    ubar = (u+unew)/2.
    vbar = (v+vnew)/2.
    wbar = (w+wnew)/2.

    if relativistic:
      gold = np.sqrt(1.+u**2+v**2+w**2)
      gnew = np.sqrt(1.+unew**2+vnew**2+wnew**2)
      gbar = (gold+gnew)/2.
    else:
     gbar = np.ones(npart)

    xbar = x + ubar/gbar*dt/2.
    ybar = y + vbar/gbar*dt/2.
    xbar = xbar%Lx
    ybar = ybar%Ly
    
    Jx, Jy, Jz = particle_to_grid_J(xbar,ybar,ubar/gbar,vbar/gbar,wbar/gbar,q)

    Exbar = (Exnew+Ex)/2.
    Eybar = (Eynew+Ey)/2.
    Ezbar = (Eznew+Ez)/2.

    curlE_x, curlE_y, curlE_z = curl(Exbar,Eybar,Ezbar,'E')

    Bxbar = Bx - dt/2.*curlE_x
    Bybar = By - dt/2.*curlE_y
    Bzbar = Bz - dt/2.*curlE_z

    curlB_x, curlB_y, curlB_z = curl(Bxbar,Bybar,Bzbar,'B')

    Jxgen, Jygen, Jzgen = cartesian_to_general(Jx, Jy, Jz, 'J')
    
    resEx = Exnew - Ex - dt*curlB_x + dt*Jxgen
    resEy = Eynew - Ey - dt*curlB_y + dt*Jygen
    resEz = Eznew - Ez - dt*curlB_z + dt*Jzgen

    Excart, Eycart, Ezcart = general_to_cartesian(Exbar, Eybar, Ezbar, 'E')
    Bxcart, Bycart, Bzcart = general_to_cartesian(Bxbar, Bybar, Bzbar, 'B')

    Exp = grid_to_particle(xbar,ybar,Excart,'LR')  #Exbar
    Eyp = grid_to_particle(xbar,ybar,Eycart,'UD')  #Eybar
    Ezp = grid_to_particle(xbar,ybar,Ezcart,'C')   #Ezbar
    Bxp = grid_to_particle(xbar,ybar,Bxcart,'UD')  #Bxbar
    Byp = grid_to_particle(xbar,ybar,Bycart,'LR')  #Bybar
    Bzp = grid_to_particle(xbar,ybar,Bzcart,'N')   #Bzbar

    resu = unew - u - QM * (Exp + vbar/gbar*Bzp - wbar/gbar*Byp)*dt
    resv = vnew - v - QM * (Eyp - ubar/gbar*Bzp + wbar/gbar*Bxp)*dt
    resw = wnew - w - QM * (Ezp + ubar/gbar*Byp - vbar/gbar*Bxp)*dt

    ykrylov = phys_to_krylov(resEx,resEy,resEz,resu,resv,resw)
    return  ykrylov

def grid_to_particle(xk, yk, f, gridtype):
    ''' Interpolation of grid quantity to particle
    '''
    global dx, dy, nx, ny, npart
    
    fp = zeros(npart,np.float64)

    fx, fy = 0., 0.
    if gridtype=='LR':
      fy = dy/2.
    elif gridtype=='UD':
      fx = dx/2.
    elif gridtype=='C':
      fx, fy = dx/2., dy/2.

    for i in range(npart):

      #  interpolate field f from grid to particle */
      xa = (xk[i]-fx)/dx
      ya = (yk[i]-fy)/dy
      i1 = int(np.floor(xa))
      i2 = i1 + 1
      j1 = int(np.floor(ya))
      j2 = j1 + 1 
      wx2 = xa - np.float64(i1)
      wx1 = 1.0 - wx2
      wy2 = ya - np.float64(j1)
      wy1 = 1.0 - wy2
      if gridtype=='LR':
        j1, j2 = j1%nyc, j2%nyc
      elif gridtype=='UD':
        i1, i2 = i1%nxc, i2%nxc
      elif gridtype=='C':
        i1, i2 = i1%nxc, i2%nxc
        j1, j2 = j1%nyc, j2%nyc

      fp[i] = wx1* wy1 * f[i1,j1] + wx2* wy1 * f[i2,j1] + wx1* wy2 * f[i1,j2] + wx2* wy2 * f[i2,j2]
    
    return fp

def particle_to_grid_rho(x, y, q):
    ''' Interpolation particle to grid - charge rho -> c
    '''
    global dx, dy, nx, ny, npart

    if electron_and_ion:
        # 2 species same charge -> positive ions bkg
        # each node have to compensate to 2.*nppc neg. charges
        #r = np.abs(q[0])*2.*nppc*ones((nx, ny), np.float64)
        r = np.abs(q[0])*2.*nppc*ones(np.shape(xc), np.float64)
    else:
        # 2 species opposit charge -> no ions bkg
        #r = zeros((nx, ny), np.float64)  
        r = zeros(np.shape(xc), np.float64)

    for i in range(npart):
      #  interpolate field Ex from grid to particle */
        xa = (x[i]-dx/2.)/dx
        ya = (y[i]-dy/2.)/dy
        i1 = int(np.floor(xa))
        i2 = i1 + 1
        j1 = int(np.floor(ya))
        j2 = j1 + 1  
        wx2 = xa - np.float64(i1)
        wx1 = 1.0 - wx2
        wy2 = ya - np.float64(j1)
        wy1 = 1.0 - wy2
        i1, i2 = i1%nxc, i2%nxc
        j1, j2 = j1%nyc, j2%nyc

        r[i1, j1] += wx1 * wy1 * q[i]
        r[i2, j1] += wx2 * wy1 * q[i]
        r[i1, j2] += wx1 * wy2 * q[i]
        r[i2, j2] += wx2 * wy2 * q[i]

    return r

def particle_to_grid_J(xk, yk, uk, vk, wk, qk): 
    ''' Interpolation particle to grid - current -> LR, UD, c
    ''' 
    global dx, dy, nxc, nyc, nxn, nyn, npart
  
    Jx = zeros(np.shape(xLR),np.float64)
    Jy = zeros(np.shape(xUD),np.float64)
    Jz = zeros(np.shape(xc),np.float64)

    for i in range(npart):

      #  interpolate p -> LR
      xa = xk[i]/dx 
      ya = (yk[i]-dy/2.)/dy
      i1 = int(np.floor(xa))
      i2 = i1 + 1
      if i2==nxn-1:
        i2=0
      j1 = int(np.floor(ya))
      j2 = j1 + 1  
      wx2 = xa - np.float64(i1)
      wx1 = 1.0 - wx2
      wy2 = ya - np.float64(j1)
      wy1 = 1.0 - wy2
      j1, j2 = j1%nyc, j2%nyc

      Jx[i1,j1] += wx1* wy1 * qk[i] * uk[i]/dx/dy
      Jx[i2,j1] += wx2* wy1 * qk[i] * uk[i]/dx/dy
      Jx[i1,j2] += wx1* wy2 * qk[i] * uk[i]/dx/dy
      Jx[i2,j2] += wx2* wy2 * qk[i] * uk[i]/dx/dy

      # interpolate p -> UD
      xa = (xk[i]-dx/2.)/dx 
      ya = yk[i]/dy
      i1 = int(np.floor(xa))
      i2 = i1 + 1
      j1 = int(np.floor(ya))
      j2 = j1 + 1  
      if j2==nyn-1:
        j2=0
      wx2 = xa - np.float64(i1)
      wx1 = 1.0 - wx2
      wy2 = ya - np.float64(j1)
      wy1 = 1.0 - wy2
      i1, i2 = i1%nxc, i2%nxc

      Jy[i1,j1] += wx1* wy1 * qk[i] * vk[i]/dx/dy
      Jy[i2,j1] += wx2* wy1 * qk[i] * vk[i]/dx/dy
      Jy[i1,j2] += wx1* wy2 * qk[i] * vk[i]/dx/dy
      Jy[i2,j2] += wx2* wy2 * qk[i] * vk[i]/dx/dy

      # interpolate p -> c
      xa = (xk[i]-dx/2.)/dx 
      ya = (yk[i]-dy/2.)/dy
      i1 = int(np.floor(xa))
      i2 = i1 + 1
      j1 = int(np.floor(ya))
      j2 = j1 + 1  
      wx2 = xa - np.float64(i1)
      wx1 = 1.0 - wx2
      wy2 = ya - np.float64(j1)
      wy1 = 1.0 - wy2
      i1, i2 = i1%nxc, i2%nxc
      j1, j2 = j1%nyc, j2%nyc

      Jz[i1,j1] += wx1* wy1 * qk[i] * wk[i]/dx/dy
      Jz[i2,j1] += wx2* wy1 * qk[i] * wk[i]/dx/dy
      Jz[i1,j2] += wx1* wy2 * qk[i] * wk[i]/dx/dy
      Jz[i2,j2] += wx2* wy2 * qk[i] * wk[i]/dx/dy

    Jx[nxn-1,:] = Jx[0,:]
    Jy[:,nyn-1] = Jy[:,0]

    return Jx, Jy, Jz

def cartesian_to_general(cartx, carty, cartz, fieldtype):
    ''' To convert fields from Cartesian geom. to General geom.
        fieltype=='E' or 'J': input -> LR,UD,c, output -> LR,UD,c
        fieltype=='B':        input -> UD,LR,n, output -> UD,LR,n
    '''
    if (fieldtype == 'E') or (fieldtype == 'J'):
      genx = J_LR*J11e*cartx + avg(avg(J_UD*J12e*carty, 'UD2C'), 'C2LR')+ avg(J_C*J13e*cartz, 'C2LR')
      geny = avg(avg(J_LR*J21e*cartx, 'LR2C'), 'C2UD') + J_UD*J22e*carty + avg(J_C*J23e*cartz, 'C2UD')
      genz = avg(J_LR*J31e*cartx, 'LR2C') + avg(J_UD*J32e*carty, 'UD2C') + J_C*J33e*cartz
    elif fieldtype == 'B':
      genx = J_UD*J11b*cartx + avg(avg(J_LR*J12b*carty, 'LR2C'), 'C2UD') + avg(J_N*J13b*cartz, 'N2UD')
      geny = avg(avg(J_UD*J21b*cartx, 'UD2C'), 'C2LR') + J_LR*J22b*carty + avg(J_N*J23b*cartz, 'N2LR')
      genz = avg(J_UD*J31b*cartx, 'UD2N') + avg(J_LR*J32b*carty, 'LR2N') + J_N*J33b*cartz
    
    return genx, geny, genz

def general_to_cartesian(genx, geny, genz, fieldtype):
    ''' To convert fields from General geom. to Cartesian geom.
        fieltype=='E' or 'J': input -> LR,UD,c, output -> LR,UD,c
        fieltype=='B':        input -> UD,LR,n, output -> UD,LR,n
    '''
    if (fieldtype == 'E') or (fieldtype == 'J'):
      cartx = J_LR*j11e*genx + avg(avg(J_UD*j12e*geny, 'UD2C'), 'C2LR')+ avg(J_C*j13e*genz, 'C2LR')
      carty = avg(avg(J_LR*j21e*genx, 'LR2C'), 'C2UD') + J_UD*j22e*geny + avg(J_C*j23e*genz, 'C2UD')
      cartz = avg(J_LR*j31e*genx, 'LR2C') + avg(J_UD*j32e*geny, 'UD2C') + J_C*j33e*genz
    elif fieldtype == 'B':
      cartx = J_UD*j11b*genx + avg(avg(J_LR*j12b*geny, 'LR2C'), 'C2UD') + avg(J_N*j13b*genz, 'N2UD')
      carty = avg(avg(J_UD*j21b*genx, 'UD2C'), 'C2LR') + J_LR*j22b*geny + avg(J_N*j23b*genz, 'N2LR')
      cartz = avg(J_UD*j31b*genx, 'UD2N') + avg(J_LR*j32b*geny, 'LR2N') + J_N*j33b*genz
    
    return cartx, carty, cartz

# main cycle
if relativistic:
  histEnergyP1=[np.sum((g[0:npart1]-1.)*abs(q[0:npart1]/QM[0:npart1]))]
  histEnergyP2=[np.sum((g[npart1:npart]-1.)*abs(q[npart1:npart]/QM[npart1:npart]))]
else:
  histEnergyP1=[np.sum((u[0:npart1]**2+v[0:npart1]**2+w[0:npart1]**2)/2.*abs(q[0:npart1]/QM[0:npart1]))]
  histEnergyP2=[np.sum((u[npart1:npart]**2+v[npart1:npart]**2+w[npart1:npart]**2)/2.*abs(q[npart1:npart]/QM[npart1:npart]))]

histEnergyEx=[np.sum(Ex[0:nxn-1,:]**2)/2.*dx*dy]
histEnergyEy=[np.sum(Ey[:,0:nyn-1]**2)/2.*dx*dy]
histEnergyEz=[np.sum(Ez[:,:]**2)/2.*dx*dy]
histEnergyBx=[np.sum(Bx[:,0:nyn-1]**2)/2.*dx*dy]
histEnergyBy=[np.sum(By[0:nxn-1,:]**2)/2.*dx*dy]
histEnergyBz=[np.sum(Bz[0:nxn-1,0:nyn-1]**2)/2.*dx*dy]
histEnergyTot=[histEnergyP1[0]+histEnergyP2[0]+
               histEnergyEx[0]+histEnergyEy[0]+histEnergyEz[0]+
               histEnergyBx[0]+histEnergyBy[0]+histEnergyBz[0]]

energyP[0] = histEnergyP1[0] + histEnergyP2[0]
energyE[0] = histEnergyEx[0] + histEnergyEy[0] + histEnergyEz[0]
energyB[0] = histEnergyBx[0] + histEnergyBy[0] + histEnergyBz[0]

histMomentumx = [np.sum(u[0:npart])]
histMomentumy = [np.sum(v[0:npart])]
histMomentumz = [np.sum(w[0:npart])]
histMomentumTot = [histMomentumx[0] + histMomentumy[0] + histMomentumz[0]]

print('cycle 0, energy=',histEnergyTot[0])
print('energyP1=',histEnergyP1[0],'energyP2=',histEnergyP2[0])
print('energyEx=',histEnergyEx[0],'energyEy=',histEnergyEy[0],'energyEz=',histEnergyEz[0])
print('energyBx=',histEnergyBx[0],'energyBy=',histEnergyBy[0],'energyBz=',histEnergyBz[0])
print('Momentumx=',histMomentumx[0],'Momentumy=',histMomentumy[0],'Momentumz=',histMomentumz[0])

rho = particle_to_grid_rho(x, y, q)
temp=0

if plot_dir == True:
    myplot_particle_map(x, y)
    filename1 = PATH1 + 'part_' + '%04d'%temp + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_phase_space(x, v, limx=(0, Lx), limy=(-2*V0x1, 2*V0x1), xlabel='x', ylabel='vx')
    filename1 = PATH1 + 'phase_' + '%04d'%temp + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_map(xn, yn, Bz, title='B_z', xlabel='x', ylabel='y')
    filename1 = PATH1 + 'Bz_' + '%04d'%temp + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_map(xc, yc, rho, title='rho', xlabel='x', ylabel='y')
    filename1 = PATH1 + 'rho_' + '%04d'%temp + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_map(xc, yc, div(Ex, Ey, Ez, 'E') - rho, title='div(E)-rho map', xlabel='x', ylabel='y')
    filename1 = PATH1 + 'div_rho_' + '%04d'%temp + '.png'
    plt.savefig(filename1, dpi=ndpi)

for it in range(1,nt+1):
    plt.clf()

    guess = phys_to_krylov(Ex,Ey,Ez,u,v,w) 

    #guess = zeros(3*nx*ny+3*npart,np.float64)

    # Uncomment the following to use python's NK methods
    #sol = newton_krylov(residual, guess, method='lgmres', verbose=1, f_tol=1e-14)#, f_rtol=1e-7)
    #print('Residual: %g' % abs(residual(sol)).max())

    # The following is a Picard iteration
    err = 1.
    tol = 1e-14
    kmax = 100
    k=0
    xkrylov = guess
    while err > tol and k<=kmax:
        k+=1
        xkold = xkrylov
        xkrylov = xkrylov - residual(xkrylov)
        err = np.linalg.norm(xkrylov-xkold)
        print(k, err)

    sol = xkrylov
    Exnew, Eynew, Eznew, unew, vnew, wnew = krylov_to_phys(sol)

    if relativistic:
        gnew = np.sqrt(1.+unew**2+vnew**2+wnew**2)
        gold = np.sqrt(1.+u**2+v**2+w**2)
        gbar = (gold+gnew)/2.
    else:
        gbar = np.ones(npart)

    ubar = (unew + u)/2.
    vbar = (vnew + v)/2.
    x += ubar/gbar*dt
    y += vbar/gbar*dt
    x = x%Lx
    y = y%Ly
    u = unew
    v = vnew
    w = wnew

    Exbar = (Exnew+Ex)/2.
    Eybar = (Eynew+Ey)/2.
    Ezbar = (Eznew+Ez)/2.

    curlE_x, curlE_y, curlE_z = curl(Exbar, Eybar, Ezbar,'E')

    Bx = Bx - dt*curlE_x
    By = By - dt*curlE_y
    Bz = Bz - dt*curlE_z
    
    rho = particle_to_grid_rho(x, y, q)
    divE[it] = np.sum(np.abs(div(Exnew, Eynew, Eznew, 'E')) - np.abs(rho))
    divB[it] = np.sum(div(Bx, By, Bz, 'B'))

    Ex = Exnew
    Ey = Eynew
    Ez = Eznew

    if relativistic:
        energyP1 = np.sum((gnew[0:npart1]-1.)*abs(q[0:npart1]/QM[0:npart1]))
        energyP2 = np.sum((gnew[npart1:npart]-1.)*abs(q[npart1:npart]/QM[npart1:npart]))
    else:
        energyP1 = np.sum((u[0:npart1]**2+v[0:npart1]**2+w[0:npart1]**2)/2.*abs(q[0:npart1]/QM[0:npart1]))
        energyP2 = np.sum((u[npart1:npart]**2+v[npart1:npart]**2+w[npart1:npart]**2)/2.*abs(q[npart1:npart]/QM[npart1:npart]))

    energyEx = np.sum(Ex[0:nxn-1,:]**2)/2.*dx*dy
    energyEy = np.sum(Ey[:,0:nyn-1]**2)/2.*dx*dy
    energyEz = np.sum(Ez[:,:]**2)/2.*dx*dy
    energyBx = np.sum(Bx[:,0:nyn-1]**2)/2.*dx*dy
    energyBy = np.sum(By[0:nxn-1,:]**2)/2.*dx*dy
    energyBz = np.sum(Bz[0:nxn-1,0:nyn-1]**2)/2.*dx*dy
    energyTot = energyP1 + energyP2 + energyEx + energyEy + energyEz + energyBx + energyBy + energyBz

    momentumx = np.sum(unew[0:npart])
    momentumy = np.sum(vnew[0:npart])
    momentumz = np.sum(wnew[0:npart])
    momentumTot = momentumx + momentumy + momentumz

    histEnergyP1.append(energyP1)
    histEnergyP2.append(energyP2)
    histEnergyEx.append(energyEx)
    histEnergyEy.append(energyEy)
    histEnergyEz.append(energyEz)
    histEnergyBx.append(energyBx)
    histEnergyBy.append(energyBy)
    histEnergyBz.append(energyBz)
    histEnergyTot.append(energyTot)
    
    histMomentumx.append(momentumx)
    histMomentumy.append(momentumy)
    histMomentumz.append(momentumz)
    histMomentumTot.append(momentumTot)

    energyP[it] = histEnergyP1[it] + histEnergyP2[it]
    energyE[it] = histEnergyEx[it] + histEnergyEy[it] + histEnergyEz[it]
    energyB[it] = histEnergyBx[it] + histEnergyBy[it] + histEnergyBz[it]

    print('cycle',it,'energy =',histEnergyTot[it])
    print('energyP1=',histEnergyP1[it],'energyP2=',histEnergyP2[it])
    print('energyEx=',histEnergyEx[it],'energyEy=',histEnergyEy[it],'energyEz=',histEnergyEz[it])
    print('energyBx=',histEnergyBx[it],'energyBy=',histEnergyBy[it],'energyBz=',histEnergyBz[it])
    print('relative energy change=',(histEnergyTot[it]-histEnergyTot[0])/histEnergyTot[0])
    print('momento totale= ', histMomentumTot[it])

    if plot_each_step == True:
        plt.figure(figsize=(12, 9))

        plt.subplot(2, 3, 1)
        plt.pcolor(xn, yn, Bz)
        plt.title('B_z map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

        #plt.subplot(2, 3, 2)
        #plt.pcolor(xUD, yUD, Ey)
        #plt.title('E_y map')
        #plt.xlabel('x')
        #plt.ylabel('y')
        #plt.colorbar()

        #plt.subplot(2, 3, 3)
        #plt.pcolor(xc, yc, Ez)
        #plt.title('E_z map')
        #plt.xlabel('x')
        #plt.ylabel('y')
        #plt.colorbar()

        plt.subplot(2, 3, 2)
        plt.pcolor(xc, yc, rho)  
        plt.title('rho map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

        plt.subplot(2, 3, 3)
        plt.plot(x[0:npart1], u[0:npart1], 'r.')
        plt.plot(x[npart1:npart], u[npart1:npart], 'b.')
        plt.xlim((0, Lx))
        plt.ylim((-2*V0x1, 2*V0x1))
        plt.title('Phase space')
        plt.xlabel('x')
        plt.ylabel('u')

        #plt.subplot(2, 3, 3)
        #plt.plot(momentumTot)
        #plt.title('Momentum evolution')
        #plt.xlabel('t')
        #plt.ylabel('p')

        plt.subplot(2, 3, 4)
        plt.plot(x[0:npart1], y[0:npart1],'r.')
        plt.plot(x[npart1:npart], y[npart1:npart],'b.')
        plt.xlim((0,Lx))
        plt.ylim((0,Ly))
        plt.title('Particles map')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(2, 3, 5)
        plt.plot((histEnergyTot-histEnergyTot[0])/histEnergyTot[0])#, label='U_tot')
        #plt.plot(energyE+energyB)#, label='U_fields')
        #plt.plot(energyE)#, label='U_elect')
        #plt.plot(energyB)#, label = 'U_mag')
        #plt.plot(energyP, label='U_part')
        plt.title('Energy evolution')
        plt.xlabel('t')
        plt.ylabel('E')
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=2)

        plt.subplot(2, 3, 6)
        plt.plot(divE, label = 'div(E)-rho')
        plt.plot(divB, label = 'div(B)')
        plt.title('Divergence free')
        plt.xlabel('t')
        plt.ylabel('div')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

        filename1 = PATH1 + 'fig_' + '%04d'%it + '.png'
        if (it % every == 0) or (it == 1):
            plt.savefig(filename1, dpi=ndpi)
        plt.pause(0.00000001)

    if plot_dir == True:
        if it == nt-1:
            myplot_func((histEnergyTot-histEnergyTot[0])/histEnergyTot[0], title='Energy', xlabel='t', ylabel='E')
            filename1 = PATH1 + '*energy_tot_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_func(energyB,  title='Energy B', xlabel='t', ylabel='U_mag')
            filename1 = PATH1 + '*energy_mag_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_func(energyE,  title='Energy E', xlabel='t', ylabel='U_el')
            filename1 = PATH1 + '*energy_elec_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_func(energyP,  title='Energy Part.', xlabel='t', ylabel='U_part')
            filename1 = PATH1 + '*energy_part_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_func(histMomentumTot, title='Momentum', xlabel='t', ylabel='p')
            filename1 = PATH1 + '*momentum_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_func(divE, title='div(E)-rho', xlabel='t', ylabel='div')
            filename1 = PATH1 + '*div(E)-rho_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_func(divB, title='div(B)', xlabel='t', ylabel='div')
            filename1 = PATH1 + '*div(B)_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)
  
        if (it % every == 0) or (it == 1):
            myplot_particle_map(x, y)
            filename1 = PATH1 + 'part_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi) 

            myplot_phase_space(x, u, limx=(0, Lx), limy=(-2*V0x1, 2*V0x1), xlabel='x', ylabel='vx')
            filename1 = PATH1 + 'phase_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_map(xn, yn, Bz, title='B_z', xlabel='x', ylabel='y')
            filename1 = PATH1 + 'Bz_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_map(xc, yc, rho, title='rho', xlabel='x', ylabel='y')
            filename1 = PATH1 + 'rho_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_map(xc, yc, div(Exnew, Eynew, Eznew, 'E') - rho, title='div(E)-rho map', xlabel='x', ylabel='y')
            filename1 = PATH1 + 'div_rho_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)
            
    if plot_data == True:
        f = open("iPiC_2D3V_cov_yee.dat", "a")
        print(it, np.sum(Ex), np.sum(Ey), np.sum(Ez), np.sum(Bx), np.sum(By), np.sum(Bz),\
                  energyEx, energyEy, energyEz, energyBx, energyBy, energyBz, energyTot, energyP1, energyP2,\
                  divE[it], divB[it], file=f)
        f.close()
