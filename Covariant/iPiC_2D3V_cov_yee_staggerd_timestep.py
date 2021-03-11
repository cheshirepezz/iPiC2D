"""
Fully Implicit, Relativistic, Covariant Particle-in-Cell - 2D3V Electromagnetic - 2 species
Authors: G. Lapenta, F. Bacchini, J. Croonen and L. Pezzini
Date: 23 Jan 2021
Copyright 2020 KULeuven
MIT License.
"""

import numpy as np
from scipy.optimize import newton_krylov
from numpy import cosh, zeros_like, mgrid, zeros, ones
import matplotlib.pyplot as plt
import time
import sys

#TODO: Build geom needs to be fixed bc we currently use wrong map
#TODO: Fix curl (dual basis and normalisation)

PATH1 = '/Users/luca_pezzini/Documents/Code/cov_pic-2d/figures/'

# metric flag
perturb = True           # perturbed metric tensor
# method flags
NK_method = False
Picard = True
# physics flags
electron_and_ion = True     # background of ions when QM1=QM2=-1
stable_plasma = True        # stable plasma set up
couter_stream_inst = False  # counterstream inst. set up
landau_damping = False      # landau damping set u
relativistic = False        # relativisitc  set up
# plot flags   
log_file = True             # to save the log file in PATH1
plot_dir = True             # to save the plots in PATH1
plot_each_step = False      # to visualise each time step (memory consuming)
plot_data = False           # to plot data in PATH1

# parameters
nx, ny = 20, 20
nxc, nyc = nx, ny
nxn, nyn = nxc+1, nyc+1
Lx, Ly = 10., 10.
dx, dy = Lx/nxc, Ly/nyc
dt = 0.05
nt = 201

ndpi = 100  # number of dpi per img (stay low 100 for monitoring purpose!)
every = 20    # how often to plot
eps = 0.    # amplitude of the pertutbation
n = 1.      # mode of oscillation
B0 = 0.01   # B field perturbation

# Constaint: nppc must be a squerable number (4, 16, 64) because particles are 
#            spread over a squared grid
nppc = 0    # number particles per cell per species
V0 = 1.      # stream velocity magnitude 
alpha = 0.1  # attenuation of VT respect V0

# Species 1
npart1 = nx * ny * nppc
WP1 = 1. # Plasma frequency
QM1 = -1. # Charge/mass ratio
V0x1 = V0 # Stream velocity
V0y1 = V0 # Stream velocity
V0z1 = V0 # Stream velocity
VT1 = alpha*V0 # thermal velocity

# Species 2
npart2 = npart1
WP2 = 1. # Plasma frequency
QM2 = -1. # Charge/mass ratio
V0x2 = V0 # Stream velocity
V0y2 = V0 # Stream velocity
V0z2 = V0 # Stream velocity
VT2 = alpha*V0 # thermal velocity

npart = npart1 + npart2
QM = zeros(npart, np.float64)
QM[0:npart1] = QM1
QM[npart1:npart] = QM2

# INIT PARTICLES
np.random.seed(1)

if nppc==0:
    dxp = 0.
    dyp = 0.
else:
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
    u[0:npart1] = VT1*np.random.randn(npart1)
    u[npart1:npart] = VT2*np.random.randn(npart2)
if couter_stream_inst:
    u[0:npart1] = V0x1+VT1*np.random.randn(npart1)
    u[npart1:npart] = V0x2+VT2*np.random.randn(npart2)
    # velocity in the odd position are negative 
    u[1:npart:2] = - u[1:npart:2]
    # to guarantee 50% of +u0 to e- and the other 50% to e+ and same fo -u0
    np.random.shuffle(u)

v = zeros(npart, np.float64)
v[0:npart1] = VT1*np.random.randn(npart1)
v[npart1:npart] = VT2*np.random.randn(npart2)
if landau_damping:
    v[0:npart1] = V0y1+VT1*np.sin(x[0:npart1]/Lx)
    v[npart1:npart] = V0y2+VT2*np.sin(x[npart1:npart]/Lx)

w = zeros(npart, np.float64)
w[0:npart1] = VT1*np.random.randn(npart1)
w[npart1:npart] = VT2*np.random.randn(npart2)

q = zeros(npart, np.float64)
q[0:npart1] = np.ones(npart1)*WP1**2/(QM1*npart1/Lx/Ly) 
q[npart1:npart] = np.ones(npart2)*WP2**2/(QM2*npart2/Lx/Ly)

if relativistic:
    g = 1./np.sqrt(1.-(u**2+v**2+w**2))
    u = u*g
    v = v*g
    w = w*g

# INIT GRID
# grid of left-right faces LR
xLR, yLR = mgrid[0.:Lx:(nxn*1j), dy/2.:Ly-dy/2.:(nyc*1j)]
# grid of up-down faces UD
xUD, yUD = mgrid[dx/2.:Lx-dx/2.:(nxc*1j), 0.:Ly:(nyn*1j)]
# grid of centres c
xc, yc = mgrid[dx/2.:Lx-dx/2.:(nxc*1j), dy/2.:Ly-dy/2.:(nyc*1j)]
# grid of corners n
xn, yn = mgrid[0.:Lx:(nxn*1j), 0.:Ly:(nyn*1j)]

# INIT FIELDS
# defined on grid LR:        Ex, Jx, By
# defined on grid UD:        Ey, Jy, Bx
# defined on grid centres c: Ez, Jz, rho
# defined on grid corners n: Bz

E1 = zeros(np.shape(xLR),np.float64)
E2 = zeros(np.shape(xUD),np.float64)
E3 = zeros(np.shape(xc),np.float64)
B1 = zeros(np.shape(xUD),np.float64)
B2 = zeros(np.shape(xLR),np.float64)
B3 = zeros(np.shape(xn),np.float64)
#time series
E1time = zeros(nt+1, np.float64)
E2time = zeros(nt+1, np.float64)
E3time = zeros(nt+1, np.float64)
B1time = zeros(nt+1, np.float64)
B2time = zeros(nt+1, np.float64)
B3time = zeros(nt+1, np.float64)

if nppc==0: 
    # delta perturbation of magnetic field
    #B3[int((nx)/2), int((ny)/2)] = B0
    # double sinusoidal perturbation
    B3 = B0 * np.sin(2.*np.pi*n*xn/Lx)*np.sin(2.*np.pi*n*yn/Ly)

rho = zeros(np.shape(xc), np.float64)
rho_ion = zeros(np.shape(xc), np.float64)

# INIT JACOBIAN MATRIX
# defined on grid LR:        (j11e, j21e, j31e)E1, J1
#                            (j11b, j21b, j31b)B1
# defined on grid UD:        (J12e, J22e, J32e)E2, J2
#                            (J12b, J22b, J32b)B2
# defined on grid centres c: (J13e, J23e, J33e)E3, J3
# defined on grid nodes n:   (J13b, J23b, J33b)B3

if perturb:
    # simusoidal perturbation map
    J11_LR = 1. + 2.*np.pi*eps*np.cos(2.*np.pi*xLR/Lx)*np.sin(2.*np.pi*yLR/Ly)/Lx
    J12_LR = 2.*np.pi*eps*np.sin(2.*np.pi*xLR/Lx)*np.cos(2.*np.pi*yLR/Ly)/Ly
    J13_LR = np.zeros(np.shape(xLR), np.float64)
    J21_LR = 2.*np.pi*eps*np.cos(2.*np.pi*xLR/Lx)*np.sin(2.*np.pi*yLR/Ly)/Lx
    J22_LR = 1. + 2.*np.pi*eps*np.sin(2.*np.pi*xLR/Lx)*np.cos(2.*np.pi*yLR/Ly)/Ly
    J23_LR = np.zeros(np.shape(xLR), np.float64)
    J31_LR = np.zeros(np.shape(xLR), np.float64)
    J32_LR = np.zeros(np.shape(xLR), np.float64)
    J33_LR = np.ones(np.shape(xLR), np.float64)

    J11_UD = 1. + 2.*np.pi*eps*np.cos(2.*np.pi*xUD/Lx)*np.sin(2.*np.pi*yUD/Ly)/Lx
    J12_UD = 2.*np.pi*eps*np.sin(2.*np.pi*xUD/Lx)*np.cos(2.*np.pi*yUD/Ly)/Ly
    J13_UD = np.zeros(np.shape(xUD), np.float64)
    J21_UD = 2.*np.pi*eps*np.cos(2.*np.pi*xUD/Lx)*np.sin(2.*np.pi*yUD/Ly)/Lx
    J22_UD = 1. + 2.*np.pi*eps*np.sin(2.*np.pi*xUD/Lx)*np.cos(2.*np.pi*yUD/Ly)/Ly
    J23_UD = np.zeros(np.shape(xUD), np.float64)
    J31_UD = np.zeros(np.shape(xUD), np.float64)
    J32_UD = np.zeros(np.shape(xUD), np.float64)
    J33_UD = np.ones(np.shape(xUD), np.float64)

    J11_C = 1. + 2.*np.pi*eps*np.cos(2.*np.pi*xc/Lx)*np.sin(2.*np.pi*yc/Ly)/Lx
    J12_C = 2.*np.pi*eps*np.sin(2.*np.pi*xc/Lx)*np.cos(2.*np.pi*yc/Ly)/Ly
    J13_C = np.zeros(np.shape(xc), np.float64)
    J21_C = 2.*np.pi*eps*np.cos(2.*np.pi*xc/Lx)*np.sin(2.*np.pi*yc/Ly)/Lx
    J22_C = 1. + 2.*np.pi*eps*np.sin(2.*np.pi*xc/Lx)*np.cos(2.*np.pi*yc/Ly)/Ly
    J23_C = np.zeros(np.shape(xc), np.float64)
    J31_C = np.zeros(np.shape(xc), np.float64)
    J32_C = np.zeros(np.shape(xc), np.float64)
    J33_C = np.ones(np.shape(xc), np.float64)

    J11_N = 1. + 2.*np.pi*eps*np.cos(2.*np.pi*xn/Lx)*np.sin(2.*np.pi*yn/Ly)/Lx
    J12_N = 2.*np.pi*eps*np.sin(2.*np.pi*xn/Lx)*np.cos(2.*np.pi*yn/Ly)/Ly
    J13_N = np.zeros(np.shape(xn), np.float64)
    J21_N = 2.*np.pi*eps*np.cos(2.*np.pi*xn/Lx)*np.sin(2.*np.pi*yn/Ly)/Lx
    J22_N = 1. + 2.*np.pi*eps*np.sin(2.*np.pi*xn/Lx)*np.cos(2.*np.pi*yn/Ly)/Ly
    J23_N = np.zeros(np.shape(xn), np.float64)
    J31_N = np.zeros(np.shape(xn), np.float64)
    J32_N = np.zeros(np.shape(xn), np.float64)
    J33_N = np.ones(np.shape(xn), np.float64)
else:
    # identity matrix 
    J11_LR = np.ones(np.shape(xLR), np.float64)
    J12_LR = np.zeros(np.shape(xLR), np.float64)
    J13_LR = np.zeros(np.shape(xLR), np.float64)
    J21_LR = np.zeros(np.shape(xLR), np.float64)
    J22_LR = np.ones(np.shape(xLR), np.float64)
    J23_LR = np.zeros(np.shape(xLR), np.float64)
    J31_LR = np.zeros(np.shape(xLR), np.float64)
    J32_LR = np.zeros(np.shape(xLR), np.float64)
    J33_LR = np.ones(np.shape(xLR), np.float64)

    J11_UD = np.ones(np.shape(xUD), np.float64)
    J12_UD = np.zeros(np.shape(xUD), np.float64)
    J13_UD = np.zeros(np.shape(xUD), np.float64)
    J21_UD = np.zeros(np.shape(xUD), np.float64)
    J22_UD = np.ones(np.shape(xUD), np.float64)
    J23_UD = np.zeros(np.shape(xUD), np.float64)
    J31_UD = np.zeros(np.shape(xUD), np.float64)
    J32_UD = np.zeros(np.shape(xUD), np.float64)
    J33_UD = np.ones(np.shape(xUD), np.float64)

    J11_C = np.ones(np.shape(xc), np.float64)
    J12_C = np.zeros(np.shape(xc), np.float64)
    J13_C = np.zeros(np.shape(xc), np.float64)
    J21_C = np.zeros(np.shape(xc), np.float64)
    J22_C = np.ones(np.shape(xc), np.float64)
    J23_C = np.zeros(np.shape(xc), np.float64)
    J31_C = np.zeros(np.shape(xc), np.float64)
    J32_C = np.zeros(np.shape(xc), np.float64)
    J33_C = np.ones(np.shape(xc), np.float64)

    J11_N = np.ones(np.shape(xn), np.float64)
    J12_N = np.zeros(np.shape(xn), np.float64)
    J13_N = np.zeros(np.shape(xn), np.float64)
    J21_N = np.zeros(np.shape(xn), np.float64)
    J22_N = np.ones(np.shape(xn), np.float64)
    J23_N = np.zeros(np.shape(xn), np.float64)
    J31_N = np.zeros(np.shape(xn), np.float64)
    J32_N = np.zeros(np.shape(xn), np.float64)
    J33_N = np.ones(np.shape(xn), np.float64)
    
# INIT JACOBIAN DETERMINANT

J_UD = np.zeros(np.shape(xUD), np.float64)
J_LR = np.zeros(np.shape(xLR), np.float64)
J_C = np.zeros(np.shape(xc), np.float64)
J_N = np.zeros(np.shape(xn), np.float64)

# DEFINE INVERSE JACOBIAN MATRIX
# defined on grid LR:        (j11, j21, j31)E1, J1
#                            (j11, j21, j31)B1
# defined on grid UD:        (j12, j22, j32)E2, J2
#                            (j12, j22, j32)B2
# defined on grid centres c: (j12, j23, j33)E3, J3
# defined on grid nodes n:   (j12, j23, j33)B3

j11_LR = np.zeros(np.shape(xLR), np.float64)
j12_LR = np.zeros(np.shape(xLR), np.float64)
j13_LR = np.zeros(np.shape(xLR), np.float64)
j21_LR = np.zeros(np.shape(xLR), np.float64)
j22_LR = np.zeros(np.shape(xLR), np.float64)
j23_LR = np.zeros(np.shape(xLR), np.float64)
j31_LR = np.zeros(np.shape(xLR), np.float64)
j32_LR = np.zeros(np.shape(xLR), np.float64)
j33_LR = np.zeros(np.shape(xLR), np.float64)

j11_UD = np.zeros(np.shape(xUD), np.float64)
j12_UD = np.zeros(np.shape(xUD), np.float64)
j13_UD = np.zeros(np.shape(xUD), np.float64)
j21_UD = np.zeros(np.shape(xUD), np.float64)
j22_UD = np.zeros(np.shape(xUD), np.float64)
j23_UD = np.zeros(np.shape(xUD), np.float64)
j31_UD = np.zeros(np.shape(xUD), np.float64)
j32_UD = np.zeros(np.shape(xUD), np.float64)
j33_UD = np.zeros(np.shape(xUD), np.float64)

j11_C = np.zeros(np.shape(xc), np.float64)
j12_C = np.zeros(np.shape(xc), np.float64)
j13_C = np.zeros(np.shape(xc), np.float64)
j21_C = np.zeros(np.shape(xc), np.float64)
j22_C = np.zeros(np.shape(xc), np.float64)
j23_C = np.zeros(np.shape(xc), np.float64)
j31_C = np.zeros(np.shape(xc), np.float64)
j32_C = np.zeros(np.shape(xc), np.float64)
j33_C = np.zeros(np.shape(xc), np.float64)

j11_N = np.zeros(np.shape(xn), np.float64)
j12_N = np.zeros(np.shape(xn), np.float64)
j13_N = np.zeros(np.shape(xn), np.float64)
j21_N = np.zeros(np.shape(xn), np.float64)
j22_N = np.zeros(np.shape(xn), np.float64)
j23_N = np.zeros(np.shape(xn), np.float64)
j31_N = np.zeros(np.shape(xn), np.float64)
j32_N = np.zeros(np.shape(xn), np.float64)
j33_N = np.zeros(np.shape(xn), np.float64)

# INIT IVERSE JACOBIAN DETERMINANT

j_LR = np.zeros(np.shape(xLR), np.float64)
j_UD = np.zeros(np.shape(xUD), np.float64)
j_C = np.zeros(np.shape(xc), np.float64)
j_N = np.zeros(np.shape(xn), np.float64)

# INIT METRIC TENSOR
# defined on grid LR:        (g11, g21, g31)E1
#                            (g11, g21, g31)B1
# defined on grid UD:        (g12, g22, g32)E2
#                            (g12, g22, g32)B2
# defined on grid centres c: (g12, g23, g33)E3
# defined on grid corners n: (g12, g23, g33)B3

g11_LR = np.zeros(np.shape(xLR), np.float64)
g12_LR = np.zeros(np.shape(xLR), np.float64)
g13_LR = np.zeros(np.shape(xLR), np.float64)
g21_LR = np.zeros(np.shape(xLR), np.float64)
g22_LR = np.zeros(np.shape(xLR), np.float64)
g23_LR = np.zeros(np.shape(xLR), np.float64)
g31_LR = np.zeros(np.shape(xLR), np.float64)
g32_LR = np.zeros(np.shape(xLR), np.float64)
g33_LR = np.zeros(np.shape(xLR), np.float64)

g11_UD = np.zeros(np.shape(xUD), np.float64)
g12_UD = np.zeros(np.shape(xUD), np.float64)
g13_UD = np.zeros(np.shape(xUD), np.float64)
g21_UD = np.zeros(np.shape(xUD), np.float64)
g22_UD = np.zeros(np.shape(xUD), np.float64)
g23_UD = np.zeros(np.shape(xUD), np.float64)
g31_UD = np.zeros(np.shape(xUD), np.float64)
g32_UD = np.zeros(np.shape(xUD), np.float64)
g33_UD = np.zeros(np.shape(xUD), np.float64)

g11_C = np.zeros(np.shape(xc), np.float64)
g12_C = np.zeros(np.shape(xc), np.float64)
g13_C = np.zeros(np.shape(xc), np.float64)
g21_C = np.zeros(np.shape(xc), np.float64)
g22_C = np.zeros(np.shape(xc), np.float64)
g23_C = np.zeros(np.shape(xc), np.float64)
g31_C = np.zeros(np.shape(xc), np.float64)
g32_C = np.zeros(np.shape(xc), np.float64)
g33_C = np.zeros(np.shape(xc), np.float64)

g11_N = np.zeros(np.shape(xn), np.float64)
g12_N = np.zeros(np.shape(xn), np.float64)
g13_N = np.zeros(np.shape(xn), np.float64)
g21_N = np.zeros(np.shape(xn), np.float64)
g22_N = np.zeros(np.shape(xn), np.float64)
g23_N = np.zeros(np.shape(xn), np.float64)
g31_N = np.zeros(np.shape(xn), np.float64)
g32_N = np.zeros(np.shape(xn), np.float64)
g33_N = np.zeros(np.shape(xn), np.float64)

# Divergence
# defined on grid c:
divE = zeros(nt, np.float64)
divE_rho = zeros(nt, np.float64)
# defined on grid n: 
divB = zeros(nt, np.float64)

# Energy
energyP = zeros(nt, np.float64) # particles
energyP1 = zeros(nt, np.float64) # particles1
energyP2 = zeros(nt, np.float64)# particles2
energyE = zeros(nt, np.float64) # Total E
energyE1 = zeros(nt, np.float64) # E1 field
energyE2 = zeros(nt, np.float64) # E2 field
energyE3 = zeros(nt, np.float64) # E3 field
energyB = zeros(nt, np.float64) # B field
energyB1 = zeros(nt, np.float64) # B1 field
energyB2 = zeros(nt, np.float64) # B2 field
energyB3 = zeros(nt, np.float64) # B3 field
energyTot = zeros(nt, np.float64) # B field
err_en = zeros(nt, np.float64) # energy error
momentumTot = zeros(nt, np.float64) # Total momentum

if log_file == True:
    f = open(PATH1 + 'log_file.txt', 'w')
    print('iPiC_2D3V_cov_yee_staggerd_timestep.py', file=f)
    print('* METRIC:', file=f)
    print('- perturbation: ', perturb, file=f)
    print('* METHOD:', file=f)
    print('- NK method: ', NK_method, file=f)
    print('- Picard iteration: ', Picard, file=f)
    print('* PHYSICS:', file=f)
    print('- perturbation amplitude B0 (if nppc=0): ', B0, file=f)
    print('- mode of oscillation (if nppc=0): ', n, file=f)
    print('- stable plasma: ', stable_plasma, file=f)
    print('- electrons & ions: ', electron_and_ion, file=f)
    print('- counter stream inst.: ', couter_stream_inst, file=f)
    print('- landau damping: ', landau_damping, file=f)
    print('- relativistic: ', relativistic, file=f)
    print('* PARAMETER:', file=f)
    print('- number nodes (x-axes): ', nx, file=f)
    print('- number nodes (y-axes): ', ny, file=f)
    print('- length of the domain (x-axes): ', Lx, file=f)
    print('- length of the domain (y-axes): ', Ly, file=f)
    print('- time steps: ', dt, file=f)
    print('- number of time steps: ', nt, file=f)
    print('- number of part. per cell: ', nppc, file=f) 
    print('* SPECIES 1:', file=f)
    print('- number of particles : ', npart1, file=f)
    print('- plasma frequency : ', WP1, file=f)
    print('- charge to mass : ', QM1, file=f)
    print('- velocity field: ', '(', V0x1, ',', V0y1, ',', V0z1, ')', file=f)
    print('- thermal velocity: ', VT1, file=f)
    print('* SPECIES 2:', file=f)
    print('- number of particles : ', npart2, file=f)
    print('- plasma frequency : ', WP2, file=f)
    print('- charge to mass : ', QM2, file=f)
    print('- velocity field: ', '(', V0x2, ',', V0y2, ',', V0z2, ')', file=f)
    print('- thermal velocity: ', VT2, file=f)
    f.close()

def myplot_map(xgrid, ygrid, field, title='a', xlabel='b', ylabel='c'):
    '''
    To plot the map of a vector fied over a grid.
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
    '''
    To plot particles position over the domain.
    '''
    plt.figure()
    plt.plot(posx[0:npart1],posy[0:npart1],'b.')
    plt.plot(posx[npart1:npart],posy[npart1:npart],'r.')
    plt.xlim((0,Lx))
    plt.ylim((0,Ly))
    plt.title('Particles map')
    plt.xlabel('x')
    plt.ylabel('y')

def myplot_phase_space(pos, vel, limx=(0, 0), limy=(0, 0), xlabel='b', ylabel='c'):
    '''To plot the phase space in one direction
    '''
    plt.figure()
    plt.plot(pos[0:npart1], vel[0:npart1], 'b.')
    plt.plot(pos[npart1:npart], vel[npart1:npart], 'r.')
    plt.xlim(limx)
    plt.ylim(limy)
    plt.title('Particles map')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def define_geometry():
    '''To construct the structure of the general geometry (for each grid type):
    - Get the Jacobian matrix and its determinant
    - Get the inverse Jacobian matrix isolate the components and calculate its determinant
    - Get the metric tensor components
    '''
    for i in range(np.shape(xLR)[0]):
        for j in range(np.shape(xLR)[1]):
            jacobian_LR = np.array([[J11_LR[i, j], J12_LR[i, j], J13_LR[i, j]], [J21_LR[i, j], J22_LR[i, j], J23_LR[i, j]], [J31_LR[i, j], J32_LR[i, j], J33_LR[i, j]]])
            J_LR[i, j] = np.linalg.det(jacobian_LR)
            J11_LR[i, j] = jacobian_LR[0, 0]
            J21_LR[i, j] = jacobian_LR[1, 0]
            J31_LR[i, j] = jacobian_LR[2, 0]
            J12_LR[i, j] = jacobian_LR[0, 1]
            J22_LR[i, j] = jacobian_LR[1, 1]
            J32_LR[i, j] = jacobian_LR[2, 1]
            J13_LR[i, j] = jacobian_LR[0, 2]
            J23_LR[i, j] = jacobian_LR[1, 2]
            J33_LR[i, j] = jacobian_LR[2, 2]
            inverse_jacobian_LR = np.linalg.inv(jacobian_LR)
            j11_LR[i, j] = inverse_jacobian_LR[0, 0]
            j21_LR[i, j] = inverse_jacobian_LR[1, 0]
            j31_LR[i, j] = inverse_jacobian_LR[2, 0]
            j12_LR[i, j] = inverse_jacobian_LR[0, 1]
            j22_LR[i, j] = inverse_jacobian_LR[1, 1]
            j32_LR[i, j] = inverse_jacobian_LR[2, 1]
            j13_LR[i, j] = inverse_jacobian_LR[0, 2]
            j23_LR[i, j] = inverse_jacobian_LR[1, 2]
            j33_LR[i, j] = inverse_jacobian_LR[2, 2]
            j_LR[i, j] = np.linalg.det(inverse_jacobian_LR)
            g11_LR[i, j] = jacobian_LR[0, 0] * jacobian_LR[0, 0] + jacobian_LR[1, 0] * jacobian_LR[1, 0] + jacobian_LR[2, 0] * jacobian_LR[2, 0]
            g21_LR[i, j] = jacobian_LR[0, 0] * jacobian_LR[0, 1] + jacobian_LR[1, 0] * jacobian_LR[1, 1] + jacobian_LR[2, 0] * jacobian_LR[2, 1]
            g31_LR[i, j] = jacobian_LR[0, 0] * jacobian_LR[0, 2] + jacobian_LR[1, 0] * jacobian_LR[1, 2] + jacobian_LR[2, 0] * jacobian_LR[2, 2]
            g12_LR[i, j] = jacobian_LR[0, 0] * jacobian_LR[0, 1] + jacobian_LR[1, 0] * jacobian_LR[1, 1] + jacobian_LR[2, 0] * jacobian_LR[2, 1]
            g22_LR[i, j] = jacobian_LR[0, 1] * jacobian_LR[0, 1] + jacobian_LR[1, 1] * jacobian_LR[1, 1] + jacobian_LR[2, 1] * jacobian_LR[2, 1]
            g32_LR[i, j] = jacobian_LR[0, 1] * jacobian_LR[0, 2] + jacobian_LR[1, 1] * jacobian_LR[1, 2] + jacobian_LR[2, 1] * jacobian_LR[2, 2]
            g13_LR[i, j] = jacobian_LR[0, 0] * jacobian_LR[0, 2] + jacobian_LR[1, 0] * jacobian_LR[1, 2] + jacobian_LR[2, 0] * jacobian_LR[2, 2]
            g23_LR[i, j] = jacobian_LR[0, 1] * jacobian_LR[0, 2] + jacobian_LR[1, 1] * jacobian_LR[1, 2] + jacobian_LR[2, 1] * jacobian_LR[2, 2]
            g33_LR[i, j] = jacobian_LR[0, 2] * jacobian_LR[0, 2] + jacobian_LR[1, 2] * jacobian_LR[1, 2] + jacobian_LR[2, 2] * jacobian_LR[2, 2]
           
    for i in range(np.shape(xUD)[0]):
        for j in range(np.shape(xUD)[1]):
            jacobian_UD = np.array([[J11_UD[i, j], J12_UD[i, j], J13_UD[i, j]], [J21_UD[i, j], J22_UD[i, j], J23_UD[i, j]], [J31_UD[i, j], J32_UD[i, j], J33_UD[i, j]]])
            J_UD[i, j] = np.linalg.det(jacobian_UD)
            J11_UD[i, j] = jacobian_UD[0, 0]
            J21_UD[i, j] = jacobian_UD[1, 0]
            J31_UD[i, j] = jacobian_UD[2, 0]
            J12_UD[i, j] = jacobian_UD[0, 1]
            J22_UD[i, j] = jacobian_UD[1, 1]
            J32_UD[i, j] = jacobian_UD[2, 1]
            J13_UD[i, j] = jacobian_UD[0, 2]
            J23_UD[i, j] = jacobian_UD[1, 2]
            J33_UD[i, j] = jacobian_UD[2, 2]
            inverse_jacobian_UD = np.linalg.inv(jacobian_UD)
            j_UD[i, j] = np.linalg.det(inverse_jacobian_UD)
            j11_UD[i, j] = inverse_jacobian_UD[0, 0]
            j21_UD[i, j] = inverse_jacobian_UD[1, 0]
            j31_UD[i, j] = inverse_jacobian_UD[2, 0]
            j12_UD[i, j] = inverse_jacobian_UD[0, 1]
            j22_UD[i, j] = inverse_jacobian_UD[1, 1]
            j32_UD[i, j] = inverse_jacobian_UD[2, 1]
            j13_UD[i, j] = inverse_jacobian_UD[0, 2]
            j23_UD[i, j] = inverse_jacobian_UD[1, 2]
            j33_UD[i, j] = inverse_jacobian_UD[2, 2]
            g11_UD[i, j] = jacobian_UD[0, 0] * jacobian_UD[0, 0] + jacobian_UD[1, 0] * jacobian_UD[1, 0] + jacobian_UD[2, 0] * jacobian_UD[2, 0]
            g21_UD[i, j] = jacobian_UD[0, 0] * jacobian_UD[0, 1] + jacobian_UD[1, 0] * jacobian_UD[1, 1] + jacobian_UD[2, 0] * jacobian_UD[2, 1]
            g31_UD[i, j] = jacobian_UD[0, 0] * jacobian_UD[0, 2] + jacobian_UD[1, 0] * jacobian_UD[1, 2] + jacobian_UD[2, 0] * jacobian_UD[2, 2] 
            g12_UD[i, j] = jacobian_UD[0, 0] * jacobian_UD[0, 1] + jacobian_UD[1, 0] * jacobian_UD[1, 1] + jacobian_UD[2, 0] * jacobian_UD[2, 1]
            g22_UD[i, j] = jacobian_UD[0, 1] * jacobian_UD[0, 1] + jacobian_UD[1, 1] * jacobian_UD[1, 1] + jacobian_UD[2, 1] * jacobian_UD[2, 1]
            g32_UD[i, j] = jacobian_UD[0, 1] * jacobian_UD[0, 2] + jacobian_UD[1, 1] * jacobian_UD[1, 2] + jacobian_UD[2, 1] * jacobian_UD[2, 2]
            g13_UD[i, j] = jacobian_UD[0, 0] * jacobian_UD[0, 2] + jacobian_UD[1, 0] * jacobian_UD[1, 2] + jacobian_UD[2, 0] * jacobian_UD[2, 2]
            g23_UD[i, j] = jacobian_UD[0, 1] * jacobian_UD[0, 2] + jacobian_UD[1, 1] * jacobian_UD[1, 2] + jacobian_UD[2, 1] * jacobian_UD[2, 2]
            g33_UD[i, j] = jacobian_UD[0, 2] * jacobian_UD[0, 2] + jacobian_UD[1, 2] * jacobian_UD[1, 2] + jacobian_UD[2, 2] * jacobian_UD[2, 2]

    for i in range(np.shape(xc)[0]):
        for j in range(np.shape(xc)[1]):
            jacobian_C = np.array([[J11_C[i, j], J12_C[i, j], J13_C[i, j]], [J21_C[i, j], J22_C[i, j], J23_C[i, j]], [J31_C[i, j], J32_C[i, j], J33_C[i, j]]])
            J_C[i, j] = np.linalg.det(jacobian_C)
            J11_C[i, j] = jacobian_C[0, 0]
            J21_C[i, j] = jacobian_C[1, 0]
            J31_C[i, j] = jacobian_C[2, 0]
            J12_C[i, j] = jacobian_C[0, 1]
            J22_C[i, j] = jacobian_C[1, 1]
            J32_C[i, j] = jacobian_C[2, 1]
            J13_C[i, j] = jacobian_C[0, 2]
            J23_C[i, j] = jacobian_C[1, 2]
            J33_C[i, j] = jacobian_C[2, 2]
            inverse_jacobian_C = np.linalg.inv(jacobian_C)
            j_C[i, j] = np.linalg.det(jacobian_C)
            j11_C[i, j] = inverse_jacobian_C[0, 0]
            j21_C[i, j] = inverse_jacobian_C[1, 0]
            j31_C[i, j] = inverse_jacobian_C[2, 0]
            j12_C[i, j] = inverse_jacobian_C[0, 1]
            j22_C[i, j] = inverse_jacobian_C[1, 1]
            j32_C[i, j] = inverse_jacobian_C[2, 1]
            j13_C[i, j] = inverse_jacobian_C[0, 2]
            j23_C[i, j] = inverse_jacobian_C[1, 2]
            j33_C[i, j] = inverse_jacobian_C[2, 2]
            g11_C[i, j] = jacobian_C[0, 0] * jacobian_C[0, 0] + jacobian_C[1, 0] * jacobian_C[1, 0] + jacobian_C[2, 0] * jacobian_C[2, 0]
            g21_C[i, j] = jacobian_C[0, 0] * jacobian_C[0, 1] + jacobian_C[1, 0] * jacobian_C[1, 1] + jacobian_C[2, 0] * jacobian_C[2, 1]
            g31_C[i, j] = jacobian_C[0, 0] * jacobian_C[0, 2] + jacobian_C[1, 0] * jacobian_C[1, 2] + jacobian_C[2, 0] * jacobian_C[2, 2] 
            g12_C[i, j] = jacobian_C[0, 0] * jacobian_C[0, 1] + jacobian_C[1, 0] * jacobian_C[1, 1] + jacobian_C[2, 0] * jacobian_C[2, 1]
            g22_C[i, j] = jacobian_C[0, 1] * jacobian_C[0, 1] + jacobian_C[1, 1] * jacobian_C[1, 1] + jacobian_C[2, 1] * jacobian_C[2, 1]
            g32_C[i, j] = jacobian_C[0, 1] * jacobian_C[0, 2] + jacobian_C[1, 1] * jacobian_C[1, 2] + jacobian_C[2, 1] * jacobian_C[2, 2]
            g13_C[i, j] = jacobian_C[0, 0] * jacobian_C[0, 2] + jacobian_C[1, 0] * jacobian_C[1, 2] + jacobian_C[2, 0] * jacobian_C[2, 2]
            g23_C[i, j] = jacobian_C[0, 1] * jacobian_C[0, 2] + jacobian_C[1, 1] * jacobian_C[1, 2] + jacobian_C[2, 1] * jacobian_C[2, 2]
            g33_C[i, j] = jacobian_C[0, 2] * jacobian_C[0, 2] + jacobian_C[1, 2] * jacobian_C[1, 2] + jacobian_C[2, 2] * jacobian_C[2, 2]

    for i in range(np.shape(xn)[0]):
        for j in range(np.shape(xn)[1]):
            jacobian_N = np.array([[J11_N[i, j], J12_N[i, j], J13_N[i, j]], [J21_N[i, j], J22_N[i, j], J23_N[i, j]], [J31_N[i, j], J32_N[i, j], J33_N[i, j]]])
            J_N[i, j] = np.linalg.det(jacobian_N)
            J11_N[i, j] = jacobian_N[0, 0]
            J21_N[i, j] = jacobian_N[1, 0]
            J31_N[i, j] = jacobian_N[2, 0]
            J12_N[i, j] = jacobian_N[0, 1]
            J22_N[i, j] = jacobian_N[1, 1]
            J32_N[i, j] = jacobian_N[2, 1]
            J13_N[i, j] = jacobian_N[0, 2]
            J23_N[i, j] = jacobian_N[1, 2]
            J33_N[i, j] = jacobian_N[2, 2]
            inverse_jacobian_N = np.linalg.inv(jacobian_N)
            j_N[i, j] = np.linalg.det(jacobian_N)
            j11_N[i, j] = inverse_jacobian_N[0, 0]
            j21_N[i, j] = inverse_jacobian_N[1, 0]
            j31_N[i, j] = inverse_jacobian_N[2, 0]
            j12_N[i, j] = inverse_jacobian_N[0, 1]
            j22_N[i, j] = inverse_jacobian_N[1, 1]
            j32_N[i, j] = inverse_jacobian_N[2, 1]
            j13_N[i, j] = inverse_jacobian_N[0, 2]
            j23_N[i, j] = inverse_jacobian_N[1, 2]
            j33_N[i, j] = inverse_jacobian_N[2, 2]
            g11_N[i, j] = jacobian_N[0, 0] * jacobian_N[0, 0] + jacobian_N[1, 0] * jacobian_N[1, 0] + jacobian_N[2, 0] * jacobian_N[2, 0]
            g21_N[i, j] = jacobian_N[0, 0] * jacobian_N[0, 1] + jacobian_N[1, 0] * jacobian_N[1, 1] + jacobian_N[2, 0] * jacobian_N[2, 1]
            g31_N[i, j] = jacobian_N[0, 0] * jacobian_N[0, 2] + jacobian_N[1, 0] * jacobian_N[1, 2] + jacobian_N[2, 0] * jacobian_N[2, 2] 
            g12_N[i, j] = jacobian_N[0, 0] * jacobian_N[0, 1] + jacobian_N[1, 0] * jacobian_N[1, 1] + jacobian_N[2, 0] * jacobian_N[2, 1]
            g22_N[i, j] = jacobian_N[0, 1] * jacobian_N[0, 1] + jacobian_N[1, 1] * jacobian_N[1, 1] + jacobian_N[2, 1] * jacobian_N[2, 1]
            g32_N[i, j] = jacobian_N[0, 1] * jacobian_N[0, 2] + jacobian_N[1, 1] * jacobian_N[1, 2] + jacobian_N[2, 1] * jacobian_N[2, 2]
            g13_N[i, j] = jacobian_N[0, 0] * jacobian_N[0, 2] + jacobian_N[1, 0] * jacobian_N[1, 2] + jacobian_N[2, 0] * jacobian_N[2, 2]
            g23_N[i, j] = jacobian_N[0, 1] * jacobian_N[0, 2] + jacobian_N[1, 1] * jacobian_N[1, 2] + jacobian_N[2, 1] * jacobian_N[2, 2]
            g33_N[i, j] = jacobian_N[0, 2] * jacobian_N[0, 2] + jacobian_N[1, 2] * jacobian_N[1, 2] + jacobian_N[2, 2] * jacobian_N[2, 2]

def norm(vecx, vecy, vecz):
    '''To calculate the norm of a covariant/contravariant/ordinary vector
    '''
    return np.sqrt(vecx**2 + vecy**2 + vecz**2)

def cartesian_to_general(cartx, carty, cartz, fieldtype):
    ''' To convert fields from Cartesian coord. (x, y, z) to General coord. (xi, eta, zeta)
    fieltype=='E' or 'J': input -> LR,UD,c, output -> LR,UD,c
    fieltype=='B':        input -> UD,LR,n, output -> UD,LR,n
    '''
    if (fieldtype == 'E') or (fieldtype == 'J'):
      genx1 = J11_LR * cartx + J11_LR * avg(avg(carty, 'UD2C'), 'C2LR')+ J11_LR * avg(cartz, 'C2LR')
      genx2 = J22_UD * avg(avg(cartx, 'LR2C'), 'C2UD') + J22_UD * carty + J22_UD * avg(cartz, 'C2UD')
      genx3 = J33_C * avg(cartx, 'LR2C') + J33_C * avg(carty, 'UD2C') + J33_C * cartz
    elif fieldtype == 'B':
      genx1 = J11_UD * cartx + J11_UD * avg(avg(carty, 'LR2C'), 'C2UD') + J11_UD * avg(cartz, 'N2UD')
      genx2 = J22_LR * avg(avg(cartx, 'UD2C'), 'C2LR') + J22_LR * carty + J22_LR * avg(cartz, 'N2LR')
      genx3 = J33_N * avg(cartx, 'UD2N') + J33_N * avg(carty, 'LR2N') + J33_N * cartz
    
    return genx1, genx2, genx3

def general_to_cartesian(genx1, genx2, genx3, fieldtype):
    ''' To convert fields from General coord. (xi, eta, zeta) to Cartesian coord (x, y, z)
    fieltype=='E' or 'J': input -> LR,UD,c, output -> LR,UD,c
    fieltype=='B':        input -> UD,LR,n, output -> UD,LR,n
    '''
    if (fieldtype == 'E') or (fieldtype == 'J'):
      cartx = j11_LR * genx1 + j11_LR * avg(avg(genx2, 'UD2C'), 'C2LR')+ j11_LR * avg(genx3, 'C2LR')
      carty = j22_UD * avg(avg(genx1, 'LR2C'), 'C2UD') + j22_UD * genx2 + j22_UD * avg(genx3, 'C2UD')
      cartz = j33_C * avg(genx1, 'LR2C') + j33_C * avg(genx2, 'UD2C') + j33_C * genx3
    elif fieldtype == 'B':
      cartx = j11_UD * genx1 + j11_UD * avg(avg(genx2, 'LR2C'), 'C2UD') + j11_UD * avg(genx3, 'N2UD')
      carty = j22_LR * avg(avg(genx1, 'UD2C'), 'C2LR') + j22_LR * genx2 + j22_LR * avg(genx3, 'N2LR')
      cartz = j33_N * avg(genx1, 'UD2N') + j33_N * avg(genx2, 'LR2N') + j33_N * genx3
    
    return cartx, carty, cartz

def cartesian_to_general_particle(cartx, carty):
    '''To convert the particles position from Cartesian coord. (x, y, z) to General coord. (xi, eta, zeta)
    '''
    genx1 = cartx + eps*np.sin(2*np.pi*cartx/Lx)*np.sin(2*np.pi*carty/Ly)
    genx2 = carty + eps*np.sin(2*np.pi*cartx/Lx)*np.sin(2*np.pi*carty/Ly)

    return genx1, genx2

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

    fieldN[1:nx-1,1:ny-1] = (fieldC[0:nx-2,0:ny-2]+fieldC[1:nx-1,0:ny-2]+fieldC[0:nx-2,1:ny-1]+fieldC[1:nx-1,1:ny-1])/4.
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

def curl_normalised(fieldx, fieldy, fieldz, fieldtype):
    ''' To take the curl of either E or B in Skew coordinate and normalise the covector to vector (coV = V/||x_xi||)
    curl^i = 1/J·(d_j·g_kq·A^q - d_k·g_jq·A^q)
    fieltype=='E': input -> LR,UD,c, output -> UD,LR,n
    fieltype=='B': input -> UD,LR,n, output -> LR,UD,c
    '''
    # normalisation of x_\xi^1:  
    nx1_C = norm(J11_C, J21_C, J31_C)
    nx1_UD = norm(J11_UD, J21_UD, J31_UD)
    nx1_LR = norm(J11_LR, J21_LR, J31_LR)
    nx1_N = norm(J11_N, J21_N, J31_N)
    # normalisation of x_\xi^2:
    nx2_C = norm(J12_C, J22_C, J32_C)
    nx2_UD = norm(J12_UD, J22_UD, J32_UD)
    nx2_LR = norm(J12_LR, J22_LR, J32_LR)
    nx2_N = norm(J12_N, J22_N, J32_N)
    # normalisation of x_\xi^3:
    nx3_C = norm(J13_C, J23_C, J33_C)
    nx3_UD = norm(J13_UD, J23_UD, J33_UD)
    nx3_LR = norm(J13_LR, J23_LR, J33_LR)
    nx3_N = norm(J13_N, J23_N, J33_N)

    if fieldtype == 'E':
        fieldx_C = avg(fieldx, 'LR2C')
        fieldy_C = avg(fieldy, 'UD2C')
        fieldx_UD = avg(avg(fieldx, 'LR2C'), 'C2UD')
        fieldz_UD = avg(fieldz, 'C2UD')
        fieldy_LR = avg(avg(fieldy, 'UD2C'), 'C2LR')
        fieldz_LR = avg(fieldz, 'C2LR')

        curl_x =   dirder(g31_C * fieldx_C / nx1_C + g32_C * fieldy_C / nx2_C + g33_C * fieldz / nx3_C, 'C2UD')/J_UD*nx1_UD
        curl_y = - dirder(g31_C * fieldx_C / nx1_C + g32_C * fieldy_C / nx2_C + g33_C * fieldz / nx3_C, 'C2LR')/J_LR*nx2_LR
        curl_z =   dirder(g21_UD * fieldx_UD / nx1_UD + g22_UD * fieldy / nx2_UD + g23_UD * fieldz_UD / nx3_UD, 'UD2N')/J_N*nx3_N\
                 - dirder(g11_LR * fieldx / nx1_LR + g12_LR * fieldy_LR / nx2_LR + g13_LR * fieldz_LR / nx3_LR, 'LR2N')/J_N*nx3_N
    elif fieldtype == 'B':
        fieldx_N = avg(fieldx, 'UD2N')
        fieldy_N = avg(fieldy, 'LR2N')
        fieldx_LR = avg(avg(fieldx, 'UD2N'), 'N2LR')
        fieldz_LR = avg(fieldz, 'N2LR')
        fieldy_UD = avg(avg(fieldy, 'LR2N'), 'N2UD')
        fieldz_UD = avg(fieldz, 'N2UD')
        
        curl_x =   dirder(g31_N * fieldx_N / nx1_N + g32_N * fieldy_N / nx2_N + J33_N * fieldz / nx3_N, 'N2LR')/J_LR*nx1_LR
        curl_y = - dirder(g31_N * fieldx_N / nx1_N + g32_N * fieldy_N / nx2_N + g33_N * fieldz / nx3_N, 'N2UD')/J_UD*nx2_UD
        curl_z =   dirder(g21_LR * fieldx_LR / nx1_LR + g22_LR * fieldy / nx2_LR + g23_LR * fieldz_LR / nx3_LR, 'LR2C')/J_C*nx3_C\
                 - dirder(g11_UD * fieldx / nx1_UD + g12_UD * fieldy_UD / nx2_UD + g13_UD * fieldz_UD / nx3_UD, 'UD2C')/J_C*nx3_C
    
    return curl_x, curl_y, curl_z

def curl(fieldx, fieldy, fieldz, fieldtype):
    ''' To take the curl of either E or B in Skew coord.
    curl^i = 1/J·(d_j·g_kq·A^q - d_k·g_jq·A^q)
    fieltype=='E': input -> LR,UD,c, output -> UD,LR,n
    fieltype=='B': input -> UD,LR,n, output -> LR,UD,c
    '''
    if fieldtype == 'E':
        fieldx_C = avg(fieldx, 'LR2C')
        fieldy_C = avg(fieldy, 'UD2C')
        fieldx_UD = avg(avg(fieldx, 'LR2C'), 'C2UD')
        fieldz_UD = avg(fieldz, 'C2UD')
        fieldy_LR = avg(avg(fieldy, 'UD2C'), 'C2LR')
        fieldz_LR = avg(fieldz, 'C2LR')

        curl_x =   dirder(g31_C * fieldx_C + g32_C * fieldy_C + g33_C * fieldz, 'C2UD')/J_UD
        curl_y = - dirder(g31_C * fieldx_C + g32_C * fieldy_C + g33_C * fieldz, 'C2LR')/J_LR
        curl_z =   dirder(g21_UD * fieldx_UD + g22_UD * fieldy + g23_UD * fieldz_UD, 'UD2N')/J_N\
                 - dirder(g11_LR * fieldx + g12_LR * fieldy_LR + g13_LR * fieldz_LR, 'LR2N')/J_N
    elif fieldtype == 'B':
        fieldx_N = avg(fieldx, 'UD2N')
        fieldy_N = avg(fieldy, 'LR2N')
        fieldx_LR = avg(avg(fieldx, 'UD2N'), 'N2LR')
        fieldz_LR = avg(fieldz, 'N2LR')
        fieldy_UD = avg(avg(fieldy, 'LR2N'), 'N2UD')
        fieldz_UD = avg(fieldz, 'N2UD')
        
        curl_x =   dirder(g31_N * fieldx_N + g32_N * fieldy_N + J33_N * fieldz, 'N2LR')/J_LR
        curl_y = - dirder(g31_N * fieldx_N + g32_N * fieldy_N + g33_N * fieldz, 'N2UD')/J_UD
        curl_z =   dirder(g21_LR * fieldx_LR + g22_LR * fieldy + g23_LR * fieldz_LR, 'LR2C')/J_C\
                 - dirder(g11_UD * fieldx + g12_UD * fieldy_UD + g13_UD * fieldz_UD, 'UD2C')/J_C
    
    return curl_x, curl_y, curl_z

def div_normalised(fieldx, fieldy, fieldz, fieldtype):
    ''' To take the divergence of either E or B in Skew coord. and normalise the covector to vector (coV = V/||x_xi||)
    div = 1/J·d_i(J·A^i)
    fieltype=='E': input -> LR,UD,c, output -> c,c,c
    fieltype=='B': input -> UD,LR,n, output -> n,n,n
    '''
    nx1_UD = norm(J11_UD, J21_UD, J31_UD)
    nx1_LR = norm(J11_LR, J21_LR, J31_LR)
    nx2_UD = norm(J12_UD, J22_UD, J32_UD)
    nx2_LR = norm(J12_LR, J22_LR, J32_LR)
    if fieldtype == 'E':
        div = (dirder(J_LR * fieldx / nx1_LR, 'LR2C') + dirder(J_UD * fieldy / nx2_UD, 'UD2C'))/J_C

    elif fieldtype == 'B':
        div = (dirder(J_UD * fieldx / nx1_UD, 'UD2N') + dirder(J_LR * fieldy / nx2_LR, 'LR2N'))/J_N

    return div

def div(fieldx, fieldy, fieldz, fieldtype):
    ''' To take the divergence of either E or B in in Skew coord.
    div = 1/J·d_i(J·A^i)
    fieltype=='E': input -> LR,UD,c, output -> c,c,c
    fieltype=='B': input -> UD,LR,n, output -> n,n,n
    '''
    if fieldtype == 'E':
        div = (dirder(J_LR * fieldx, 'LR2C') + dirder(J_UD * fieldy, 'UD2C'))/J_C

    elif fieldtype == 'B':
        div = (dirder(J_UD * fieldx, 'UD2N') + dirder(J_LR * fieldy, 'LR2N'))/J_N

    return div

def phys_to_krylov(E1k, E2k, E3k, uk, vk, wk):
    ''' To populate the Krylov vector using physiscs vectors
    E1,E2,E3 are 2D arrays
    u,v,w of dimensions npart
    '''
    global nxc,nyc,nxn,nyn,npart

    ykrylov = zeros(nxn*nyc+nxc*nyn+nxc*nyc+3*npart,np.float64)
    ykrylov[0:nxn*nyc] = E1k.reshape(nxn*nyc)
    ykrylov[nxn*nyc:nxn*nyc+nxc*nyn] = E2k.reshape(nxc*nyn)
    ykrylov[nxn*nyc+nxc*nyn:nxn*nyc+nxc*nyn+nxc*nyc] = E3k.reshape(nxc*nyc)
    ykrylov[nxn*nyc+nxc*nyn+nxc*nyc:nxn*nyc+nxc*nyn+nxc*nyc+npart] = uk
    ykrylov[nxn*nyc+nxc*nyn+nxc*nyc+npart:nxn*nyc+nxc*nyn+nxc*nyc+2*npart] = vk
    ykrylov[nxn*nyc+nxc*nyn+nxc*nyc+2*npart:nxn*nyc+nxc*nyn+nxc*nyc+3*npart] = wk
    return ykrylov

def krylov_to_phys(xkrylov):
    ''' To populate the physiscs vectors using the Krylov space vector
    E1,E2,E3 are 2D arrays of dimension (nx,ny)
    unew,vnew,wnew of dimensions npart1+npart2
    '''
    global nx,ny,npart

    E1k = np.reshape(xkrylov[0:nxn*nyc],(nxn,nyc))
    E2k = np.reshape(xkrylov[nxn*nyc:nxn*nyc+nxc*nyn],(nxc,nyn))
    E3k = np.reshape(xkrylov[nxn*nyc+nxc*nyn:nxn*nyc+nxc*nyn+nxc*nyc],(nxc,nyc))
    uk = xkrylov[nxn*nyc+nxc*nyn+nxc*nyc:nxn*nyc+nxc*nyn+nxc*nyc+npart]
    vk = xkrylov[nxn*nyc+nxc*nyn+nxc*nyc+npart:nxn*nyc+nxc*nyn+nxc*nyc+2*npart]
    wk = xkrylov[nxn*nyc+nxc*nyn+nxc*nyc+2*npart:nxn*nyc+nxc*nyn+nxc*nyc+3*npart]
    return E1k, E2k, E3k, uk, vk, wk

def residual(xkrylov):
    ''' Calculation of the residual of the equations
    This is the most important part: the definition of the problem
    '''
    #global E1, E2, E3, B1, B2, B3, x, y, u, v, w, QM, q, npart, dt
    E1new, E2new, E3new, unew, vnew, wnew = krylov_to_phys(xkrylov)

    # t:    -1/2        0           1/2         1
    #       B           Bbar        Bnew
    #                   E                       Enew
    #       u           ubar        unew
    #                   x           xbar        xnew
    #                   curlE       curlB
    #                               J/xgen1
    #                   Ep/Bp/xgen2

    xnew = x + unew*dt
    ynew = y + vnew*dt

    xnew = xnew % Lx
    ynew = ynew % Ly

    xbar = (xnew + x)/2.
    ybar = (ynew + y)/2.

    if perturb:
        xgen1, ygen1 = cartesian_to_general_particle(xbar, ybar)
    else:
        xgen1, ygen1 = xbar, ybar

    Jx, Jy, Jz = particle_to_grid_J(xgen1, ygen1, unew, vnew, wnew, q)

    J1, J2, J3 = cartesian_to_general(Jx, Jy, Jz, 'J')

    curlE1, curlE2, curlE3 = curl(E1, E2, E3, 'E')

    B1new = B1 - dt * curlE1
    B2new = B2 - dt * curlE2
    B3new = B3 - dt * curlE3

    curlB1, curlB2, curlB3 = curl(B1new, B2new, B3new,'B')

    resE1 = E1new - E1 - dt * curlB1 + dt * J1
    resE2 = E2new - E2 - dt * curlB2 + dt * J2
    resE3 = E3new - E3 - dt * curlB3 + dt * J3

    B1bar = (B1new + B1) / 2.
    B2bar = (B2new + B2) / 2.
    B3bar = (B3new + B3) / 2.

    ubar = (unew + u) / 2.
    vbar = (vnew + v) / 2.
    wbar = (wnew + w) / 2.

    Ex, Ey, Ez = general_to_cartesian(E1, E2, E3, 'E')
    Bx, By, Bz = general_to_cartesian(B1bar, B2bar, B3bar, 'B')

    if perturb:
        xgen2, ygen2 = cartesian_to_general_particle(x, y)
    else:
        xgen2, ygen2 = x, y

    Exp = grid_to_particle(xgen2, ygen2, Ex, 'LR')
    Eyp = grid_to_particle(xgen2, ygen2, Ey, 'UD')
    Ezp = grid_to_particle(xgen2, ygen2, Ez, 'C')
    Bxp = grid_to_particle(xgen2, ygen2, Bx, 'UD')
    Byp = grid_to_particle(xgen2, ygen2, By, 'LR')
    Bzp = grid_to_particle(xgen2, ygen2, Bz, 'N')

    resu = unew - u - QM * (Exp + vbar * Bzp - wbar * Byp) * dt
    resv = vnew - v - QM * (Eyp - ubar * Bzp + wbar * Bxp) * dt
    resw = wnew - w - QM * (Ezp + ubar * Byp - vbar * Bxp) * dt

    ykrylov = phys_to_krylov(resE1, resE2, resE3, resu, resv, resw)
    return ykrylov

'''
global E1, E2, E3, B1, B2, B3, x, y, u, v, w, QM, q, npart, dt
E1new, E2new, E3new, unew, vnew, wnew = krylov_to_phys(xkrylov)
# method:       YEE          Fabio's YEE
# u:            t = n+1/2 -> t = n 
# unew:         t = n+3/2 -> t = n+1
# ubar:         t = n+1   -> t = n+1/2    (=J1)
ubar = (unew + u)/2.
vbar = (vnew + v)/2.
wbar = (wnew + w)/2.
if relativistic:
    gold = np.sqrt(1.+u**2+v**2+w**2)
    gnew = np.sqrt(1.+unew**2+vnew**2+wnew**2)
    gbar = (gold+gnew)/2.
else:
    gbar = np.ones(npart)

# x:            t = n   -> t = n
# xnew:         t = n+1 -> t = n+1
# xbar :        t = n+1/2 -> t = n+1/2
xbar = x + ubar/gbar*dt/2.
ybar = y + vbar/gbar*dt/2.

# periodic BC: modulo operator "%" which finds the reminder (ex. 10.1%10=0.1)
xbar = xbar%Lx
ybar = ybar%Ly
# conversion to general geom.
if perturb:
    xgenbar, ygenbar = cartesian_to_general_particle(xbar, ybar)
else:
    xgenbar, ygenbar = xbar, ybar

Jx, Jy, Jz = particle_to_grid_J(xgenbar,ygenbar,ubar/gbar,vbar/gbar,wbar/gbar,q)
# J1:           t = n+1/2 -> t = n+1/2  (=curlB1)
J1, J2, J3 = cartesian_to_general(Jx, Jy, Jz, 'J')

# E1:           t = n+1/2 -> t = n
# E1new:        t = n+3/2 -> t = n+1
# E1bar:        t = n+1   -> t = n+1/2
E1bar = (E1new + E1)/2. 
E2bar = (E2new + E2)/2.
E3bar = (E3new + E3)/2.

# curlE1:       t = n -> t = n+1/2  (=E1bar)
curlE1, curlE2, curlE3 = curl(E1bar,E2bar,E3bar,'E')

# B1:           t = -1/2 -> t = n
# B1bar:        t =  1/2 -> t = n+1/2
B1bar = B1 - dt/2.*curlE1
B2bar = B2 - dt/2.*curlE2
B3bar = B3 - dt/2.*curlE3

#curlB1:        t = n+1/2 -> t = n+1/2  (=B1bar)
curlB1, curlB2, curlB3 = curl(B1bar,B2bar,B3bar,'B')

#res:           t = n+1/2 -> t = n+1/2  (=curlB1,J1)
resE1 = E1new - E1 - dt*curlB1 + dt*J1
resE2 = E2new - E2 - dt*curlB2 + dt*J2
resE3 = E3new - E3 - dt*curlB3 + dt*J3

# Ex:           t = n -> t = n+1/2  (=E1bar)
Ex, Ey, Ez = general_to_cartesian(E1bar, E2bar, E3bar, 'E')
# Bx:           t = n+1/2 -> t = n+1/2  (=B1bar)
Bx, By, Bz = general_to_cartesian(B1bar, B2bar, B3bar, 'B')

# Exp           t = n -> t = n+1/2  (=E1bar)
Exp = grid_to_particle(xgenbar,ygenbar,Ex,'LR')
Eyp = grid_to_particle(xgenbar,ygenbar,Ey,'UD')
Ezp = grid_to_particle(xgenbar,ygenbar,Ez,'C') 
# Bxp           t = n+1/2 -> t = n+1/2  (=B1bar)
Bxp = grid_to_particle(xgenbar,ygenbar,Bx,'UD')
Byp = grid_to_particle(xgenbar,ygenbar,By,'LR')
Bzp = grid_to_particle(xgenbar,ygenbar,Bz,'N') 

# resu:         t = n -> t = n+1/2 (=Exp,Bxp,ubar)
resu = unew - u - QM * (Exp + vbar/gbar*Bzp - wbar/gbar*Byp)*dt
resv = vnew - v - QM * (Eyp - ubar/gbar*Bzp + wbar/gbar*Bxp)*dt
resw = wnew - w - QM * (Ezp + ubar/gbar*Byp - vbar/gbar*Bxp)*dt

ykrylov = phys_to_krylov(resE1,resE2,resE3,resu,resv,resw)
return  ykrylov
'''

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

def particle_to_grid_rho(xk, yk, q):
    ''' Interpolation particle to grid - charge rho -> c
    '''
    global dx, dy, nx, ny, npart, rho_ion
    
    rho = zeros(np.shape(xc), np.float64)

    for i in range(npart):
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

        rho[i1, j1] += wx1 * wy1 * q[i]
        rho[i2, j1] += wx2 * wy1 * q[i]
        rho[i1, j2] += wx1 * wy2 * q[i]
        rho[i2, j2] += wx2 * wy2 * q[i]
      
    if electron_and_ion:
        rho += rho_ion

    return rho 

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

# main cycle

define_geometry()

#print(g11_C)
#print(g12_C)
#print(g13_C)
#print(g22_C)
#print(g23_C)
#print(g33_C)

#if perturb:
#    xgen, ygen = cartesian_to_general_particle(x, y)
#else:
#    xgen, ygen = x, y
#
#Jx, Jy, Jz = particle_to_grid_J(xgen, ygen, u, v, w, q)
#J1, J2, J3 = cartesian_to_general(Jx, Jy, Jz, 'J')
#
#if relativistic:
#    histEnergyP1 = [np.sum((g[0:npart1]-1.)*abs(q[0:npart1]/QM[0:npart1]))]
#    histEnergyP2 = [np.sum((g[npart1:npart]-1.)*abs(q[npart1:npart]/QM[npart1:npart]))]
##elif perturb:
##    histEnergyP1 = [0.5*np.sum((J_C * g11_C * avg(E1 * J1, 'LR2C') + 2. * J_C * g12_C * avg(E1, 'LR2C') * avg(J2, 'UD2C')
##                    + J_C * g22_C * avg(E2 * J2, 'UD2C') + J_C * g33_C * E3 * J3)/2.*dx*dy)]
##    histEnergyP2 = [0.5*np.sum((J_C * g11_C * avg(E1 * J1, 'LR2C') + 2. * J_C * g12_C * avg(E1, 'LR2C') * avg(J2, 'UD2C')
##                    + J_C * g22_# * avg(E2 * J2, 'UD2C') + J_C * g33_C * E3 * J3)/2.*dx*dy)]
##elif (not relativistic) and (not perturb):
##    histEnergyP1 = [np.sum((u[0:npart1]**2+v[0:npart1]**2+w[0:npart1]**2)/2.*abs(q[0:npart1]/QM[0:npart1]))]
##    histEnergyP2 = [np.sum((u[npart1:npart]**2+v[npart1:npart]**2 +w[npart1:npart]**2)/2.*abs(q[npart1:npart]/QM[npart1:npart]))]
#else:
#    histEnergyP1 = [np.sum((u[0:npart1]**2+v[0:npart1]**2+w[0:npart1]**2)/2.*abs(q[0:npart1]/QM[0:npart1]))]
#    histEnergyP2 = [np.sum((u[npart1:npart]**2+v[npart1:npart]**2 +w[npart1:npart]**2)/2.*abs(q[npart1:npart]/QM[npart1:npart]))]
#
##if perturb:
##    # Energy -> defined in C
##    histEnergyE = [(np.sum(J_C * g11_C * avg(E1**2, 'LR2C')) +  2.*np.sum(J_C * g12_C * avg(E1, 'LR2C') * avg(E2, 'UD2C'))\
##                 +  np.sum(J_C * g22_C * avg(E2**2, 'UD2C')) +  np.sum(J_C * g33_C * E3**2))/2.*dx*dy]
##    histEnergyB = [(np.sum(J_C * g11_C * avg(B1**2, 'UD2C')) + 2.*np.sum(J_C * g12_C * avg(B1, 'UD2C') * avg(B2, 'LR2C'))
##                 +  np.sum(J_C * g22_C * avg(B2**2, 'LR2C')) +  np.sum(J_C * g33_C * avg(avg(B3**2, 'N2LR'), 'LR2C')))/2.*dx*dy]
##
##    histEnergyTot=[histEnergyE[0]+histEnergyB[0]+histEnergyP1[0]+histEnergyP2[0]]
##
##    histMomentumx = [np.sum(u[0:npart])]
##    histMomentumy = [np.sum(v[0:npart])]
##    histMomentumz = [np.sum(w[0:npart])]
##    histMomentumTot = [histMomentumx[0] + histMomentumy[0] + histMomentumz[0]]
##
##    print('cycle 0, energy=',histEnergyTot[0])
##    print('energyP1=',histEnergyP1[0],'energyP2=',histEnergyP2[0])
##    print('energyE=',histEnergyE[0])
##    print('energyB=',histEnergyB[0])
##    print('Momentumx=',histMomentumx[0],'Momentumy=',histMomentumy[0],'Momentumz=',histMomentumz[0])
#if perturb:
#    # Energy -> defined in C
#    histEnergyE1=[np.sum(J_C * g11_C * avg(E1**2, 'LR2C') \
#                       + J_C * g12_C * avg(E1, 'LR2C') * avg(E2, 'UD2C'))/2.*dx*dy] #\
#                       #+ J_C * g13_C * avg(E1, 'LR2C') * E3)/2.*dx*dy]
#    histEnergyE2=[np.sum(J_C * g21_C * avg(E2, 'UD2C') * avg(E1, 'LR2C') \
#                       + J_C * g22_C * avg(E2**2, 'UD2C'))/2.*dx*dy]# \
#                       #+ J_C * g23_C * avg(E2, 'UD2C') * E3)/2.*dx*dy]
#    histEnergyE3=[np.sum(#J_C * g31_C * E3 * avg(E1, 'LR2C') \
#                       #+ J_C * g32_C * E3 * avg(E2, 'UD2C') \
#                       + J_C * g33_C * E3**2)/2.*dx*dy]
#    histEnergyB1=[np.sum(J_C * g11_C * avg(B1**2, 'UD2C')
#                       + J_C * g12_C * avg(B1, 'UD2C') * avg(B2, 'LR2C'))/2.*dx*dy]# \
#                       #+ J_C * g13_C * avg(B1, 'UD2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy]
#    histEnergyB2=[np.sum(J_C * g21_C * avg(B2, 'LR2C') * avg(B1, 'UD2C')\
#                       + J_C * g22_C * avg(B2**2, 'LR2C'))/2.*dx*dy]# \
#                       #+ J_C * g23_C * avg(B2, 'LR2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy]
#    histEnergyB3=[np.sum(#J_C * g31_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(B1, 'UD2C') \
#                       #+ J_C * g32_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(B2, 'LR2C') \
#                       + J_C * g33_C * avg(avg(B3**2, 'N2LR'), 'LR2C'))/2.*dx*dy]
#else:
#    histEnergyE1=[np.sum(E1[0:nxn-1,:]**2)/2.*dx*dy]
#    histEnergyE2=[np.sum(E2[:,0:nyn-1]**2)/2.*dx*dy]
#    histEnergyE3=[np.sum(E3[:,:]**2)/2.*dx*dy]
#    histEnergyB1=[np.sum(B1[:,0:nyn-1]**2)/2.*dx*dy]
#    histEnergyB2=[np.sum(B2[0:nxn-1,:]**2)/2.*dx*dy]
#    histEnergyB3=[np.sum(B3[0:nxn-1,0:nyn-1]**2)/2.*dx*dy]
#
#histEnergyTot=[histEnergyP1[0]+histEnergyP2[0]+histEnergyE1[0]+histEnergyE2[0]+histEnergyE3[0]+histEnergyB1[0]+histEnergyB2[0]+histEnergyB3[0]]
#
#histMomentumx = [np.sum(u[0:npart])]
#histMomentumy = [np.sum(v[0:npart])]
#histMomentumz = [np.sum(w[0:npart])]
#histMomentumTot = [histMomentumx[0] + histMomentumy[0] + histMomentumz[0]]
#
#energyP[0] = histEnergyP1[0] + histEnergyP2[0]
#energyE[0] = histEnergyE1[0] + histEnergyE2[0] + histEnergyE3[0]
#energyB[0] = histEnergyB1[0] + histEnergyB2[0] + histEnergyB3[0]
#
#
#
#if perturb:
#    xgen, ygen = cartesian_to_general_particle(x, y)
#else:
#    xgen, ygen = x, y
#
#rho_ion = - particle_to_grid_rho(xgen, ygen, q)
#rho = particle_to_grid_rho(xgen, ygen, q)
#temp = 0
#
#
#if plot_dir == True:
#    myplot_map(xn, yn, B3, title='B3', xlabel='x', ylabel='y')
#    filename1 = PATH1 + 'B3_' + '%04d'%temp + '.png'
#    plt.savefig(filename1, dpi=ndpi)
#
#    if nppc!=0:
#        myplot_particle_map(x, y)
#        filename1 = PATH1 + 'part_' + '%04d'%temp + '.png'
#        plt.savefig(filename1, dpi=ndpi)
#
#        myplot_phase_space(x, v, limx=(0, Lx), limy=(-2*V0x1, 2*V0x1), xlabel='x', ylabel='vx')
#        filename1 = PATH1 + 'phase_' + '%04d'%temp + '.png'
#        plt.savefig(filename1, dpi=ndpi)
#
#        myplot_map(xc, yc, rho, title='rho', xlabel='x', ylabel='y')
#        filename1 = PATH1 + 'rho_' + '%04d'%temp + '.png'
#        plt.savefig(filename1, dpi=ndpi)
#
#        myplot_map(xc, yc, div(E1, E2, E3, 'E') - rho, title='div(E)-rho map', xlabel='x', ylabel='y')
#        filename1 = PATH1 + 'div_rho_' + '%04d'%temp + '.png'
#        plt.savefig(filename1, dpi=ndpi)
#
#cpu_time = zeros(nt+1, np.float64)

#print('cycle 0, energy=',histEnergyTot[0])
#print('energyP1=',histEnergyP1[0],'energyP2=',histEnergyP2[0])
#print('energyEx=',histEnergyE1[0],'energyEy=',histEnergyE2[0],'energyEz=',histEnergyE3[0])
#print('energyBx=',histEnergyB1[0],'energyBy=',histEnergyB2[0],'energyBz=',histEnergyB3[0])
#print('Momentumx=',histMomentumx[0],'Momentumy=',histMomentumy[0],'Momentumz=',histMomentumz[0])

start = time.time()

for it in range(0,nt):
    plt.clf()
    #start = time.time()

    if NK_method:
        # The following is python's NK methods
        #guess = zeros(2*nxn*nyc+2*nxc*nyn+nxc*nxc+nxn*nxn+3*2*part,np.float64)
        guess = phys_to_krylov(E1, E2, E3, u, v, w)
        sol = newton_krylov(residual, guess, method='lgmres', verbose=1, f_tol=1e-14)#, f_rtol=1e-7)
        print('Residual: %g' % abs(residual(sol)).max())
    elif Picard:
        # The following is a Picard iteration
        guess = phys_to_krylov(E1, E2, E3, u, v, w) 
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

    #stop = time.time()
    #cpu_time[it] = stop - start

    #if relativistic:
    #    gnew = np.sqrt(1.+unew**2+vnew**2+wnew**2)
    #    gold = np.sqrt(1.+u**2+v**2+w**2)
    #    gbar = (gold+gnew)/2.
    #else:
    #    gbar = np.ones(npart)
    
    # position is evolved in physical space 
    # pushed by general geom. fields converted
    #ubar = (unew + u)/2.
    #vbar = (vnew + v)/2.

    # t:    -1/2        0           1/2         1
    #       B           Bbar        Bnew
    #                   E/Eold                  Enew
    #       u                       unew
    #                   x                       xnew
    #                   curlE
    #                   Energy

    E1new, E2new, E3new, unew, vnew, wnew = krylov_to_phys(sol)

    xnew = x + unew * dt
    ynew = y + vnew * dt

    xnew = xnew % Lx
    ynew = ynew % Ly

    curlE1, curlE2, curlE3 = curl(E1, E2, E3, 'E')

    B1new = B1 - dt * curlE1
    B2new = B2 - dt * curlE2
    B3new = B3 - dt * curlE3

    B1bar = (B1new + B1)/2.
    B2bar = (B2new + B2)/2.
    B3bar = (B3new + B3)/2.

    E1old = E1
    E2old = E2
    E3old = E3
    E3old = E3

    B1old = B1
    B2old = B2
    B3old = B3

    x = xnew
    y = ynew
    u = unew
    v = vnew
    w = wnew

    E1 = E1new
    E2 = E2new
    E3 = E3new
    B1 = B1new
    B2 = B2new
    B3 = B3new


    """
    x += u*dt
    y += v*dt
    x = x%Lx
    y = y%Ly
    u = unew
    v = vnew
    w = wnew

    E1bar = (E1new + E1)/2.
    E2bar = (E2new + E2)/2.
    E3bar = (E3new + E3)/2.

    curlE1, curlE2, curlE3 = curl(E1bar, E2bar, E3bar,'E')

    B1 = B1 - dt*curlE1
    B2 = B2 - dt*curlE2
    B3 = B3 - dt*curlE3
    
    if perturb:
        xgen, ygen = cartesian_to_general_particle(x, y)
    else:
        xgen, ygen = x, y
    """

    if perturb:
        xgen, ygen = cartesian_to_general_particle(x, y)
    else:
        xgen, ygen = x, y

    rho = particle_to_grid_rho(xgen, ygen, q)
    divE[it] = np.sum(div(E1, E2, E3, 'E'))
    divB[it] = np.sum(div(B1, B2, B3, 'B'))
    divE_rho[it] = np.sum(np.abs(div(E1new, E2new, E3new, 'E')) - np.abs(rho))
    
    E1time[it] = np.sum(E1)
    E2time[it] = np.sum(E2)
    E3time[it] = np.sum(E3)
    B1time[it] = np.sum(B1)
    B2time[it] = np.sum(B2)
    B3time[it] = np.sum(B3)

    if relativistic:
        energyP1 = np.sum((g[0:npart1]-1.)*abs(q[0:npart1]/QM[0:npart1]))
        energyP2 = np.sum((g[npart1:npart]-1.)*abs(q[npart1:npart]/QM[npart1:npart]))
    #elif perturb:
    #    energyP1 = 0.5*np.sum((J_C * g11_C * avg(E1 * J1, 'LR2C') + 2. * J_C * avg(g12_LR, 'LR2C') * avg(E1, 'LR2C') * avg(J2, 'UD2C')
    #                         + J_C * g22_C * avg(E2 * J2, 'UD2C') + J_C * g33_C * E3 * J3)/2.*dx*dy)
    #    energyP2 = 0.5*np.sum((J_C * g11_C * avg(E1 * J1, 'LR2C') + 2. * J_C * avg(g12_LR, 'LR2C') * avg(E1, 'LR2C') * avg(J2, 'UD2C')
    #                         + J_C * g22_C * avg(E2 * J2, 'UD2C') + J_C * g33_C * E3 * J3)/2.*dx*dy)
    #elif (not relativistic) and (not perturb):
    else:
        energyP1[it] = np.sum((u[0:npart1]**2+v[0:npart1]**2+w[0:npart1]**2)/2.*abs(q[0:npart1]/QM[0:npart1]))
        energyP2[it] = np.sum((u[npart1:npart]**2+v[npart1:npart]**2 +w[npart1:npart]**2)/2.*abs(q[npart1:npart]/QM[npart1:npart]))
 
#    if perturb:
#        # Energy -> defined in C
#        energyE = (np.sum(J_C * g11_C * avg(E1**2, 'LR2C')) +  2.*np.sum(J_C * g12_C * avg(E1, 'LR2C') * avg(E2, 'UD2C'))\
#                +  np.sum(J_C * g22_C * avg(E2**2, 'UD2C')) +  np.sum(J_C * g33_C * E3**2))/2.*dx*dy
#        energyB = (np.sum(J_C * g11_C * avg(B1**2, 'UD2C')) + 2.*np.sum(J_C * g12_C * avg(B1, 'UD2C') * avg(B2, 'LR2C'))\
#                +  np.sum(J_C * g22_C * avg(B2**2, 'LR2C')) +  np.sum(J_C * g33_C * avg(avg(B3**2, 'N2LR'), 'LR2C')))/2.*dx*dy
#        
#        energyTot = energyE + energyB + energyP1 + energyP2 
#
#        momentumx = np.sum(unew[0:npart])
#        momentumy = np.sum(vnew[0:npart])
#        momentumz = np.sum(wnew[0:npart])
#        momentumTot = momentumx + momentumy + momentumz
#
#        histEnergyP1.append(energyP1)
#        histEnergyP2.append(energyP2)
#        histEnergyE.append(energyE)
#        histEnergyB.append(energyB)
#        histEnergyTot.append(energyTot)
#    
#        histMomentumx.append(momentumx)
#        histMomentumy.append(momentumy)
#        histMomentumz.append(momentumz)
#        histMomentumTot.append(momentumTot)
#
#        energyP[it] = histEnergyP1[it] + histEnergyP2[it]
#        
#        print('cycle',it,'energy =',histEnergyTot[it])
#        print('energyP1=',histEnergyP1[it],'energyP2=',histEnergyP2[it])
#        print('energyE=',histEnergyE[it])
#        print('energyB=',histEnergyB[it])
#        print('relative energy change=',(histEnergyTot[it]-histEnergyTot[0])/histEnergyTot[0])
#        print('momento totale= ', histMomentumTot[it])
    
    if perturb:
        # Energy -> defined in C

        #energyE1[it]= np.sum(J_C * g11_C * avg(E1**2, 'LR2C') \
        #                + J_C * g12_C * avg(E1, 'LR2C') * avg(E2, 'UD2C'))/2.*dx*dy# \
        #                #+ J_C * g13_C * avg(E1, 'LR2C') * E3)/2.*dx*dy 
        #energyE2[it]= np.sum(J_C * g21_C * avg(E2, 'UD2C') * avg(E1, 'LR2C') \
        #                 + J_C * g22_C * avg(E2**2, 'UD2C'))/2.*dx*dy# \
        #                #+ J_C * g23_C * avg(E2, 'UD2C') * E3)/2.*dx*dy 
        #energyE3[it]= np.sum(#J_C * g31_C * E3 * avg(E1, 'LR2C') \
        #               #+ J_C * g32_C * E3 * avg(E2, 'UD2C') \
        #               + J_C * g33_C * E3**2)/2.*dx*dy
        #energyB1[it]= np.sum(J_C * g11_C * avg(B1*B1old, 'UD2C') \
        #               + J_C * g12_C * avg(B1old, 'UD2C') * avg(B2, 'LR2C'))/2.*dx*dy# \
        #               #+ J_C * g13_C * avg(B1, 'UD2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy 
        #energyB2[it]= np.sum(J_C * g21_C * avg(B2old, 'LR2C') * avg(B1, 'UD2C')\
        #                 + J_C * g22_C * avg(B2 * B2old, 'LR2C'))/2.*dx*dy  # \
        #               #+ J_C * g23_C * avg(B2, 'LR2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy 
        #energyB3[it]= np.sum(#J_C * g31_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(B1, 'UD2C') \
        #               #+ J_C * g32_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(B2, 'LR2C') \
        #               + J_C * g33_C * avg(avg(B3*B3old, 'N2LR'), 'LR2C'))/2.*dx*dy

        energyE1[it]= np.sum(J_C * g11_C * avg(E1**2, 'LR2C') \
                           + J_C * g12_C * avg(E1, 'LR2C') * avg(E2, 'UD2C') \
                           + J_C * g13_C * avg(E1, 'LR2C') * E3)/2.*dx*dy 
        energyE2[it]= np.sum(J_C * g21_C * avg(E2, 'UD2C') * avg(E1, 'LR2C') \
                           + J_C * g22_C * avg(E2**2, 'UD2C') \
                           + J_C * g23_C * avg(E2, 'UD2C') * E3)/2.*dx*dy 
        energyE3[it]= np.sum(J_C * g31_C * E3 * avg(E1, 'LR2C') \
                           + J_C * g32_C * E3 * avg(E2, 'UD2C') \
                           + J_C * g33_C * E3**2)/2.*dx*dy
        energyB1[it]= np.sum(J_C * g11_C * avg(B1bar**2, 'UD2C') \
                           + J_C * g12_C * avg(B1bar, 'UD2C') * avg(B2bar, 'LR2C') \
                           + J_C * g13_C * avg(B1, 'UD2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy 
        energyB2[it]= np.sum(J_C * g21_C * avg(B2bar, 'LR2C') * avg(B1bar, 'UD2C')\
                           + J_C * g22_C * avg(B2bar**2, 'LR2C') \
                           + J_C * g23_C * avg(B2, 'LR2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy 
        energyB3[it]= np.sum(J_C * g31_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(B1, 'UD2C') \
                           + J_C * g32_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(B2, 'LR2C') \
                           + J_C * g33_C * avg(avg(B3bar**2, 'N2LR'), 'LR2C'))/2.*dx*dy
    else:
        energyE1 = np.sum(E1[0:nxn-1,:]**2)/2.*dx*dy
        energyE2 = np.sum(E2[:,0:nyn-1]**2)/2.*dx*dy
        energyE3 = np.sum(E3[:,:]**2)/2.*dx*dy
        energyB1 = np.sum(B1[:,0:nyn-1]**2)/2.*dx*dy
        energyB2 = np.sum(B2[0:nxn-1,:]**2)/2.*dx*dy
        energyB3 = np.sum(B3[0:nxn-1,0:nyn-1]**2)/2.*dx*dy
    
    energyTot[it] = energyP1[it] + energyP2[it] + energyE1[it] + energyE2[it] + energyE3[it] + energyB1[it] + energyB2[it] + energyB3[it]
    
    if it==0:
        err_en[it] = 0.
    else:
        err_en[it] = (energyTot[it] - energyTot[it-1])/energyTot[it-1]

    momentumx = np.sum(unew[0:npart])
    momentumy = np.sum(vnew[0:npart])   
    momentumz = np.sum(wnew[0:npart])
    momentumTot[it] = momentumx + momentumy + momentumz

   #histEnergyP1.append(energyP1)
   #histEnergyP2.append(energyP2)
   #histEnergyE1.append(energyE1)
   #histEnergyE2.append(energyE2)
   #histEnergyE3.append(energyE3)
   #histEnergyB1.append(energyB1)
   #histEnergyB2.append(energyB2)
   #histEnergyB3.append(energyB3)
   #histEnergyTot.append(energyTot)
   #
   #histMomentumx.append(momentumx)
   #histMomentumy.append(momentumy)
   #histMomentumz.append(momentumz)
   #histMomentumTot.append(momentumTot)

    energyP[it] = energyP1[it] + energyP2[it]
    energyE[it] = energyE1[it] + energyE2[it] + energyE3[it]
    energyB[it] = energyB1[it] + energyB2[it] + energyB3[it]
        
    print('cycle',it,'energy =',energyTot[it])
    print('energyP1=',energyP1[it],'energyP2=',energyP2[it])
    print('energyE1=',energyE1[it],'energyE2=',energyE2[it],'energyE3=',energyE3[it])
    print('energyB1=',energyB1[it],'energyB2=',energyB2[it],'energyB3=',energyB3[it])
    print('relative energy change=',(energyTot[it]-energyTot[0])/energyTot[0])
    print('momento totale= ', momentumTot[it])
    print('')

    if plot_each_step == True:
        plt.figure(figsize=(12, 9))

        plt.subplot(2, 3, 1)
        plt.pcolor(xn, yn, B3)
        plt.title('B3 map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

        #plt.subplot(2, 3, 2)
        #plt.pcolor(xUD, yUD, E2)
        #plt.title('E_y map')
        #plt.xlabel('x')
        #plt.ylabel('y')
        #plt.colorbar()

        #plt.subplot(2, 3, 3)
        #plt.pcolor(xc, yc, E3)
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
        plt.plot(x[0:npart1], u[0:npart1], 'b.')
        plt.plot(x[npart1:npart], u[npart1:npart], 'r.')
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
        plt.plot(x[0:npart1], y[0:npart1],'b.')
        plt.plot(x[npart1:npart], y[npart1:npart],'r.')
        plt.xlim((0,Lx))
        plt.ylim((0,Ly))
        plt.title('Particles map')
        plt.xlabel('x')
        plt.ylabel('y')

        #plt.subplot(2, 3, 5)
        #plt.plot((histEnergyTot-histEnergyTot[0])/histEnergyTot[0])#, label='U_tot')
        #plt.plot(energyE+energyB)#, label='U_fields')
        #plt.plot(energyE)#, label='U_elect')
        #plt.plot(energyB)#, label = 'U_mag')
        #plt.plot(energyP, label='U_part')
        #plt.title('Energy evolution')
        #plt.xlabel('t')
        #plt.ylabel('E')
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

    if (it % every == 0) or (it == 1):
        '''
        if nppc!=0:
            myplot_particle_map(x, y)
            filename1 = PATH1 + 'part_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi) 

            myplot_phase_space(x, u, limx=(0, Lx), limy=(-2*V0x1, 2*V0x1), xlabel='x', ylabel='vx')
            filename1 = PATH1 + 'phase_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_map(xc, yc, rho, title='rho', xlabel='x', ylabel='y')
            filename1 = PATH1 + 'rho_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)
        '''
        myplot_map(xn, yn, B3, title='B_3', xlabel='x', ylabel='y')
        filename1 = PATH1 + 'B3_' + '%04d' % it + '.png'
        plt.savefig(filename1, dpi=ndpi)

        myplot_map(xLR, yLR, E1, title='E_1', xlabel='x', ylabel='y')
        filename1 = PATH1 + 'E1_' + '%04d' % it + '.png'
        plt.savefig(filename1, dpi=ndpi)
    if plot_data == True:
        f = open("iPiC_2D3V_cov_yee.dat", "a")
        print(it, np.sum(E1), np.sum(E2), np.sum(E3), np.sum(B1), np.sum(B2), np.sum(B3), \
              energyE1, energyE2, energyE3, energyB1, energyB2, energyB3, energyTot, energyP1, energyP2, \
              divE[it], divB[it], file=f)
        f.close()


stop = time.time()
if plot_dir == True:
    myplot_func(energyTot, title='Total Energy', xlabel='t', ylabel='U_Tot')
    filename1 = PATH1 + '@energy_tot_' + '%04d' % it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func((energyTot-energyTot[0])/energyTot[0], title='Relative error on total energy', xlabel='t', ylabel='err(E)')
    filename1 = PATH1 + '@error_rel_E[0]_' + '%04d' % it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(err_en, title='E[t]–E[t-1]/E[t-1]', xlabel='t', ylabel='err(E)')
    filename1 = PATH1 + '@error_rel_E[t-1]_' + '%04d' % it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(energyB,  title='Energy B', xlabel='t', ylabel='U_mag')
    filename1 = PATH1 + '@energy_mag_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(energyE,  title='Energy E', xlabel='t', ylabel='U_el')
    filename1 = PATH1 + '@energy_elec_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(energyP,  title='Energy Part.', xlabel='t', ylabel='U_part')
    filename1 = PATH1 + '@energy_part_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(momentumTot, title='Momentum', xlabel='t', ylabel='p')
    filename1 = PATH1 + '@momentum_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(divE_rho, title='div(E)-rho', xlabel='t', ylabel='div')
    filename1 = PATH1 + '@div(E)-rho_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(divE, title='div(E)', xlabel='t', ylabel='div(E)')
    filename1 = PATH1 + '@div(E)_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(divB, title='div(B)', xlabel='t', ylabel='div')
    filename1 = PATH1 + '@div(B)_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(B1time,  title='B1 time evolution', xlabel='t', ylabel='B1')
    filename1 = PATH1 + '@B1_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(B2time,  title='B2 time evolution', xlabel='t', ylabel='B2')
    filename1 = PATH1 + '@B2_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(B3time,  title='B3 time evolution', xlabel='t', ylabel='B3')
    filename1 = PATH1 + '@B3_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(E1time,  title='E1 time evolution', xlabel='t', ylabel='E1')
    filename1 = PATH1 + '@E1_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(E2time,  title='E2 time evolution', xlabel='t', ylabel='E2')
    filename1 = PATH1 + '@E2_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

    myplot_func(E3time,  title='E3 time evolution', xlabel='t', ylabel='E3')
    filename1 = PATH1 + '@E3_' + '%04d'%it + '.png'
    plt.savefig(filename1, dpi=ndpi)

print('Total cpu time:', stop-start)
