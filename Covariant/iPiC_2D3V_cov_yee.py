#
# Fully Implicit, Relativistic, Covariant Particle-in-Cell - 2D3V Electromagnetic - 2 species
# Authors: G. Lapenta, F. Bacchini, J. Croonen and L. Pezzini
# Date: 23 Jan 2021
# Copyright 2021 KULeuven (CmPA)
# MIT License
#

'''
TODO 
(Priority):
 - Energy definition, Delta enegry -> Surface integral
 - Particle -> add J·E to the flux (Poynting theorem)
 - Flux map -> Ucolored pixel in the bottom left Boundary (-1 <-> nxc Pb?)
 - Add a colorbar in the scatterpot for the computational space
 - Extended to 1d
 (- BE CAREFUL -> roduct of avg or avg of the product)

(Future steps):
 - Particles -> Covariant form (Cristoffel symbol)
 - Field -> Electromagnetic tensor (EM tensor F)
 - Explicit version 
 - Collocated Scheme
 - Cylindrical geometry
 - High order FDTD
 - Runge Kutta
 - Emission?
'''

from scipy.optimize import newton_krylov, minimize
import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

PATH1 = '/Users/luca_pezzini/Documents/Code/cov_pic-2d/figures/'

# method flags
pic              = True     # True -> particles & fields; False -> only fields
Picard           = True     # Picard iteration
NK_method        = False    # Newton Krylov non linear solver

# metric flag
metric           = False    # True -> activate non-identity metric tensor (False = Cartesian)
skew             = False    # perturbed metric: skew in the eta axes
squared          = False    # squared cartesian x^2, y^2, 1
CandC            = False    # harmonic perturbed metric: Chacoon and Chen 2016

# fields only 
pert_delta       = False    # delta function in the centre of the dom. (B field pert.)
pert_sin         = False    # double sinusoidal function (B field pert.)

# plasma flags
ion_bkg          = False    # False -> QM1=-1 QM2=+1 (two species); True -> QM1=QM2=-1 (electrons \wion bkg)
stable_plasma    = True     # stable plasma set up
harmonic         = False    # harmonic oscillator -> initial particle velocity VTH along x
stream_inst      = False    # counterstream inst. set up
landau_damping   = False    # landau damping set up
relativistic     = False    # relativisitc  set up

# plot flags   
file_log         = True     # to save the log file in PATH1
data_save        = True     # to save data a .dat in PATH1
plot_save        = True     # to save the plots in PATH1
plot_each        = False    # to visualise each time step (memory consuming)

# parameters to change
nt = 100                    # number of time steps
nx, ny = 16, 16             # number of grid points
ndpi = 300                  # number of dpi per img (stay low 100 for monitoring purpose!)
every = 100                 # how often to plot
eps = 0.2                   # amplitude of the pertutbation in CandC
theta = np.pi/4.            # angle of skewness
n = 1.                      # mode of oscillation in double sinusoidal B pert (field only)
B0 = 0.01                   # B field perturbation amplitude (field only)
V0 = 1.                     # stream velocity magnitude
alpha = 0.1                 # attenuation of VT respect V0
pcs = 16                    # number particles per cell per species

# !be careful to modify from now on!

nxc, nyc = nx, ny
nxn, nyn = nxc+1, nyc+1
Lx, Ly = 10., 10.
dx, dy = Lx/nxc, Ly/nyc
if squared:
    dt = 0.01 
else:
    dt = 0.05

# Constaint: nppc must be a squarable number (4, 16, 64) because particles are 
#            spread over a squared grid
if pic:
    nppc = pcs              # number particles per cell per species
else:
    nppc = 0                # fields only

# Species 1:
npart1 = nx * ny * nppc
WP1 = 1.                    # plasma frequency
QM1 = -1.                   # charge/mass ratio
V0x1 = V0                   # stream velocity
V0y1 = V0                   # stream velocity
V0z1 = V0                   # stream velocity
VT1 = alpha*V0              # thermal velocity

# Species 2:
npart2 = npart1
WP2 = 1.                    # plasma frequency
if ion_bkg:
    QM2 = -1.               # negative charge/mass ratio 
else:
    QM2 = +1.               # positive charge/mass ratio
V0x2 = V0                   # stream velocity
V0y2 = V0                   # stream velocity
V0z2 = V0                   # stream velocity
VT2 = alpha*V0              # thermal velocity

npart = npart1 + npart2
QM = np.zeros(npart, np.float64)
QM[0:npart1] = QM1
QM[npart1:npart] = QM2

# INIT PARTICLES
np.random.seed(1)

if nppc==0:
    # only fields
    dxp = 0.
    dyp = 0.
else:
    dxp = Lx/np.sqrt(npart1)
    dyp = Ly/np.sqrt(npart1)

xp, yp = np.mgrid[dxp/2.:Lx-dxp/2.:(np.sqrt(npart1)*1j), dyp/2.:Ly-dyp/2.:(np.sqrt(npart1)*1j)]

x = np.zeros(npart, np.float64)
x[0:npart1] = xp.reshape(npart1)
x[0:npart1] = Lx*np.random.rand(npart1)
x[npart1:npart] = x[0:npart1]

y = np.zeros(npart, np.float64)
y[0:npart1] = yp.reshape(npart1)
y[0:npart1] = Ly*np.random.rand(npart1)
y[npart1:npart] = y[0:npart1]

u = np.zeros(npart, np.float64)
if harmonic:
    u[0:npart1] = VT1
    u[npart1:npart] = VT2
elif stable_plasma: 
    u[0:npart1] = VT1*np.random.randn(npart1)
    u[npart1:npart] = VT2*np.random.randn(npart2)
elif stream_inst:
    #u[0:npart1] = V0x1+VT1*np.random.randn(npart1)
    #u[npart1:npart] = V0x2+VT2*np.random.randn(npart2)
    u[0:npart1] = V0x1+VT1*np.random.randn(npart1)
    u[npart1:npart] = V0x2+VT2*np.random.randn(npart2)
    u[1:npart:2] = - u[1:npart:2] # velocity in the odd position are negative
    #np.random.shuffle(u) # to guarantee 50% of +u0 to e- and the other 50% to e+ and same fo -u0

v = np.zeros(npart, np.float64)
if stable_plasma:
    v[0:npart1] = VT1*np.random.randn(npart1)
    v[npart1:npart] = VT2*np.random.randn(npart2)
elif landau_damping:
    v[0:npart1] = V0y1+VT1*np.sin(x[0:npart1]/Lx)
    v[npart1:npart] = V0y2+VT2*np.sin(x[npart1:npart]/Lx)

w = np.zeros(npart, np.float64)
if stable_plasma: 
    w[0:npart1] = VT1*np.random.randn(npart1)
    w[npart1:npart] = VT2*np.random.randn(npart2)

mod_vel = np.zeros(npart, np.float64) # module of velocity

q = np.zeros(npart, np.float64)
q[0:npart1] = np.ones(npart1)*WP1**2/(QM1*npart1/Lx/Ly) 
q[npart1:npart] = np.ones(npart2)*WP2**2/(QM2*npart2/Lx/Ly)

if relativistic:
    g = 1./np.sqrt(1.-(u**2+v**2+w**2))
    u = u*g
    v = v*g
    w = w*g

# INIT LOGIC GRID

if squared:
    xiLR, etaLR = np.mgrid[0.:Lx:(nxn * 1j), dy/2.:Ly - dy/2.:(nyc * 1j)] + dx
    xiUD, etaUD = np.mgrid[dx/2.:Lx - dx/2.:(nxc * 1j), 0.:Ly:(nyn * 1j)] + dx
    xiC, etaC = np.mgrid[dx/2.:Lx - dx/2.:(nxc * 1j), dy/2.:Ly - dy/2.:(nyc * 1j)] + dx
    xiN, etaN = np.mgrid[0.:Lx:(nxn * 1j), 0.:Ly:(nyn * 1j)] + dx
else:
    xiLR, etaLR = np.mgrid[0.:Lx:(nxn * 1j), dy/2.:Ly - dy/2.:(nyc * 1j)] # grid of left-right faces LR
    xiUD, etaUD = np.mgrid[dx/2.:Lx - dx/2.:(nxc * 1j), 0.:Ly:(nyn * 1j)] # grid of up-down faces UD
    xiC, etaC = np.mgrid[dx/2.:Lx - dx/2.:(nxc * 1j), dy/2.:Ly - dy / 2.:(nyc * 1j)] # grid of centres C
    xiN, etaN = np.mgrid[0.:Lx:(nxn * 1j), 0.:Ly:(nyn * 1j)] # grid of corners N

# INIT PHYS GRID
# Unperturbed, the physical grid and logical grid are identical
xLR = xiLR.copy()
xUD = xiUD.copy()
xC = xiC.copy()
xN = xiN.copy()
yLR = etaLR.copy()
yUD = etaUD.copy()
yC = etaC.copy()
yN = etaN.copy()

# INIT FIELDS
# defined on grid LR:        Ex, Jx, By
# defined on grid UD:        Ey, Jy, Bx
# defined on grid centres c: Ez, Jz, rho
# defined on grid corners n: Bz

E1 = np.zeros(np.shape(xiLR), np.float64)
E2 = np.zeros(np.shape(xiUD), np.float64)
E3 = np.zeros(np.shape(xiC), np.float64)
B1 = np.zeros(np.shape(xiUD), np.float64)
B2 = np.zeros(np.shape(xiLR), np.float64)
B3 = np.zeros(np.shape(xiN), np.float64)
# Time series:
E1time = np.zeros(nt+1, np.float64)
E2time = np.zeros(nt+1, np.float64)
E3time = np.zeros(nt+1, np.float64)
B1time = np.zeros(nt+1, np.float64)
B2time = np.zeros(nt+1, np.float64)
B3time = np.zeros(nt+1, np.float64)

if nppc == 0:
    if pert_delta:
        B3[int((nx)/2), int((ny)/2)] = B0
    elif pert_sin:
        B3 = B0 * np.sin(2. * np.pi * n * xiN/Lx) * np.sin(2. * np.pi * n * etaN/Ly)

# CHARGE DENSITY
rho = np.zeros(np.shape(xiC), np.float64)
rho_ion = np.zeros(np.shape(xiC), np.float64)

# INIT JACOBIAN MATRIX
# defined on grid LR:        (J11, J21, J31)E1, J1, B1
# defined on grid UD:        (J12, J22, J32)E2, J2, B2
# defined on grid centres c: (J13, J23, J33)E3, J3
# defined on grid nodes n:   (J13, J23, J33) B3

J11_LR, J22_LR, J33_LR = (np.ones_like(xiLR) for i in range(3))
J12_LR, J13_LR, J21_LR, J23_LR, J31_LR, J32_LR = (np.zeros_like(xiLR, np.float64) for i in range(6))

J11_UD, J22_UD, J33_UD = (np.ones_like(xiUD) for i in range(3))
J12_UD, J13_UD, J21_UD, J23_UD, J31_UD, J32_UD = (np.zeros_like(xiUD, np.float64) for i in range(6))

J11_C, J22_C, J33_C = (np.ones_like(xiC) for i in range(3))
J12_C, J13_C, J21_C, J23_C, J31_C, J32_C = (np.zeros_like(xiC, np.float64) for i in range(6))

J11_N, J22_N, J33_N = (np.ones_like(xiN) for i in range(3))
J12_N, J13_N, J21_N, J23_N, J31_N, J32_N = (np.zeros_like(xiN, np.float64) for i in range(6))

# INIT JACOBIAN DETERMINANT

J_UD = np.ones(np.shape(xiUD), np.float64)
J_LR = np.ones(np.shape(xiLR), np.float64)
J_C = np.ones(np.shape(xiC), np.float64)
J_N = np.ones(np.shape(xiN), np.float64)

# INIT INVERSE JACOBIAN MATRIX
# defined on grid LR:        (j11, j21, j31)E1, J1, B1
# defined on grid UD:        (j12, j22, j32)E2, J2, B2
# defined on grid centres c: (j13, j23, j33)E3, J3
# defined on grid nodes n:   (j13, j23, j33) B3

j11_LR, j22_LR, j33_LR = (np.ones_like(xiLR) for i in range(3))
j12_LR, j13_LR, j21_LR, j23_LR, j31_LR, j32_LR = (np.zeros_like(xiLR, np.float64) for i in range(6))

j11_UD, j22_UD, j33_UD = (np.ones_like(xiUD) for i in range(3))
j12_UD, j13_UD, j21_UD, j23_UD, j31_UD, j32_UD = (np.zeros_like(xiUD, np.float64) for i in range(6))

j11_C, j22_C, j33_C = (np.ones_like(xiC) for i in range(3))
j12_C, j13_C, j21_C, j23_C, j31_C, j32_C = (np.zeros_like(xiC, np.float64) for i in range(6))

j11_N, j22_N, j33_N = (np.ones_like(xiN) for i in range(3))
j12_N, j13_N, j21_N, j23_N, j31_N, j32_N = (np.zeros_like(xiN, np.float64) for i in range(6))

# INIT INVERSE JACOBIAN DETERMINANT

j_LR = np.ones(np.shape(xiLR), np.float64)
j_UD = np.ones(np.shape(xiUD), np.float64)
j_C = np.ones(np.shape(xiC), np.float64)
j_N = np.ones(np.shape(xiN), np.float64)

# INIT METRIC TENSOR
# defined on grid LR:        (g11, g21, g31)E1, J1, B1
# defined on grid UD:        (g12, g22, g32)E2, J2, B2
# defined on grid centres c: (g13, g23, g33)E3, J3
# defined on grid nodes n:   (g13, g23, g33) B3

g11_LR, g22_LR, g33_LR = (np.ones_like(xiLR) for i in range(3))
g12_LR, g13_LR, g21_LR, g23_LR, g31_LR, g32_LR = (np.zeros_like(xiLR, np.float64) for i in range(6))

g11_UD, g22_UD, g33_UD = (np.ones_like(xiUD) for i in range(3))
g12_UD, g13_UD, g21_UD, g23_UD, g31_UD, g32_UD = (np.zeros_like(xiUD, np.float64) for i in range(6))

g11_C, g22_C, g33_C = (np.ones_like(xiC) for i in range(3))
g12_C, g13_C, g21_C, g23_C, g31_C, g32_C = (np.zeros_like(xiC, np.float64) for i in range(6))

g11_N, g22_N, g33_N = (np.ones_like(xiN) for i in range(3))
g12_N, g13_N, g21_N, g23_N, g31_N, g32_N = (np.zeros_like(xiN, np.float64) for i in range(6))

# DIVERGENCE
# defined on grid c:
divE = np.zeros(nt+1, np.float64)
divE_rho = np.zeros(nt+1, np.float64)
# defined on grid n:
divB = np.zeros(nt+1, np.float64)

# Energy:
energyP     = np.zeros(nt+1, np.float64)  # particles
energyE     = np.zeros(nt+1, np.float64)  # E field
energyB     = np.zeros(nt+1, np.float64)  # B field
err_en      = np.zeros(nt+1, np.float64)  # energy tot error
histdelta_U = np.zeros(nt+1, np.float64)  # partial_t U time series

# TESTING curl:
delta_curlB1 = np.zeros(nt, np.float64)
delta_curlB2 = np.zeros(nt, np.float64)
delta_curlB3 = np.zeros(nt, np.float64)

delta_curlE1 = np.zeros(nt, np.float64)
delta_curlE2 = np.zeros(nt, np.float64)
delta_curlE3 = np.zeros(nt, np.float64)

if file_log == True:
    f = open(PATH1 + 'log_file.txt', 'w')
    print('iPiC_2D3V_cov_yee.py', file=f)
    print('- Particle in Cell: ', pic, file=f)
    print('* METRIC:', file=f)
    print('- non identity metric tensor: ', metric, file=f)
    print('- Chacoon & Chen metric: ', CandC, file=f)
    print('- perturbation amplitude eps_g: ', eps, file=f)
    print('- skew metric: ', skew, file=f)
    print('- angle of the skewness : ', theta, file=f)
    print('- squared metric: ', squared, file=f)
    print('* METHOD:', file=f)
    print('- NK method: ', NK_method, file=f)
    print('- Picard iteration: ', Picard, file=f)
    print('* PHYSICS:', file=f)
    print('- perturbation amplitude B0 (if nppc=0): ', B0, file=f)
    print('- field perturbation delta: ', pert_delta, file=f)
    print('- field perturbation sin: ', pert_sin, file=f)
    print('- mode of oscillation n (if nppc=0): ', n, file=f)
    print('- stable plasma: ', stable_plasma, file=f)
    print('- harmonic oscillator: ', harmonic, file=f)
    print('- electrons & ions: ', ion_bkg, file=f)
    print('- counter stream inst.: ', stream_inst, file=f)
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
    print('* SPECIES 2: ', file=f)
    print('- number of particles : ', npart2, file=f)
    print('- plasma frequency : ', WP2, file=f)
    print('- charge to mass : ', QM2, file=f)
    print('- velocity field: ', '(', V0x2, ',', V0y2, ',', V0z2, ')', file=f)
    print('- thermal velocity: ', VT2, file=f)
    f.close()

def myplot_map(xgrid, ygrid, field, xlabel='x', ylabel='y', title='title'):
    '''To plot the map of a vector fied over a grid
    '''
    plt.figure()
    plt.pcolor(xgrid, ygrid, field)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.title(title)

def myplot_pert_map(xgrid, ygrid, field, xlabel='x', ylabel='y', title='title'):
    '''To plot the map of a vector fied over a perturbed grid (Physical space)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.scatter(xgrid.flatten(), ygrid.flatten(), c=field.flatten(), s=50)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def myplot_time_series(field, ylabel= 'a', title='title'):
    '''To plot the behavior of a scalar fied in time
    '''
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    plt.figure()
    plt.plot(field)
    plt.xlabel(r'$t\omega_{pe}$')
    plt.ylabel(ylabel)
    plt.title(title)

def myplot_particle_map(posx, posy):
    '''To plot particles position over the domain
    '''
    plt.figure()
    plt.plot(posx[0:npart1],posy[0:npart1],'b.')
    plt.plot(posx[npart1:npart],posy[npart1:npart],'r.')
    plt.xlim((0,Lx))
    plt.ylim((0,Ly))
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.title('Particles map')

def myplot_particle_hystogram(velocity):
    plt.figure()
    plt.hist(velocity, bins=40, align='right', edgecolor='black')
    plt.xlabel(r'$|v|$')
    plt.ylabel('frequency')

def myplot_particle_hystogram_sns(velocity):
    vel = pd.Series(velocity, name="x variable")
    ax = sns.distplot(vel)
    ax.set_xlabel( r"$|v|$") 
    ax.set_ylabel( "Frequency") 
    
def myplot_phase_space(pos, vel, limx=(0, 0), limy=(0, 0), xlabel='b', ylabel='c'):
    '''To plot the phase space in one direction
    '''
    plt.figure()
    plt.plot(pos[0:npart1], vel[0:npart1], 'b.')
    plt.plot(pos[npart1:npart], vel[npart1:npart], 'r.')
    plt.xlim(limx)
    plt.ylim(limy)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title('Particles map')

def perturbed_inverse_jacobian_elements(x, y):
    if CandC:
        j11 = 1. + 2. * np.pi * eps * np.cos(2. * np.pi * x / Lx) * np.sin(2. * np.pi * y / Ly) / Lx
        j12 = 2. * np.pi * eps * np.sin(2. * np.pi * x / Lx) * np.cos(2. * np.pi * y / Ly) / Ly
        j13 = np.zeros(np.shape(x), np.float64)
        j21 = 2. * np.pi * eps * np.cos(2. * np.pi * x / Lx) * np.sin(2. * np.pi * y / Ly) / Lx
        j22 = 1. + 2. * np.pi * eps * np.sin(2. * np.pi * x / Lx) * np.cos(2. * np.pi * y / Ly) / Ly
        j23 = np.zeros(np.shape(x), np.float64)
        j31 = np.zeros(np.shape(x), np.float64)
        j32 = np.zeros(np.shape(x), np.float64)
        j33 = np.ones(np.shape(x), np.float64)
    elif skew:
        j11 = np.ones(np.shape(x), np.float64)
        j12 = np.zeros(np.shape(x), np.float64)
        j13 = np.zeros(np.shape(x), np.float64)
        j21 = np.cos(theta)
        j22 = np.sin(theta)
        j23 = np.zeros(np.shape(x), np.float64)
        j31 = np.zeros(np.shape(x), np.float64)
        j32 = np.zeros(np.shape(x), np.float64)
        j33 = np.ones(np.shape(x), np.float64)
    elif squared:
        j11 = 2. * x
        j12 = np.zeros(np.shape(x), np.float64)
        j13 = np.zeros(np.shape(x), np.float64)
        j21 = np.zeros(np.shape(x), np.float64)
        j22 = 2. * y
        j23 = np.zeros(np.shape(x), np.float64)
        j31 = np.zeros(np.shape(x), np.float64)
        j32 = np.zeros(np.shape(x), np.float64)
        j33 = np.ones(np.shape(x), np.float64)
  
    return j11, j12, j13, j21, j22, j23, j31, j32, j33

def define_geometry():
    '''To construct the structure of the general geometry (for each grid type):
    - Get the Jacobian matrix and its determinant
    - Get the inverse Jacobian matrix isolate the components and calculate its determinant
    - Get the metric tensor components
    '''
    for i in range(np.shape(xiLR)[0]):
        for j in range(np.shape(xiLR)[1]):
            j11_LR[i, j], j12_LR[i, j], j13_LR[i, j], j21_LR[i, j], j22_LR[i, j], j23_LR[i, j], j31_LR[i,j], j32_LR[i, j], j33_LR[i, j] = perturbed_inverse_jacobian_elements(xLR[i, j], yLR[i, j])
            inverse_jacobian_LR = np.array([[j11_LR[i, j], j12_LR[i, j], j13_LR[i, j]], [j21_LR[i, j], j22_LR[i, j], j23_LR[i, j]], [j31_LR[i, j], j32_LR[i, j], j33_LR[i, j]]])
            jacobian_LR = np.linalg.inv(inverse_jacobian_LR)
            j_LR[i, j] = np.linalg.det(inverse_jacobian_LR)
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
            g11_LR[i, j] = jacobian_LR[0, 0] * jacobian_LR[0, 0] + jacobian_LR[1, 0] * jacobian_LR[1, 0] + jacobian_LR[2, 0] * jacobian_LR[2, 0]
            g21_LR[i, j] = jacobian_LR[0, 0] * jacobian_LR[0, 1] + jacobian_LR[1, 0] * jacobian_LR[1, 1] + jacobian_LR[2, 0] * jacobian_LR[2, 1]
            g31_LR[i, j] = jacobian_LR[0, 0] * jacobian_LR[0, 2] + jacobian_LR[1, 0] * jacobian_LR[1, 2] + jacobian_LR[2, 0] * jacobian_LR[2, 2]
            g12_LR[i, j] = jacobian_LR[0, 0] * jacobian_LR[0, 1] + jacobian_LR[1, 0] * jacobian_LR[1, 1] + jacobian_LR[2, 0] * jacobian_LR[2, 1]
            g22_LR[i, j] = jacobian_LR[0, 1] * jacobian_LR[0, 1] + jacobian_LR[1, 1] * jacobian_LR[1, 1] + jacobian_LR[2, 1] * jacobian_LR[2, 1]
            g32_LR[i, j] = jacobian_LR[0, 1] * jacobian_LR[0, 2] + jacobian_LR[1, 1] * jacobian_LR[1, 2] + jacobian_LR[2, 1] * jacobian_LR[2, 2]
            g13_LR[i, j] = jacobian_LR[0, 0] * jacobian_LR[0, 2] + jacobian_LR[1, 0] * jacobian_LR[1, 2] + jacobian_LR[2, 0] * jacobian_LR[2, 2]
            g23_LR[i, j] = jacobian_LR[0, 1] * jacobian_LR[0, 2] + jacobian_LR[1, 1] * jacobian_LR[1, 2] + jacobian_LR[2, 1] * jacobian_LR[2, 2]
            g33_LR[i, j] = jacobian_LR[0, 2] * jacobian_LR[0, 2] + jacobian_LR[1, 2] * jacobian_LR[1, 2] + jacobian_LR[2, 2] * jacobian_LR[2, 2]

    for i in range(np.shape(xiUD)[0]):
        for j in range(np.shape(xiUD)[1]):
            j11_UD[i, j], j12_UD[i, j], j13_UD[i, j], j21_UD[i, j], j22_UD[i, j], j23_UD[i, j], j31_UD[i, j], j32_UD[i, j], j33_UD[i, j] = perturbed_inverse_jacobian_elements(xUD[i, j], yUD[i, j])
            inverse_jacobian_UD = np.array([[j11_UD[i, j], j12_UD[i, j], j13_UD[i, j]], [j21_UD[i, j], j22_UD[i, j], j23_UD[i, j]], [j31_UD[i, j], j32_UD[i, j], j33_UD[i, j]]])
            jacobian_UD = np.linalg.inv(inverse_jacobian_UD)
            j_UD[i, j] = np.linalg.det(inverse_jacobian_UD)
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
            g11_UD[i, j] = jacobian_UD[0, 0] * jacobian_UD[0, 0] + jacobian_UD[1, 0] * jacobian_UD[1, 0] + jacobian_UD[2, 0] * jacobian_UD[2, 0]
            g21_UD[i, j] = jacobian_UD[0, 0] * jacobian_UD[0, 1] + jacobian_UD[1, 0] * jacobian_UD[1, 1] + jacobian_UD[2, 0] * jacobian_UD[2, 1]
            g31_UD[i, j] = jacobian_UD[0, 0] * jacobian_UD[0, 2] + jacobian_UD[1, 0] * jacobian_UD[1, 2] + jacobian_UD[2, 0] * jacobian_UD[2, 2]
            g12_UD[i, j] = jacobian_UD[0, 0] * jacobian_UD[0, 1] + jacobian_UD[1, 0] * jacobian_UD[1, 1] + jacobian_UD[2, 0] * jacobian_UD[2, 1]
            g22_UD[i, j] = jacobian_UD[0, 1] * jacobian_UD[0, 1] + jacobian_UD[1, 1] * jacobian_UD[1, 1] + jacobian_UD[2, 1] * jacobian_UD[2, 1]
            g32_UD[i, j] = jacobian_UD[0, 1] * jacobian_UD[0, 2] + jacobian_UD[1, 1] * jacobian_UD[1, 2] + jacobian_UD[2, 1] * jacobian_UD[2, 2]
            g13_UD[i, j] = jacobian_UD[0, 0] * jacobian_UD[0, 2] + jacobian_UD[1, 0] * jacobian_UD[1, 2] + jacobian_UD[2, 0] * jacobian_UD[2, 2]
            g23_UD[i, j] = jacobian_UD[0, 1] * jacobian_UD[0, 2] + jacobian_UD[1, 1] * jacobian_UD[1, 2] + jacobian_UD[2, 1] * jacobian_UD[2, 2]
            g33_UD[i, j] = jacobian_UD[0, 2] * jacobian_UD[0, 2] + jacobian_UD[1, 2] * jacobian_UD[1, 2] + jacobian_UD[2, 2] * jacobian_UD[2, 2]

    for i in range(np.shape(xiC)[0]):
        for j in range(np.shape(xiC)[1]):
            j11_C[i, j], j12_C[i, j], j13_C[i, j], j21_C[i, j], j22_C[i, j], j23_C[i, j], j31_C[i, j], j32_C[i, j], j33_C[i, j] = perturbed_inverse_jacobian_elements(xC[i, j], yC[i, j])
            inverse_jacobian_C = np.array([[j11_C[i, j], j12_C[i, j], j13_C[i, j]], [ j21_C[i, j], j22_C[i, j], j23_C[i, j]], [j31_C[i, j], j32_C[i, j], j33_C[i, j]]])
            jacobian_C = np.linalg.inv(inverse_jacobian_C)
            j_C[i, j] = np.linalg.det(inverse_jacobian_C)
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
            g11_C[i, j] = jacobian_C[0, 0] * jacobian_C[0, 0] + jacobian_C[1, 0] * jacobian_C[1, 0] + jacobian_C[2, 0] * jacobian_C[2, 0]
            g21_C[i, j] = jacobian_C[0, 0] * jacobian_C[0, 1] + jacobian_C[1, 0] * jacobian_C[1, 1] + jacobian_C[2, 0] * jacobian_C[2, 1]
            g31_C[i, j] = jacobian_C[0, 0] * jacobian_C[0, 2] + jacobian_C[1, 0] * jacobian_C[1, 2] + jacobian_C[2, 0] * jacobian_C[2, 2]
            g12_C[i, j] = jacobian_C[0, 0] * jacobian_C[0, 1] + jacobian_C[1, 0] * jacobian_C[1, 1] + jacobian_C[2, 0] * jacobian_C[2, 1]
            g22_C[i, j] = jacobian_C[0, 1] * jacobian_C[0, 1] + jacobian_C[1, 1] * jacobian_C[1, 1] + jacobian_C[2, 1] * jacobian_C[2, 1]
            g32_C[i, j] = jacobian_C[0, 1] * jacobian_C[0, 2] + jacobian_C[1, 1] * jacobian_C[1, 2] + jacobian_C[2, 1] * jacobian_C[2, 2]
            g13_C[i, j] = jacobian_C[0, 0] * jacobian_C[0, 2] + jacobian_C[1, 0] * jacobian_C[1, 2] + jacobian_C[2, 0] * jacobian_C[2, 2]
            g23_C[i, j] = jacobian_C[0, 1] * jacobian_C[0, 2] + jacobian_C[1, 1] * jacobian_C[1, 2] + jacobian_C[2, 1] * jacobian_C[2, 2]
            g33_C[i, j] = jacobian_C[0, 2] * jacobian_C[0, 2] + jacobian_C[1, 2] * jacobian_C[1, 2] + jacobian_C[2, 2] * jacobian_C[2, 2]

    for i in range(np.shape(xiN)[0]):
        for j in range(np.shape(xiN)[1]):
            j11_N[i, j], j12_N[i, j], j13_N[i, j], j21_N[i, j], j22_N[i, j], j23_N[i, j], j31_N[i, j], j32_N[i, j], j33_N[i, j] = perturbed_inverse_jacobian_elements(xN[i, j], yN[i, j])
            inverse_jacobian_N = np.array([[j11_N[i, j], j12_N[i, j], j13_N[i, j]], [ j21_N[i, j], j22_N[i, j], j23_N[i, j]], [j31_N[i, j], j32_N[i, j], j33_N[i, j]]])
            jacobian_N = np.linalg.inv(inverse_jacobian_N)
            j_N[i, j] = np.linalg.det(inverse_jacobian_N)
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
            g11_N[i, j] = jacobian_N[0, 0] * jacobian_N[0, 0] + jacobian_N[1, 0] * jacobian_N[1, 0] + jacobian_N[2, 0] * jacobian_N[2, 0]
            g21_N[i, j] = jacobian_N[0, 0] * jacobian_N[0, 1] + jacobian_N[1, 0] * jacobian_N[1, 1] + jacobian_N[2, 0] * jacobian_N[2, 1]
            g31_N[i, j] = jacobian_N[0, 0] * jacobian_N[0, 2] + jacobian_N[1, 0] * jacobian_N[1, 2] + jacobian_N[2, 0] * jacobian_N[2, 2]
            g12_N[i, j] = jacobian_N[0, 0] * jacobian_N[0, 1] + jacobian_N[1, 0] * jacobian_N[1, 1] + jacobian_N[2, 0] * jacobian_N[2, 1]
            g22_N[i, j] = jacobian_N[0, 1] * jacobian_N[0, 1] + jacobian_N[1, 1] * jacobian_N[1, 1] + jacobian_N[2, 1] * jacobian_N[2, 1]
            g32_N[i, j] = jacobian_N[0, 1] * jacobian_N[0, 2] + jacobian_N[1, 1] * jacobian_N[1, 2] + jacobian_N[2, 1] * jacobian_N[2, 2]
            g13_N[i, j] = jacobian_N[0, 0] * jacobian_N[0, 2] + jacobian_N[1, 0] * jacobian_N[1, 2] + jacobian_N[2, 0] * jacobian_N[2, 2]
            g23_N[i, j] = jacobian_N[0, 1] * jacobian_N[0, 2] + jacobian_N[1, 1] * jacobian_N[1, 2] + jacobian_N[2, 1] * jacobian_N[2, 2]
            g33_N[i, j] = jacobian_N[0, 2] * jacobian_N[0, 2] + jacobian_N[1, 2] * jacobian_N[1, 2] + jacobian_N[2, 2] * jacobian_N[2, 2]

def norm(vec1, vec2, vec3):
    '''To calculate the norm of a covariant/contravariant/ordinary vector
    '''
    return np.sqrt(vec1**2 + vec2**2 + vec3**2)

def cartesian_to_general_normalised(cartx, carty, cartz, fieldtype):
    ''' To convert fields from Cartesian coord. (x, y, z) to General Skew coord. (xi, eta, zeta) and normalise the covector to vector (coV = V/||x_xi||)
    fieltype=='E' or 'J': input -> LR,UD,c, output -> LR,UD,c
    fieltype=='B':        input -> UD,LR,n, output -> UD,LR,n
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

    if (fieldtype == 'E') or (fieldtype == 'J'):
        carty_LR = avg(avg(carty, 'UD2C'), 'C2LR')
        cartz_LR = avg(cartz, 'C2LR')
        cartx_UD = avg(avg(cartx, 'LR2C'), 'C2UD')
        cartz_UD = avg(cartz, 'C2UD')
        cartx_C  = avg(cartx, 'LR2C')
        carty_C  = avg(carty, 'UD2C')
        genx1 = (J11_LR * cartx / nx1_LR + J12_LR * carty_LR / nx2_LR + J13_LR * cartz_LR / nx3_LR)*nx3_LR
        genx2 = (J21_UD * cartx_UD / nx1_UD + J22_UD * carty / nx2_UD + J23_UD * cartz_UD / nx3_UD)*nx3_UD
        genx3 = (J31_C * cartx_C / nx1_C + J32_C * carty_C / nx2_C + J33_C * cartz / nx3_C)/nx3_C
    elif fieldtype == 'B':
        carty_UD = avg(avg(carty, 'LR2C'), 'C2UD')
        cartz_UD = avg(cartz, 'N2UD')
        cartx_LR = avg(avg(cartx, 'UD2C'), 'C2LR')
        cartz_LR = avg(cartz, 'N2LR')
        cartx_N = avg(cartx, 'UD2N')
        carty_N = avg(carty, 'LR2N')
        genx1 = (J11_UD * cartx / nx1_UD + J12_UD * carty_UD / nx2_UD + J13_UD * cartz_UD / nx3_UD)*nx3_UD
        genx2 = (J21_LR * cartx_LR / nx1_LR + J22_LR * carty / nx2_LR + J23_LR * cartz_LR / nx3_LR)*nx3_LR
        genx3 = (J31_N * cartx_N / nx1_N + J32_N * carty_N / nx2_N + J33_N * cartz / nx3_N)*nx3_N
    
    return genx1, genx2, genx3

def cartesian_to_general(cartx, carty, cartz, fieldtype):
    ''' To convert fields from Cartesian coord. (x, y, z) to General Skew coord. (xi, eta, zeta)
    fieltype=='E' or 'J': input -> LR,UD,c, output -> LR,UD,c
    fieltype=='B':        input -> UD,LR,n, output -> UD,LR,n
    '''
    if (fieldtype == 'E') or (fieldtype == 'J'):
        carty_LR = avg(avg(carty, 'UD2C'), 'C2LR')
        cartz_LR = avg(cartz, 'C2LR')
        cartx_UD = avg(avg(cartx, 'LR2C'), 'C2UD')
        cartz_UD = avg(cartz, 'C2UD')
        cartx_C = avg(cartx, 'LR2C')
        carty_C = avg(carty, 'UD2C')
        genx1 = J11_LR * cartx    + J12_LR * carty_LR + J13_LR * cartz_LR
        genx2 = J21_UD * cartx_UD + J22_UD * carty    + J23_UD * cartz_UD
        genx3 = J31_C  * cartx_C  + J32_C  * carty_C  + J33_C  * cartz
    elif fieldtype == 'B':
        carty_UD = avg(avg(carty, 'LR2C'), 'C2UD')
        cartz_UD = avg(cartz, 'N2UD')
        cartx_LR = avg(avg(cartx, 'UD2C'), 'C2LR')
        cartz_LR = avg(cartz, 'N2LR')
        cartx_N = avg(cartx, 'UD2N')
        carty_N = avg(carty, 'LR2N')
        genx1 = J11_UD * cartx    + J12_UD * carty_UD + J13_UD * cartz_UD
        genx2 = J21_LR * cartx_LR + J22_LR * carty    + J23_LR * cartz_LR
        genx3 = J31_N  * cartx_N  + J32_N  * carty_N  + J33_N  * cartz
    
    return genx1, genx2, genx3

def general_to_cartesian(genx1, genx2, genx3, fieldtype):
    ''' To convert fields from General coord. (xi, eta, zeta) to Cartesian coord (x, y, z)
    fieltype=='E' or 'J': input -> LR,UD,c, output -> LR,UD,c
    fieltype=='B':        input -> UD,LR,n, output -> UD,LR,n
    '''
    if (fieldtype == 'E') or (fieldtype == 'J'):
        genx2_LR = avg(avg(genx2, 'UD2C'), 'C2LR')
        genx3_LR = avg(genx3, 'C2LR')
        genx1_UD = avg(avg(genx1, 'LR2C'), 'C2UD')
        genx3_UD = avg(genx3, 'C2UD')
        genx1_C = avg(genx1, 'LR2C')
        genx2_C = avg(genx2, 'UD2C')
        cartx = j11_LR * genx1    + j12_LR * genx2_LR + j13_LR * genx3_LR
        carty = j21_UD * genx1_UD + j22_UD * genx2    + j23_UD * genx3_UD
        cartz = j31_C  * genx1_C  + j32_C  * genx2_C  + j33_C  * genx3
    elif fieldtype == 'B':
        genx2_UD = avg(avg(genx2, 'LR2C'), 'C2UD')
        genx3_UD = avg(genx3, 'N2UD')
        genx1_LR = avg(avg(genx1, 'UD2C'), 'C2LR')
        genx3_LR = avg(genx3, 'N2LR')
        genx1_N = avg(genx1, 'UD2N')
        genx2_N = avg(genx2, 'LR2N')
        cartx = j11_UD * genx1    + j12_UD * genx2_UD + j13_UD * genx3_UD
        carty = j21_LR * genx1_LR + j22_LR * genx2    + j23_LR * genx3_LR
        cartz = j31_N  * genx1_N  + j32_N  * genx2_N  + j33_N  * genx3
    
    return cartx, carty, cartz

def general_to_cartesian_delta(gendx1, gendx2, gridtype):
    ''' To convert space increment (dx, dy) from General coord. (xi, eta, zeta) to Cartesian coord (x, y, z)
    '''
    if gridtype == 'LR':
        cartx = j11_LR * gendx1 + j12_LR * gendx2
        carty = j21_LR * gendx1 + j22_LR * gendx2 
    elif gridtype == 'UD':
        cartx = j11_UD * gendx1 + j12_UD * gendx2
        carty = j21_UD * gendx1 + j22_UD * gendx2
    elif gridtype == 'C':
        cartx = j11_C * gendx1 + j12_C * gendx2
        carty = j21_C * gendx1 + j22_C * gendx2
    elif gridtype == 'N':
        cartx = j11_N * gendx1 + j12_N * gendx2
        carty = j21_N * gendx1 + j22_N * gendx2

    return cartx, carty

def cartesian_to_general_particle(cartx, carty):
    '''To convert the particles position from Cartesian geom. to General geom.
    '''
    if CandC:
        genx1 = cartx + eps*np.sin(2*np.pi*cartx/Lx)*np.sin(2*np.pi*carty/Ly)
        genx2 = carty + eps*np.sin(2*np.pi*cartx/Lx)*np.sin(2*np.pi*carty/Ly)
    elif skew:
        genx1 = cartx
        genx2 = np.cos(theta) * cartx + np.sin(theta) * carty
    elif squared:
        genx1 = cartx**2
        genx2 = carty**2
    
    return genx1, genx2

def diff_for_inversion(param, target):
    xi, eta = cartesian_to_general_particle(param[0], param[1])
    return (xi - target[0]) ** 2 + (eta - target[1]) ** 2

def cart_grid_calculator(xi, eta):
    if xi.shape != eta.shape:
        raise ValueError
    x = np.zeros_like(xi)
    y = np.zeros_like(eta)

    init0 = xi.copy()
    init1 = eta.copy()

    init = np.stack((init0, init1), axis=-1)
    target = np.stack((xi, eta), axis=-1)

    # use this to set bounds on the values of x and y
    bnds = ((None, None), (None, None))
    for i in range(xi.shape[0]):
        for j in range(xi.shape[1]):
            res = minimize(lambda param: diff_for_inversion(param, target[i, j, :]), init[i, j, :], bounds=bnds, tol=1e-16)
            x[i, j] = res.x[0]
            y[i, j] = res.x[1]
    return x, y

def dirder(field, dertype):
    ''' To take the directional derivative of a quantity
    dertype defines input/output grid type and direction
    '''
    global nxn, nyn, nxc, nyc, dx, dy

    if dertype == 'C2UD':  # centres to UD faces, y-derivative
      derfield = np.zeros((nxc, nyn), np.float64)

      derfield[0:nxc, 1:nyn-1] = (field[0:nxc, 1:nyc]-field[0:nxc, 0:nyc-1])/dy
      derfield[0:nxc, 0] = (field[0:nxc, 0]-field[0:nxc, nyc-1])/dy
      derfield[0:nxc, nyn-1] = derfield[0:nxc, 0]

    elif dertype == 'C2LR':  # centres to LR faces, x-derivative
      derfield = np.zeros((nxn, nyc), np.float64)

      derfield[1:nxn-1, 0:nyc] = (field[1:nxc, 0:nyc]-field[0:nxc-1, 0:nyc])/dx
      derfield[0, 0:nyc] = (field[0, 0:nyc]-field[nxc-1, 0:nyc])/dx
      derfield[nxn-1, 0:nyc] = derfield[0, 0:nyc]

    elif dertype == 'UD2N':  # UD faces to nodes, x-derivative
      derfield = np.zeros((nxn, nyn), np.float64)

      derfield[1:nxn-1, 0:nyn] = (field[1:nxc, 0:nyn]-field[0:nxc-1, 0:nyn])/dx
      derfield[0, 0:nyn] = (field[0, 0:nyn]-field[nxc-1, 0:nyn])/dx
      derfield[nxn-1, 0:nyn] = derfield[0, 0:nyn]

    elif dertype == 'LR2N':  # LR faces to nodes, y-derivative
      derfield = np.zeros((nxn, nyn), np.float64)

      derfield[0:nxn, 1:nyn-1] = (field[0:nxn, 1:nyc]-field[0:nxn, 0:nyc-1])/dy
      derfield[0:nxn, 0] = (field[0:nxn, 0]-field[0:nxn, nyc-1])/dy
      derfield[0:nxn, nyn-1] = derfield[0:nxn, 0]

    elif dertype == 'N2LR':  # nodes to LR faces, y-derivative
      derfield = np.zeros((nxn, nyc), np.float64)

      derfield[0:nxn, 0:nyc] = (field[0:nxn, 1:nyn]-field[0:nxn, 0:nyn-1])/dy

    elif dertype == 'N2UD':  # nodes to UD faces, x-derivative
      derfield = np.zeros((nxc, nyn), np.float64)

      derfield[0:nxc, 0:nyn] = (field[1:nxn, 0:nyn]-field[0:nxn-1, 0:nyn])/dx

    elif dertype == 'LR2C':  # LR faces to centres, x-derivative
      derfield = np.zeros((nxc, nyc), np.float64)

      derfield[0:nxc, 0:nyc] = (field[1:nxn, 0:nyc]-field[0:nxn-1, 0:nyc])/dx

    elif dertype == 'UD2C':  # UD faces to centres, y-derivative
      derfield = np.zeros((nxc, nyc), np.float64)

      derfield[0:nxc, 0:nyc] = (field[0:nxc, 1:nyn]-field[0:nxc, 0:nyn-1])/dy

    return derfield

def avgC2N(fieldC):
    ''' To average a 2D field defined on centres to the nodes
    '''
    global nx,ny

    fieldN = np.zeros((nx,ny),np.float64)

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
    ''' To take the average of a quantity with mine method
    avgtype defines input/output grid type and direction
    '''
    global nxn, nyn, nxc, nyc, dx, dy

    if avgtype == 'C2UD':  # centres to UD faces, y-average
      avgfield = np.zeros((nxc, nyn), np.float64)

      avgfield[0:nxc, 1:nyn-1] = (field[0:nxc, 1:nyc]+field[0:nxc, 0:nyc-1])/2.
      avgfield[0:nxc, 0] = (field[0:nxc, 0]+field[0:nxc, nyc-1])/2.
      avgfield[0:nxc, nyn-1] = avgfield[0:nxc, 0]

    elif avgtype == 'C2LR':  # centres to LR faces, x-average
      avgfield = np.zeros((nxn, nyc), np.float64)

      avgfield[1:nxn-1, 0:nyc] = (field[1:nxc, 0:nyc]+field[0:nxc-1, 0:nyc])/2.
      avgfield[0, 0:nyc] = (field[0, 0:nyc]+field[nxc-1, 0:nyc])/2.
      avgfield[nxn-1, 0:nyc] = avgfield[0, 0:nyc]

    elif avgtype == 'UD2N':  # UD faces to nodes, x-average
      avgfield = np.zeros((nxn, nyn), np.float64)

      avgfield[1:nxn-1, 0:nyn] = (field[1:nxc, 0:nyn]+field[0:nxc-1, 0:nyn])/2.
      avgfield[0, 0:nyn] = (field[0, 0:nyn]+field[nxc-1, 0:nyn])/2.
      avgfield[nxn-1, 0:nyn] = avgfield[0, 0:nyn]

    elif avgtype == 'LR2N':  # LR faces to nodes, y-average
      avgfield = np.zeros((nxn, nyn), np.float64)

      avgfield[0:nxn, 1:nyn-1] = (field[0:nxn, 1:nyc]+field[0:nxn, 0:nyc-1])/2.
      avgfield[0:nxn, 0] = (field[0:nxn, 0]+field[0:nxn, nyc-1])/2.
      avgfield[0:nxn, nyn-1] = avgfield[0:nxn, 0]

    elif avgtype == 'N2LR':  # nodes to LR faces, y-average
      avgfield = np.zeros((nxn, nyc), np.float64)

      avgfield[0:nxn, 0:nyc] = (field[0:nxn, 1:nyn]+field[0:nxn, 0:nyn-1])/2.

    elif avgtype == 'N2UD':  # nodes to UD faces, x-average
      avgfield = np.zeros((nxc, nyn), np.float64)

      avgfield[0:nxc, 0:nyn] = (field[1:nxn, 0:nyn]+field[0:nxn-1, 0:nyn])/2.

    elif avgtype == 'LR2C':  # LR faces to centres, x-average
      avgfield = np.zeros((nxc, nyc), np.float64)

      avgfield[0:nxc, 0:nyc] = (field[1:nxn, 0:nyc]+field[0:nxn-1, 0:nyc])/2.

    elif avgtype == 'UD2C':  # UD faces to centres, y-average
      avgfield = np.zeros((nxc, nyc), np.float64)

      avgfield[0:nxc, 0:nyc] = (field[0:nxc, 1:nyn]+field[0:nxc, 0:nyn-1])/2.

    return avgfield

def shift(mat, x, y):
    '''To shift the matrix in xy-position with periodic boundary conditions
    '''
    result = np.roll(mat, -x, 0)
    result = np.roll(result, -y, 1)

    return result

def avg2(A, x, y):
    '''To generalise the average function. Input: the field A and the movemente in the x and y direction.
    '''
    res = (shift(A, x, 0) + A) / 2
    res = (shift(res, 0, y) + res) / 2
    return res

def avg_joost(field, type):
    ''' To take the average of a quantity with Joost's method
    avgtype defines input/output grid type and direction
    '''
    if type == 'C2UD':
        avgfield = np.zeros_like(xiUD)
        avgfield[:, 0:-1] = avg2(field, 0, -1)
        avgfield[:, -1] = avgfield[:, 0]
    if type == 'C2LR':
        avgfield = np.zeros_like(xiLR)
        avgfield[0:-1, :] = avg2(field, -1, 0)
        avgfield[-1, :] = avgfield[0, :]
    if type == 'C2N':
        avgfield = np.zeros_like(xiN)
        avgfield[0:-1, 0:-1] = avg2(field, -1, -1)
        avgfield[:, -1] = avgfield[:, 0]
        avgfield[-1, :] = avgfield[0, :]
        avgfield[-1, -1] = avgfield[0, 0]
    if type == 'N2UD':
        field = field[0:-1, 0:-1]
        avgfield = np.zeros_like(xiUD)
        avgfield[:, 0:-1] = avg2(field, 0, 1)
        avgfield[:, -1] = avgfield[:, 0]
    if type == 'N2LR':
        field = field[0:-1, 0:-1]
        avgfield = np.zeros_like(xiLR)
        avgfield[0:-1, :] = avg2(field, 1, 0)
        avgfield[-1, :] = avgfield[0, :]
    if type == 'N2C':
        field = field[0:-1, 0:-1]
        avgfield = np.zeros_like(xiC)
        avgfield = avg2(field, 1, 1)
    if type == 'LR2C':
        field = field[0:-1, :]
        avgfield = np.zeros_like(xiC)
        avgfield = avg2(field, 1, 0)
    if type == 'LR2UD':
        field = field[0:-1, :]
        avgfield = np.zeros_like(xiUD)
        avgfield[:, 0:-1] = avg2(field, 1, -1)
        avgfield[:, -1] = avgfield[:, 0]
    if type == 'LR2N':
        field = field[0:-1, :]
        avgfield = np.zeros_like(xiN)
        avgfield[0:-1, 0:-1] = avg2(field, 0, -1)
        avgfield[:, -1] = avgfield[:, 0]
        avgfield[-1, :] = avgfield[0, :]
        avgfield[-1, -1] = avgfield[0, 0]
    if type == 'UD2C':
        field = field[:, 0:-1]
        avgfield = np.zeros_like(xiC)
        avgfield = avg2(field, 0, 1)
    if type == 'UD2N':
        field = field[:, 0:-1]
        avgfield = np.zeros_like(xiN)
        avgfield[0:-1, 0:-1] = avg2(field, -1, 0)
        avgfield[:, -1] = avgfield[:, 0]
        avgfield[-1, :] = avgfield[0, :]
        avgfield[-1, -1] = avgfield[0, 0]
    if type == 'UD2LR':
        field = field[:, 0:-1]
        avgfield = np.zeros_like(xiLR)
        avgfield[0:-1, :] = avg2(field, -1, 1)
        avgfield[-1, :] = avgfield[0, :]
    return avgfield

def curl(fieldx, fieldy, fieldz, fieldtype):
    ''' To take the curl of either E or B in General coord.
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
        
        curl_x =   dirder(g31_N * fieldx_N + g32_N * fieldy_N + g33_N * fieldz, 'N2LR')/J_LR
        curl_y = - dirder(g31_N * fieldx_N + g32_N * fieldy_N + g33_N * fieldz, 'N2UD')/J_UD
        curl_z =   dirder(g21_LR * fieldx_LR + g22_LR * fieldy + g23_LR * fieldz_LR, 'LR2C')/J_C\
                 - dirder(g11_UD * fieldx + g12_UD * fieldy_UD + g13_UD * fieldz_UD, 'UD2C')/J_C
    
    return curl_x, curl_y, curl_z

def curl_normalised(fieldx, fieldy, fieldz, fieldtype):
    ''' To take the curl of either E or B in General coordinate and normalise the covector to vector (coV = V/||x_xi||)
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
        
        curl_x =   dirder(g31_N * fieldx_N / nx1_N + g32_N * fieldy_N / nx2_N + g33_N * fieldz / nx3_N, 'N2LR')/J_LR*nx1_LR
        curl_y = - dirder(g31_N * fieldx_N / nx1_N + g32_N * fieldy_N / nx2_N + g33_N * fieldz / nx3_N, 'N2UD')/J_UD*nx2_UD
        curl_z =   dirder(g21_LR * fieldx_LR / nx1_LR + g22_LR * fieldy / nx2_LR + g23_LR * fieldz_LR / nx3_LR, 'LR2C')/J_C*nx3_C\
                 - dirder(g11_UD * fieldx / nx1_UD + g12_UD * fieldy_UD / nx2_UD + g13_UD * fieldz_UD / nx3_UD, 'UD2C')/J_C*nx3_C
    
    return curl_x, curl_y, curl_z

def curl_skewed(fieldx, fieldy, fieldz, fieldtype):
    ''' DISCLAIMER: this is the test function to proove curl written in general way is written correctly!
    To take the curl of either E or B in skew coord.
    fieltype=='E': input -> LR,UD,c, output -> UD,LR,n
    fieltype=='B': input -> UD,LR,n, output -> LR,UD,c
    '''
    if fieldtype == 'E':      
        fieldx_UD = avg(avg(fieldx, 'LR2N'), 'N2UD')
        fieldy_LR = avg(avg(fieldy, 'UD2N'), 'N2LR')
        
        #curl_x =   dirder(fieldz, 'C2UD') * sin(theta)
        #curl_y = - dirder(fieldz, 'C2LR') * sin(theta)
        #curl_z =   dirder(- (cos(theta)/(sin(theta))**2) * fieldx_UD + (1./(sin(theta))**2) * fieldy, 'UD2N') * sin(theta)\
        #         - dirder(  1/sin(theta))**2 * fieldx - (cos(theta)/(sin(theta))**2) * fieldy_LR, 'LR2N') * sin(theta)
        curl_x =   dirder(fieldz, 'C2UD') * np.sin(theta)
        curl_y = - dirder(fieldz, 'C2LR') * np.sin(theta)
        curl_z = dirder(- np.cos(theta) * fieldx_UD + fieldy, 'UD2N')/np.sin(theta) \
               - dirder(  fieldx - np.cos(theta) * fieldy_LR, 'LR2N')/np.sin(theta) # 
    elif fieldtype == 'B':
        fieldx_LR = avg(avg(fieldx, 'UD2N'), 'N2LR')
        fieldy_UD = avg(avg(fieldy, 'LR2N'), 'N2UD')
       
        curl_x =   dirder(fieldz, 'N2LR') * np.sin(theta)
        curl_y = - dirder(fieldz, 'N2UD') * np.sin(theta)
        curl_z =   dirder(- (np.cos(theta)/(np.sin(theta))**2) * fieldx_LR + (1./(np.sin(theta))**2) * fieldy, 'LR2C') * np.sin(theta)\
                 - dirder(  (np.cos(theta)/np.sin(theta))**2 * fieldx + (np.cos(theta)/(np.sin(theta))**2) * fieldy_UD, 'UD2C') * np.sin(theta)

    return curl_x, curl_y, curl_z

def div_normalised(fieldx, fieldy, fieldz, fieldtype):
    ''' To take the divergence of either E or B in General coord. and normalise the covector to vector (coV = V/||x_xi||)
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
    ''' To take the divergence of either E or B in in General coord.
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

    ykrylov = np.zeros(nxn*nyc+nxc*nyn+nxc*nyc+3*npart,np.float64)
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
    global E1, E2, E3, B1, B2, B3, u, v, w, QM, q, npart, dt

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
    
    # x:            t = n     -> t = n
    # xnew:         t = n+1   -> t = n+1
    # xbar :        t = n+1/2 -> t = n+1/2
    xbar = x + ubar/gbar*dt/2.
    ybar = y + vbar/gbar*dt/2.

    # periodic BC: modulo operator "%" which finds the reminder (ex. 10.1%10=0.1)
    xbar = xbar%Lx
    ybar = ybar%Ly
    # conversion to general geom.
    if metric:
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
    #E1bar1, E2bar1, E3bar1 = cartesian_to_general(E1bar, E2bar, E3bar, 'E')
    #curlE1, curlE2, curlE3 = curl(E1bar1, E2bar1, E3bar1, 'E')
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

def grid_to_particle(xk, yk, f, gridtype):
    ''' Interpolation of grid quantity to particle
    '''
    global dx, dy, nx, ny, npart
    
    fp = np.zeros(npart,np.float64)

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
    
    rho = np.zeros(np.shape(xiC), np.float64)

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
      
    if ion_bkg:
        rho += rho_ion

    return rho 

def particle_to_grid_J(xk, yk, uk, vk, wk, qk): 
    ''' Interpolation particle to grid - current -> LR, UD, c
    ''' 
    global dx, dy, nxc, nyc, nxn, nyn, npart
  
    Jx = np.zeros(np.shape(xiLR),np.float64)
    Jy = np.zeros(np.shape(xiUD),np.float64)
    Jz = np.zeros(np.shape(xiC),np.float64)

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

def poynting_flux_nometric():
    '''To calculate the Poynting flux as int ExB·n dS. 
       The components of S are defined in the center of the edges: Sx -> L, R and Sy -> U, D   
        __Sy__
       |      |
       Sx     Sx
       |__Sy__|
    '''
    # Quantity in the left grid L have pure indices
    E2L = avg(avg(E2, 'UD2C'), 'C2LR')[0:nxn-1, :]
    E3L = avg(E3, 'C2LR')[0:nxn-1, :]
    B2L = B2[0:nxn-1, :]
    B3L = avg(B3, 'N2LR')[0:nxn-1, :]
    # Quantity in the right grid R have indices advanced by one in x
    E2R = avg(avg(E2, 'UD2C'), 'C2LR')[1:nxn, :]
    E3R = avg(E3, 'C2LR')[1:nxn, :]
    B2R = B2[1:nxn, :]
    B3R = avg(B3, 'N2LR')[1:nxn, :]
    # Quantity in the down grid D have pure indices
    E1D = avg(avg(E1, 'LR2C'), 'C2UD')[:, 0:nyn-1]
    E3D = avg(E3, 'C2UD')[:, 0:nyn-1]
    B1D = B1[:, 0:nyn-1]
    B3D = avg(B3, 'N2UD')[:, 0:nyn-1]
    # Quantity in the up grid U have pure indices advanced by one in y
    E1U = avg(avg(E1, 'LR2C'), 'C2UD')[:, 1:nyn]
    E3U = avg(E3, 'C2UD')[:, 1:nyn]
    B1U = B1[:, 1:nyn]
    B3U = avg(B3, 'N2UD')[:, 1:nyn]

    fluxL =    E2L * B3L - E3L * B2L
    fluxR =    E2R * B3R - E3R * B2R
    fluxD = - (E1D * B3D - E3D * B1D)
    fluxU = - (E1U * B3U - E3U * B1U)

    return (fluxU - fluxD)*dx + (fluxR - fluxL)*dy

def poynting_flux():
    '''To calculate the Poynting flux as int ExB·n dS. 
       The components of S are defined in the center of the edges: Sx -> L, R and Sy -> U, D   
        __Sy__
       |      |
       Sx     Sx
       |__Sy__|
    '''
    fluxUD = np.zeros(np.shape(xiC), np.float64)
    fluxLR = np.zeros(np.shape(xiC), np.float64)

    # Quantity in the left grid L have pure indices
    g21L = g21_LR[0:nxn-1, :]
    g22L = g22_LR[0:nxn-1, :]
    g23L = g23_LR[0:nxn-1, :]
    g31L = g31_LR[0:nxn-1, :]
    g32L = g32_LR[0:nxn-1, :]
    g33L = g33_LR[0:nxn-1, :]
    E1L = E1[0:nxn-1, :]
    E2L = avg(avg(E2, 'UD2C'), 'C2LR')[0:nxn-1, :]
    E3L = avg(E3, 'C2LR')[0:nxn-1, :]
    B1L = avg(avg(B1, 'UD2C'), 'C2LR')[0:nxn-1, :]
    B2L = B2[0:nxn-1, :]
    B3L = avg(B3, 'N2LR')[0:nxn-1, :]
    # Quantity in the right grid R have indices advanced by one in xi
    g21R = g21_LR[1:nxn, :]
    g22R = g22_LR[1:nxn, :]
    g23R = g23_LR[1:nxn, :]
    g31R = g31_LR[1:nxn, :]
    g32R = g32_LR[1:nxn, :]
    g33R = g33_LR[1:nxn, :]
    E1R = E1[1:nxn, :]
    E2R = avg(avg(E2, 'UD2C'), 'C2LR')[1:nxn, :]
    E3R = avg(E3, 'C2LR')[1:nxn, :]
    B1R = avg(avg(B1, 'UD2C'), 'C2LR')[1:nxn, :]
    B2R = B2[1:nxn, :]
    B3R = avg(B3, 'N2LR')[1:nxn, :]
    # Quantity in the down grid D have pure indices
    g11D = g11_UD[:, 0:nyn-1]
    g12D = g12_UD[:, 0:nyn-1]
    g13D = g13_UD[:, 0:nyn-1]
    g31D = g31_UD[:, 0:nyn-1]
    g32D = g32_UD[:, 0:nyn-1]
    g33D = g33_UD[:, 0:nyn-1]
    E1D = avg(avg(E1, 'LR2C'), 'C2UD')[:, 0:nyn-1]
    E2D = E2[:, 0:nyn-1]
    E3D = avg(E3, 'C2UD')[:, 0:nyn-1]
    B1D = B1[:, 0:nyn-1]
    B2D = avg(avg(B2, 'LR2C'), 'C2UD')[:, 0:nyn-1]
    B3D = avg(B3, 'N2UD')[:, 0:nyn-1]
    # Quantity in the up grid U have pure indices advanced by one in eta
    g11U = g11_UD[:, 1:nyn]
    g12U = g12_UD[:, 1:nyn]
    g13U = g13_UD[:, 1:nyn]
    g31U = g31_UD[:, 1:nyn]
    g32U = g32_UD[:, 1:nyn]
    g33U = g33_UD[:, 1:nyn]
    E1U = avg(avg(E1, 'LR2C'), 'C2UD')[:, 1:nyn]
    E2U = E2[:, 1:nyn]
    E3U = avg(E3, 'C2UD')[:, 1:nyn]
    B1U = B1[:, 1:nyn]
    B2U = avg(avg(B2, 'LR2C'), 'C2UD')[:, 1:nyn]
    B3U = avg(B3, 'N2UD')[:, 1:nyn]

    fluxL = (g21L*E1L + g22L*E2L + g23L*E3L) * (g31L*B1L + g32L*B2L + g33L*B3L)\
          - (g31L*E1L + g32L*E2L + g33L*E3L) * (g21L*B1L + g22L*B2L + g23L*B3L)
    fluxR = (g21R*E1R + g22R*E2R + g23R*E3R) * (g31R*B1R + g32R*B2R + g33R*B3R)\
          - (g31R*E1R + g32R*E2R + g33R*E3R) * (g21R*B1R + g22R*B2R + g23R*B3R)
    fluxD = (g31D*E1D + g32D*E2D + g33D*E3D) * (g11D*B1R + g12D*B2R + g13D*B3R)\
          - (g11D*E1D + g12D*E2D + g13D*E3D) * (g31D*B1D + g32D*B2D + g33D*B3D)
    fluxU = (g31U*E1U + g32U*E2U + g33U*E3U) * (g11U*B1U + g12U*B2U + g13U*B3U)\
          - (g11U*E1U + g12U*E2U + g13U*E3U) * (g31U*B1U + g32U*B2U + g33U*B3U)
    
    # Why if we substitute -1 with nyc is dimentional problematic? 
    fluxUD[0:nxc, 0] = fluxD[0:nxc, 0] - fluxU[0:nxc, -1]
    fluxUD[0:nxc, 1:nyc] = fluxD[0:nxc, 1:nyc] - fluxU[0:nxc, 0:nyc-1]
    fluxLR[0, 0:nyc] = fluxL[0, 0:nyc] - fluxR[-1, 0:nyc]
    fluxLR[1:nxc, 0:nyc] = fluxL[1:nxc, 0:nyc] - fluxR[0:nxc-1, 0:nyc]

    return fluxUD * dx + fluxLR * dy

def poynting_flux_joost():
    '''To calculate the Poynting flux as int ExB·n dS with Joost avg function version. 
       The components of S are defined in the center of the edges: Sx -> L, R and Sy -> U, D   
        __Sy__
       |      |
       Sx     Sx
       |__Sy__|
    '''
    fluxUD = np.zeros(np.shape(xiC), np.float64)
    fluxLR = np.zeros(np.shape(xiC), np.float64)

    # Quantity in the left grid L have pure indices
    g21L = g21_LR[0:nxn-1, :]
    g22L = g22_LR[0:nxn-1, :]
    g23L = g23_LR[0:nxn-1, :]
    g31L = g31_LR[0:nxn-1, :]
    g32L = g32_LR[0:nxn-1, :]
    g33L = g33_LR[0:nxn-1, :]
    E1L = E1[0:nxn-1, :]
    E2L = avg(E2, 'UD2LR')[0:nxn-1, :]
    E3L = avg(E3, 'C2LR')[0:nxn-1, :]
    B1L = avg(B1, 'UD2LR')[0:nxn-1, :]
    B2L = B2[0:nxn-1, :]
    B3L = avg(B3, 'N2LR')[0:nxn-1, :]
    # Quantity in the right grid R have indices advanced by one in xi
    g21R = g21_LR[1:nxn, :]
    g22R = g22_LR[1:nxn, :]
    g23R = g23_LR[1:nxn, :]
    g31R = g31_LR[1:nxn, :]
    g32R = g32_LR[1:nxn, :]
    g33R = g33_LR[1:nxn, :]
    E1R = E1[1:nxn, :]
    E2R = avg(E2, 'UD2LR')[1:nxn, :]
    E3R = avg(E3, 'C2LR')[1:nxn, :]
    B1R = avg(B1, 'UD2LR')[1:nxn, :]
    B2R = B2[1:nxn, :]
    B3R = avg(B3, 'N2LR')[1:nxn, :]
    # Quantity in the down grid D have pure indices
    g11D = g11_UD[:, 0:nyn-1]
    g12D = g12_UD[:, 0:nyn-1]
    g13D = g13_UD[:, 0:nyn-1]
    g31D = g31_UD[:, 0:nyn-1]
    g32D = g32_UD[:, 0:nyn-1]
    g33D = g33_UD[:, 0:nyn-1]
    E1D = avg(E1, 'LR2UD')[:, 0:nyn-1]
    E2D = E2[:, 0:nyn-1]
    E3D = avg(E3, 'C2UD')[:, 0:nyn-1]
    B1D = B1[:, 0:nyn-1]
    B2D = avg(B2, 'LR2UD')[:, 0:nyn-1]
    B3D = avg(B3, 'N2UD')[:, 0:nyn-1]
    # Quantity in the up grid U have pure indices advanced by one in eta
    g11U = g11_UD[:, 1:nyn]
    g12U = g12_UD[:, 1:nyn]
    g13U = g13_UD[:, 1:nyn]
    g31U = g31_UD[:, 1:nyn]
    g32U = g32_UD[:, 1:nyn]
    g33U = g33_UD[:, 1:nyn]
    E1U = avg(E1, 'LR2UD')[:, 1:nyn]
    E2U = E2[:, 1:nyn]
    E3U = avg(E3, 'C2UD')[:, 1:nyn]
    B1U = B1[:, 1:nyn]
    B2U = avg(B2, 'LR2UD')[:, 1:nyn]
    B3U = avg(B3, 'N2UD')[:, 1:nyn]

    fluxL = (g21L*E1L + g22L*E2L + g23L*E3L) * (g31L*B1L + g32L*B2L + g33L*B3L)\
          - (g31L*E1L + g32L*E2L + g33L*E3L) * (g21L*B1L + g22L*B2L + g23L*B3L)
    fluxR = (g21R*E1R + g22R*E2R + g23R*E3R) * (g31R*B1R + g32R*B2R + g33R*B3R)\
          - (g31R*E1R + g32R*E2R + g33R*E3R) * (g21R*B1R + g22R*B2R + g23R*B3R)
    fluxD = (g31D*E1D + g32D*E2D + g33D*E3D) * (g11D*B1R + g12D*B2R + g13D*B3R)\
          - (g11D*E1D + g12D*E2D + g13D*E3D) * (g31D*B1D + g32D*B2D + g33D*B3D)
    fluxU = (g31U*E1U + g32U*E2U + g33U*E3U) * (g11U*B1U + g12U*B2U + g13U*B3U)\
          - (g11U*E1U + g12U*E2U + g13U*E3U) * (g31U*B1U + g32U*B2U + g33U*B3U)

    fluxUD[0:nxc, 0] = fluxD[0:nxc, 0] - fluxU[0:nxc, -1] 
    fluxUD[0:nxc, 1:nyc] = fluxD[0:nxc, 1:nyc] - fluxU[0:nxc, 0:nyc-1]
    fluxLR[0, 0:nyc] = fluxL[0, 0:nyc] - fluxR[-1, 0:nyc]
    fluxLR[1:nxc, 0:nyc] = fluxL[1:nxc, 0:nyc] - fluxR[0:nxc-1, 0:nyc]

    return fluxUD * dx + fluxLR * dy

# main cycle
# Initialisation of geometry
print('Initialising geometry ...')
start_geom = time.time()

if metric:
    # to get the map x^i=f(\xi^i)
    xLR, yLR = cart_grid_calculator(xiLR, etaLR)
    xUD, yUD = cart_grid_calculator(xiUD, etaUD)
    xC, yC = cart_grid_calculator(xiC, etaC)
    xN, yN = cart_grid_calculator(xiN, etaN)
    define_geometry()
    xgen, ygen = cartesian_to_general_particle(x, y)
else:
    xgen, ygen = x, y

stop_geom = time.time()

#print(g11_C)
#print(g12_C)
#print(g13_C)
#print(g22_C)
#print(g23_C)
#print(g33_C)
    
Jx, Jy, Jz = particle_to_grid_J(xgen, ygen, u, v, w, q)
J1, J2, J3 = cartesian_to_general(Jx, Jy, Jz, 'J')

if relativistic:
    histEnergyP1 = [np.sum((g[0:npart1]-1.)*abs(q[0:npart1]/QM[0:npart1]))]
    histEnergyP2 = [np.sum((g[npart1:npart]-1.)*abs(q[npart1:npart]/QM[npart1:npart]))]
else:
    #histEnergyP1 = [np.sum((u[0:npart1]**2+v[0:npart1]**2+w[0:npart1]**2)/2.*abs(q[0:npart1]/QM[0:npart1]))]
    #histEnergyP2 = [np.sum((u[npart1:npart]**2+v[npart1:npart]**2 +w[npart1:npart]**2)/2.*abs(q[npart1:npart]/QM[npart1:npart]))]
    histEnergyP1 = [np.sum((u[0:npart1]**2+v[0:npart1]**2+w[0:npart1]**2)/2.*abs(q[0:npart1]/QM[0:npart1]))]
    histEnergyP2 = [np.sum((u[npart1:npart]**2+v[npart1:npart]**2 +w[npart1:npart]**2)/2.*abs(q[npart1:npart]/QM[npart1:npart]))]

if metric:    
    # in Physical space -> set in natural position; yes BC
    #Ex, Ey, Ez = general_to_cartesian(E1, E2, E3, 'E')
    #Bx, By, Bz = general_to_cartesian(B1, B2, B3, 'B')
    #deltax_LR, deltay_LR = general_to_cartesian_delta(dx, dy, 'LR')
    #deltax_UD, deltay_UD = general_to_cartesian_delta(dx, dy, 'UD')
    #deltax_C, deltay_C = general_to_cartesian_delta(dx, dy, 'C')
    #deltax_N, deltay_N = general_to_cartesian_delta(dx, dy, 'N')

    #histEnergyE1 = [np.sum(Ex[0:nxn-1,:]**2/2.*deltax_LR[0:nxn-1,:]*deltay_LR[0:nxn-1,:])]
    #histEnergyE2 = [np.sum(Ey[:,0:nyn-1]**2/2.*deltax_UD[:,0:nyn-1]*deltay_UD[:,0:nyn-1])]
    #histEnergyE3 = [np.sum(Ez[:,:]**2/2.*deltax_C[:,:]*deltay_C[:,:])]
    #histEnergyB1 = [np.sum(Bx[:,0:nyn-1]**2/2.*deltax_UD[:,0:nyn-1]*deltay_UD[:,0:nyn-1])]
    #histEnergyB2 = [np.sum(By[0:nxn-1,:]**2/2.*deltax_LR[0:nxn-1,:]*deltay_LR[0:nxn-1,:])]
    #histEnergyB3 = [np.sum(Bz[0:nxn-1,0:nyn-1]**2/2.*deltax_N[0:nxn-1,0:nyn-1]*deltay_N[0:nxn-1,0:nyn-1])]

    # in Logical space -> set in mixed position no BC
    #histEnergyE1 = [np.sum(J_LR * g11_LR * E1 * E1 \
    #                     + J_LR * g12_LR * E1 * avg(avg(E2, 'UD2C'), 'C2LR') \
    #                     + J_LR * g13_LR * E1 * avg(E3, 'C2LR'))/2.*dx*dy]
    #histEnergyE2 = [np.sum(J_UD * g21_UD * E2 * avg(avg(E1, 'LR2C'), 'C2UD') \
    #                     + J_UD * g22_UD * E2 * E2\
    #                     + J_UD * g23_UD * E2 * avg(E3, 'C2UD'))/2.*dx*dy]
    #histEnergyE3 = [np.sum(J_C * g31_C * E3 * avg(E1, 'LR2C') \
    #                     + J_C * g32_C * E3 * avg(E2, 'UD2C') \
    #                     + J_C * g33_C * E3 * E3)/2.*dx*dy]
    #histEnergyB1 = [np.sum(J_UD * g11_UD * B1 * B1 \
    #                     + J_UD * g12_UD * B1 * avg(avg(B2, 'LR2C'), 'C2UD') \
    #                     + J_UD * g13_UD * B1 * avg(B3, 'N2UD'))/2.*dx*dy]
    #histEnergyB2 =[ np.sum(J_LR * g21_LR * B2 * avg(avg(B1, 'UD2C'), 'C2LR') \
    #                     + J_LR * g22_LR * B2 * B2 \
    #                     + J_LR * g23_LR * B2 * avg(B3, 'N2LR'))/2.*dx*dy]
    #histEnergyB3 = [np.sum(J_N * g31_N * B3 * avg(B1, 'UD2N') \
    #                     + J_N * g32_N * B3 * avg(B2, 'LR2N') \
    #                     + J_N * g33_N * B3 * B3)/2.*dx*dy]

    # in Logical space -> set in natural position yes BC 
    #histEnergyE1 = [np.sum(J_LR[0:nxn-1,:] * g11_LR[0:nxn-1,:] * E1[0:nxn-1,:] * E1[0:nxn-1,:] \
    #                     + J_LR[0:nxn-1,:] * g12_LR[0:nxn-1,:] * E1[0:nxn-1,:] * avg(avg(E2, 'UD2C'), 'C2LR')[0:nxn-1,:] \
    #                     + J_LR[0:nxn-1,:] * g13_LR[0:nxn-1,:] * E1[0:nxn-1,:] * avg(E3, 'C2LR')[0:nxn-1,:])/2.*dx*dy]
    #histEnergyE2 = [np.sum(J_UD[:,0:nyn-1] * g21_UD[:,0:nyn-1] * E2[:,0:nyn-1] * avg(avg(E1, 'LR2C'), 'C2UD')[:,0:nyn-1] \
    #                     + J_UD[:,0:nyn-1] * g22_UD[:,0:nyn-1] * E2[:,0:nyn-1] * E2[:,0:nyn-1] \
    #                     + J_UD[:,0:nyn-1] * g23_UD[:,0:nyn-1] * E2[:,0:nyn-1] * avg(E3, 'C2UD')[:,0:nyn-1])/2.*dx*dy]
    #histEnergyE3 = [np.sum(J_C[:,:] * g31_C[:,:] * E3[:,:] * avg(E1, 'LR2C')[:,:] \
    #                     + J_C[:,:] * g32_C[:,:] * E3[:,:] * avg(E2, 'UD2C')[:,:] \
    #                     + J_C[:,:] * g33_C[:,:] * E3[:,:] * E3[:,:])/2.*dx*dy]
    #histEnergyB1 = [np.sum(J_UD[:,0:nyn-1] * g11_UD[:,0:nyn-1] * B1[:,0:nyn-1] * B1[:,0:nyn-1] \
    #                     + J_UD[:,0:nyn-1] * g12_UD[:,0:nyn-1] * B1[:,0:nyn-1] * avg(avg(B2, 'LR2C'), 'C2UD')[:,0:nyn-1] \
    #                     + J_UD[:,0:nyn-1] * g13_UD[:,0:nyn-1] * B1[:,0:nyn-1] * avg(B3, 'N2UD')[:,0:nyn-1])/2.*dx*dy]
    #histEnergyB2 =[ np.sum(J_LR[0:nxn-1,:] * g21_LR[0:nxn-1,:] * B2[0:nxn-1,:] * avg(avg(B1, 'UD2C'), 'C2LR')[0:nxn-1,:] \
    #                     + J_LR[0:nxn-1,:] * g22_LR[0:nxn-1,:] * B2[0:nxn-1,:] * B2[0:nxn-1,:] \
    #                     + J_LR[0:nxn-1,:] * g23_LR[0:nxn-1,:] * B2[0:nxn-1,:] * avg(B3, 'N2LR')[0:nxn-1,:])/2.*dx*dy]
    #histEnergyB3 = [np.sum(J_N[0:nxn-1,0:nyn-1] * g31_N[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] \
    #                     + J_N[0:nxn-1,0:nyn-1] * g32_N[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] \
    #                     + J_N[0:nxn-1,0:nyn-1] * g33_N[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1])/2.*dx*dy]
    
    # in Logical space -> set in mixed natural position yes BC
    #histEnergyE1 = [np.sum(g11_LR[0:nxn-1, :] * E1[0:nxn-1, :] * E1[0:nxn-1, :] \
    #                + g12_C[:, :] * avg(E1, 'LR2C')[:, :] * avg(E2, 'UD2C')[:, :] \
    #                + g13_LR[0:nxn-1, :] * E1[0:nxn-1, :] * avg(E3, 'C2LR')[0:nxn-1, :])/2.*dx*dy]
    #histEnergyE2 = [np.sum(g21_C[:, :] * avg(E2, 'UD2C')[:, :] * avg(E1, 'LR2C')[:, :] \
    #                + g22_UD[:, 0:nyn-1] * E2[:, 0:nyn-1] * E2[:, 0:nyn-1] \
    #                + g23_UD[:, 0:nyn-1] * E2[:, 0:nyn-1] * avg(E3, 'C2UD')[:, 0:nyn-1])/2.*dx*dy]
    #histEnergyE3 = [np.sum(g31_LR[0:nxn-1, :] * avg(E3, 'C2LR')[0:nxn-1, :] * E1[0:nxn-1, :] \
    #                + g32_UD[:,0:nyn-1] * avg(E3, 'C2UD')[:,0:nyn-1] * E2[:, 0:nyn-1] \
    #                + g33_C[:, :] * E3[:, :] * E3[:, :])/2.*dx*dy]
    #histEnergyB1 = [np.sum(g11_UD[:, 0:nyn-1] * B1[:, 0:nyn-1] * B1[:, 0:nyn-1] \
    #                + g12_N[0:nxn-1, 0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1, 0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1, 0:nyn-1] \
    #                + g13_LR[0:nxn-1, :] * avg(avg(B1, 'UD2C'), 'C2LR')[0:nxn-1, :] * avg(B3, 'N2LR')[0:nxn-1, :])/2.*dx*dy]
    #histEnergyB2 = [np.sum(g21_N[0:nxn-1, 0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1, 0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1, 0:nyn-1] \
    #                + g22_UD[:, 0:nyn-1] * avg(avg(B2, 'LR2N'), 'N2UD')[:,0:nyn-1] * avg(avg(B2, 'LR2C'), 'C2UD')[:,0:nyn-1] \
    #                + g23_UD[:, 0:nyn-1] * avg(avg(B2, 'LR2C'), 'C2UD')[:, 0:nyn-1] * avg(B3, 'N2UD')[:, 0:nyn-1])/2.*dx*dy]
    #histEnergyB3 = [np.sum(g31_LR[0:nxn-1, :] * avg(B3, 'N2LR')[0:nxn-1, :] * avg(avg(B1, 'UD2C'), 'C2LR')[:, :] \
    #                + g32_UD[:, 0:nyn-1] * avg(B3, 'N2UD')[:, 0:nyn-1] * avg(avg(B2, 'LR2C'), 'C2UD')[:, 0:nyn-1] \
    #                + g33_C[:, :] * avg(avg(B3, 'N2UD'), 'UD2C')[:, :] * avg(avg(B3, 'N2LR'), 'LR2C')[:, :])/2.*dx*dy]

    # in Logical space -> in the centre yes BC
    #histEnergyE1 = [np.sum(J_C * g11_C * avg(E1, 'LR2C') * avg(E1, 'LR2C') \
    #                     + J_C * g12_C * avg(E1, 'LR2C') * avg(E2, 'UD2C') \
    #                     + J_C * g13_C * avg(E1, 'LR2C') * E3)/2.*dx*dy] 
    #histEnergyE2 = [np.sum(J_C * g21_C * avg(E2, 'UD2C') * avg(E1, 'LR2C') \
    #                     + J_C * g22_C * avg(E2, 'UD2C') * avg(E2, 'UD2C') \
    #                     + J_C * g23_C * avg(E2, 'UD2C') * E3)/2.*dx*dy ]
    #histEnergyE3 = [np.sum(J_C * g31_C * E3 * avg(E1, 'LR2C') \
    #                     + J_C * g32_C * E3 * avg(E2, 'UD2C') \
    #                     + J_C * g33_C * E3 * E3)/2.*dx*dy]
    #histEnergyB1 = [np.sum(J_C * g11_C * avg(B1, 'UD2C') * avg(B1, 'UD2C')
    #                     + J_C * g12_C * avg(B1, 'UD2C') * avg(B2, 'LR2C') \
    #                     + J_C * g13_C * avg(B1, 'UD2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy] 
    #histEnergyB2 = [np.sum(J_C * g21_C * avg(B2, 'LR2C') * avg(B1, 'UD2C')\
    #                     + J_C * g22_C * avg(B2, 'LR2C') * avg(B2, 'LR2C') \
    #                     + J_C * g23_C * avg(B2, 'LR2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy ]
    #histEnergyB3 = [np.sum(J_C * g31_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(B1, 'UD2C') \
    #                     + J_C * g32_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(B2, 'LR2C') \
    #                     + J_C * g33_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy]

    # in Logical space -> Electric energy in Centre & Magnetic energy in Node
    histEnergyE1 = [np.sum(J_C * g11_C * avg(E1, 'LR2C') * avg(E1, 'LR2C') \
                         + J_C * g12_C * avg(E1, 'LR2C') * avg(E2, 'UD2C') \
                         + J_C * g13_C * avg(E1, 'LR2C') * E3)/2.*dx*dy] 
    histEnergyE2 = [np.sum(J_C * g21_C * avg(E2, 'UD2C') * avg(E1, 'LR2C') \
                         + J_C * g22_C * avg(E2, 'UD2C') * avg(E2, 'UD2C') \
                         + J_C * g23_C * avg(E2, 'UD2C') * E3)/2.*dx*dy ]
    histEnergyE3 = [np.sum(J_C * g31_C * E3 * avg(E1, 'LR2C') \
                         + J_C * g32_C * E3 * avg(E2, 'UD2C') \
                         + J_C * g33_C * E3 * E3)/2.*dx*dy]
    histEnergyB1 = [np.sum(J_N[0:nxn-1,0:nyn-1] * g11_N[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1]
                         + J_N[0:nxn-1,0:nyn-1] * g12_N[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] \
                         + J_N[0:nxn-1,0:nyn-1] * g13_N[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1])/2.*dx*dy] 
    histEnergyB2 = [np.sum(J_N[0:nxn-1,0:nyn-1] * g21_N[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1]\
                         + J_N[0:nxn-1,0:nyn-1] * g22_N[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] 
                         + J_N[0:nxn-1,0:nyn-1] * g23_N[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1])/2.*dx*dy]
    histEnergyB3 = [np.sum(J_N[0:nxn-1,0:nyn-1] * g31_N[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] \
                         + J_N[0:nxn-1,0:nyn-1] * g32_N[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] \
                         + J_N[0:nxn-1,0:nyn-1] * g33_N[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1])/2.*dx*dy]
else:
    histEnergyE1=[np.sum(E1[0:nxn-1,:]**2)/2.*dx*dy]
    histEnergyE2=[np.sum(E2[:,0:nyn-1]**2)/2.*dx*dy]
    histEnergyE3=[np.sum(E3[:,:]**2)/2.*dx*dy]
    histEnergyB1=[np.sum(B1[:,0:nyn-1]**2)/2.*dx*dy]
    histEnergyB2=[np.sum(B2[0:nxn-1,:]**2)/2.*dx*dy]
    histEnergyB3=[np.sum(B3[0:nxn-1,0:nyn-1]**2)/2.*dx*dy]

histEnergyTot=[histEnergyP1[0]+histEnergyP2[0]+histEnergyE1[0]+histEnergyE2[0]+histEnergyE3[0]+histEnergyB1[0]+histEnergyB2[0]+histEnergyB3[0]]

histMomentumx = [np.sum(u[0:npart])/VT1]
histMomentumy = [np.sum(v[0:npart])/VT2]
histMomentumz = [np.sum(w[0:npart])]
histMomentumTot = [histMomentumx[0] + histMomentumy[0] + histMomentumz[0]]

energyP[0] = histEnergyP1[0] + histEnergyP2[0]
energyE[0] = histEnergyE1[0] + histEnergyE2[0] + histEnergyE3[0]
energyB[0] = histEnergyB1[0] + histEnergyB2[0] + histEnergyB3[0]
 
print('cycle 0, energy=',histEnergyTot[0])
print('energyP1=',histEnergyP1[0],'energyP2=',histEnergyP2[0])
print('energyEx=',histEnergyE1[0],'energyEy=',histEnergyE2[0],'energyEz=',histEnergyE3[0])
print('energyBx=',histEnergyB1[0],'energyBy=',histEnergyB2[0],'energyBz=',histEnergyB3[0])
print('momentumx=',histMomentumx[0],'momentumy=',histMomentumy[0],'momentumz=',histMomentumz[0])
  
if metric:
    xgen, ygen = cartesian_to_general_particle(x, y)
else:
    xgen, ygen = x, y

rho_ion = - particle_to_grid_rho(xgen, ygen, q)
rho = particle_to_grid_rho(xgen, ygen, q)
temp = 0
start_loop = time.time()

if plot_save:
    myplot_map(xiN, etaN, B3, title='B3', xlabel='x', ylabel='y')
    filename1 = PATH1 + 'B3_' + '%04d'%temp + '.png'
    plt.savefig(filename1, dpi=ndpi)

    if nppc!=0:
        myplot_particle_map(x, y)
        filename1 = PATH1 + 'part_' + '%04d'%temp + '.png'
        plt.savefig(filename1, dpi=ndpi)
   
        myplot_phase_space(x, v, limx=(0, Lx), limy=(-2*V0x1, 2*V0x1), xlabel='x', ylabel='vx')
        filename1 = PATH1 + 'phase_' + '%04d'%temp + '.png'
        plt.savefig(filename1, dpi=ndpi)

        myplot_map(xiC, etaC, rho, title='rho', xlabel='x', ylabel='y')
        filename1 = PATH1 + 'rho_' + '%04d'%temp + '.png'
        plt.savefig(filename1, dpi=ndpi)

        myplot_map(xiC, etaC, div(E1, E2, E3, 'E') - rho, title='div(E)-rho map', xlabel='x', ylabel='y')
        filename1 = PATH1 + 'div_rho_' + '%04d'%temp + '.png'
        plt.savefig(filename1, dpi=ndpi)

cpu_time = np.zeros(nt+1, np.float64)

for it in range(1,nt+1):
    plt.clf()
    #start = time.time()

    if NK_method:
        # The following is python's NK methods
        #guess = zeros(2*nxn*nyc+2*nxc*nyn+nxc*nxc+nxn*nxn+3*2*part,np.float64)
        guess = phys_to_krylov(E1, E2, E3, u, v, w)
        sol = newton_krylov(residual, guess, method='lgmres', verbose=1, f_tol=1e-18)#, f_rtol=1e-7)
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

    E1new, E2new, E3new, unew, vnew, wnew = krylov_to_phys(sol)

    if relativistic:
        gnew = np.sqrt(1.+unew**2+vnew**2+wnew**2)
        gold = np.sqrt(1.+u**2+v**2+w**2)
        gbar = (gold+gnew)/2.
    else:
        gbar = np.ones(npart)
    
    # position is evolved in physical space 
    # pushed by general geom. fields converted
    ubar = (unew + u)/2.
    vbar = (vnew + v)/2.
    x += ubar/gbar*dt
    y += vbar/gbar*dt
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

    curlB1, curlB2, curlB3 = curl(B1, B2, B3, 'B')

    # TESTING: variation of energy in time through the Poynting Theorem!
    # Def. without metric tensor:
    # delta_U = avg(E1 * curlB1, 'LR2C') + avg(E2 * curlB2, 'UD2C') + E3 * curlB3 \
    #       - (avg(B1 * curlE1, 'UD2C') + avg(B2 * curlE2, 'LR2C') + avg(avg(B3 * curlE3, 'N2UD'), 'UD2C'))
    # Def. without Jacobians and spitting the avgs:
    #delta_U = g11_C * avg(E1*curlB1, 'LR2C') + g12_C * avg(E1, 'LR2C') * avg(curlB2, 'UD2C') + g21_C * avg(E2, 'UD2C') * avg(curlB1, 'LR2C') \
    #        + g22_C * avg(E2*curlB2, 'UD2C') + g33_C * E3*curlB3 \
    #        -( g11_C * avg(B1*curlE1, 'UD2C') + g12_C * avg(B1, 'UD2C') * avg(curlE2, 'LR2C') + g21_C * avg(B2, 'LR2C') * avg(curlE1, 'UD2C') \
    #        + g22_C * avg(B2*curlE2, 'LR2C') + g33_C * avg(avg(B3*curlE3, 'N2UD'), 'UD2C'))
    # Def. with Jacobians and spitting the avgs:
    #delta_U = J_C * (g11_C * avg(E1, 'LR2C') * avg(curlB1, 'LR2C') + g12_C * avg(E1, 'LR2C') * avg(curlB2, 'UD2C') + g21_C * avg(E2, 'UD2C') * avg(curlB1, 'LR2C') \
    #        +        g22_C * avg(E2, 'UD2C') * avg(curlB2, 'UD2C') + g33_C * E3 * curlB3) \
    #        - J_C * (g11_C * avg(B1, 'UD2C') * avg(curlE1, 'UD2C') + g12_C * avg(B1, 'UD2C') * avg(curlE2, 'LR2C') + g21_C * avg(B2, 'LR2C') * avg(curlE1, 'UD2C') \
    #        +        g22_C * avg(B2, 'LR2C') * avg(curlE2, 'LR2C') + g33_C * avg(avg(B3, 'N2UD'), 'UD2C') * avg(avg(curlE3, 'N2UD'), 'UD2C'))
    # Def. Cartesian:
    #delta_U = E1[0:-1, :] * curlB1[0:-1, :] + E2[:, 0:-1] * curlB2[:, 0:-1] + E3 * curlB3 \
    #      - ( B1[:, 0:-1] * curlE1[:, 0:-1] + B2[0:-1, :] * curlE2[0:-1, :] + B3[0:-1, 0:-1] * curlE3[0:-1, 0:-1])
    # Def. Electic energy in Center & Magnetic energy in Nodes:
    delta_U = J_C     * (g11_C * avg(E1, 'LR2C') * avg(curlB1, 'LR2C') \
                       + g12_C * avg(E1, 'LR2C') * avg(curlB2, 'UD2C') \
                       + g21_C * avg(E2, 'UD2C') * avg(curlB1, 'LR2C') \
                       + g22_C * avg(E2, 'UD2C') * avg(curlB2, 'UD2C') \
                       + g33_C * E3 * curlB3) \
    - J_N[0:nxn-1,0:nyn-1] * (g11_N[0:nxn-1,0:nyn-1] * B1[0:nxn-1,0:nyn-1] * curlE1[0:nxn-1,0:nyn-1] \
                       + g12_N[0:nxn-1,0:nyn-1] * B1[0:nxn-1,0:nyn-1] * avg(curlE2, 'LR2N')[0:nxn-1,0:nyn-1] \
                       + g21_N[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] * avg(curlE1, 'UD2N')[0:nxn-1,0:nyn-1] \
                       + g22_N[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] * avg(curlE2, 'LR2N')[0:nxn-1,0:nyn-1] \
                       + g33_N[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1] * curlE3[0:nxn-1,0:nyn-1])
    # Time series of dU, integration in space of delta_U
    histdelta_U[it] = np.sum(delta_U)
    
    if (metric) and (skew):
        # TESTING: covariant curl using skew coord.!
        curlB1_skew, curlB2_skew, curlB3_skew = curl_skewed(B1, B2, B3, 'B')
        curlE1_skew, curlE2_skew, curlE3_skew = curl_skewed(E1bar, E2bar, E3bar, 'E')
        
        delta_curlB1[it-1] = np.sum(curlB1 - curlB1_skew)
        delta_curlB2[it-1] = np.sum(curlB2 - curlB2_skew)
        delta_curlB3[it-1] = np.sum(curlB3 - curlB3_skew)
    
        delta_curlE1[it-1] = np.sum(curlE1 - curlE1_skew)
        delta_curlE2[it-1] = np.sum(curlE2 - curlE2_skew)
        delta_curlE3[it-1] = np.sum(curlE3 - curlE3_skew)

    if metric:
        xgen, ygen = cartesian_to_general_particle(x, y)
    else:
        xgen, ygen = x, y

    rho = particle_to_grid_rho(xgen, ygen, q)
    divE[it] = np.sum(div(E1new, E2new, E3new, 'E'))
    divB[it] = np.sum(div(B1, B2, B3, 'B'))
    divE_rho[it] = np.sum(np.abs(div(E1new, E2new, E3new, 'E')) - np.abs(rho))
    fluxS = poynting_flux()   
    
    E1 = E1new
    E2 = E2new
    E3 = E3new
    
    E1time[it] = np.sum(E1)
    E2time[it] = np.sum(E2)
    E3time[it] = np.sum(E3)
    B1time[it] = np.sum(B1)
    B2time[it] = np.sum(B2)
    B3time[it] = np.sum(B3)

    if relativistic:
        energyP1 = np.sum((g[0:npart1]-1.)*abs(q[0:npart1]/QM[0:npart1]))
        energyP2 = np.sum((g[npart1:npart]-1.)*abs(q[npart1:npart]/QM[npart1:npart]))
    else:
        #energyP1 = 0.5*np.sum((J_C * g11_C * avg(E1 * J1, 'LR2C') + 2. * J_C * avg(g12_LR, 'LR2C') * avg(E1, 'LR2C') * avg(J2, 'UD2C')
        #                     + J_C * g22_C * avg(E2 * J2, 'UD2C') + J_C * g33_C * E3 * J3)/2.*dx*dy)
        #energyP2 = 0.5*np.sum((J_C * g11_C * avg(E1 * J1, 'LR2C') + 2. * J_C * avg(g12_LR, 'LR2C') * avg(E1, 'LR2C') * avg(J2, 'UD2C')
        energyP1 = np.sum((u[0:npart1]**2+v[0:npart1]**2+w[0:npart1]**2)/2.*abs(q[0:npart1]/QM[0:npart1]))
        energyP2 = np.sum((u[npart1:npart]**2+v[npart1:npart]**2 +w[npart1:npart]**2)/2.*abs(q[npart1:npart]/QM[npart1:npart]))
 
    #Bx, By, Bz = general_to_cartesian(B1, B2, B3, 'B')

    if metric:
        # in Physical space -> set in natural position
        #Ex, Ey, Ez = general_to_cartesian(E1, E2, E3, 'E')
        #Bx, By, Bz = general_to_cartesian(B1, B2, B3, 'B')
        #deltax_LR, deltay_LR = general_to_cartesian_delta(dx, dy, 'LR')
        #deltax_UD, deltay_UD = general_to_cartesian_delta(dx, dy, 'UD')
        #deltax_C, deltay_C = general_to_cartesian_delta(dx, dy, 'C')
        #deltax_N, deltay_N = general_to_cartesian_delta(dx, dy, 'N')

        #energyE1 = np.sum(Ex[0:nxn-1,:]**2/2.*deltax_LR[0:nxn-1,:]*deltay_LR[0:nxn-1,:])
        #energyE2 = np.sum(Ey[:,0:nyn-1]**2/2.*deltax_UD[:,0:nyn-1]*deltay_UD[:,0:nyn-1])
        #energyE3 = np.sum(Ez[:,:]**2/2.*deltax_C[:,:]*deltay_C[:,:])
        #energyB1 = np.sum(Bx[:,0:nyn-1]**2/2.*deltax_UD[:,0:nyn-1]*deltay_UD[:,0:nyn-1])
        #energyB2 = np.sum(By[0:nxn-1,:]**2/2.*deltax_LR[0:nxn-1,:]*deltay_LR[0:nxn-1,:])
        #energyB3 = np.sum(Bz[0:nxn-1,0:nyn-1]**2/2.*deltax_N[0:nxn-1,0:nyn-1]*deltay_N[0:nxn-1,0:nyn-1])
        
        # in Logical space -> set in mixed position no BC
        #energyE1 = np.sum(J_LR * g11_LR * E1 * E1 \
        #                + J_LR * g12_LR * E1 * avg(avg(E2, 'UD2C'), 'C2LR') \
        #                + J_LR * g13_LR * E1 * avg(E3, 'C2LR'))/2.*dx*dy
        #energyE2 = np.sum(J_UD * g21_UD * E2 * avg(avg(E1, 'LR2C'), 'C2UD') \
        #                + J_UD * g22_UD * E2 * E2 \
        #                + J_UD * g23_UD * E2 * avg(E3, 'C2UD'))/2.*dx*dy
        #energyE3 = np.sum(J_C * g31_C * E3 * avg(E1, 'LR2C') \
        #                + J_C * g32_C * E3 * avg(E2, 'UD2C') \
        #                + J_C * g33_C * E3 * E3)/2.*dx*dy
        #energyB1 = np.sum(J_UD * g11_UD * B1 * B1 \
        #                + J_UD * g12_UD * B1 * avg(avg(B2, 'LR2C'), 'C2UD') \
        #                + J_UD * g13_UD * B1 * avg(B3, 'N2UD'))/2.*dx*dy
        #energyB2 = np.sum(J_LR * g21_LR * B2 * avg(avg(B1, 'UD2C'), 'C2LR') \
        #                + J_LR * g22_LR * B2 * B2 \
        #                + J_LR * g23_LR * B2 * avg(B3, 'N2LR'))/2.*dx*dy
        #energyB3 = np.sum(J_N * g31_N * B3 * avg(B1, 'UD2N') \
        #                + J_N * g32_N * B3 * avg(B2, 'LR2N') \
        #                + J_N * g33_N * B3 * B3)/2.*dx*dy

        # in Logical space -> set in natural position yes BC
        #energyE1 = np.sum(g11_LR[0:nxn-1, :] * E1[0:nxn-1, :] * E1[0:nxn-1, :] \
        #                + g12_C[:, :] * avg(E1, 'LR2C')[:, :] * avg(E2, 'UD2C')[:, :] \
        #                + g13_LR[0:nxn-1, :] * E1[0:nxn-1, :] * avg(E3, 'C2LR')[0:nxn-1, :])/2.*dx*dy
        #energyE2 = np.sum(g21_C[:, :] * avg(E2, 'UD2C')[:, :] * avg(E1, 'LR2C')[:, :] \
        #                + g22_UD[:, 0:nyn-1] * E2[:, 0:nyn-1] * E2[:, 0:nyn-1] \
        #                + g23_UD[:, 0:nyn-1] * E2[:, 0:nyn-1] * avg(E3, 'C2UD')[:, 0:nyn-1])/2.*dx*dy
        #energyE3 = np.sum(g31_LR[0:nxn-1, :] * avg(E3, 'C2LR')[0:nxn-1, :] * E1[0:nxn-1, :] \
        #                + g32_UD[:,0:nyn-1] * avg(E3, 'C2UD')[:,0:nyn-1] * E2[:, 0:nyn-1] \
        #                + g33_C[:, :] * E3[:, :] * E3[:, :])/2.*dx*dy
        #energyB1 = np.sum(g11_UD[:, 0:nyn-1] * B1[:, 0:nyn-1] * B1[:, 0:nyn-1] \
        #                + g12_N[0:nxn-1, 0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1, 0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1, 0:nyn-1] \
        #                + g13_LR[0:nxn-1, :] * avg(avg(B1, 'UD2C'), 'C2LR')[0:nxn-1, :] * avg(B3, 'N2LR')[0:nxn-1, :])/2.*dx*dy
        #energyB2 = np.sum(g21_N[0:nxn-1, 0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1, 0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1, 0:nyn-1] \
        #                + g22_UD[:,0:nyn-1] * avg(avg(B2, 'LR2N'), 'N2UD')[:,0:nyn-1] * avg(avg(B2, 'LR2C'), 'C2UD')[:,0:nyn-1] \
        #                + g23_UD[:,0:nyn-1] * avg(avg(B2, 'LR2C'), 'C2UD')[:,0:nyn-1] * avg(B3, 'N2UD')[0:nxn-1,:])/2.*dx*dy
        #energyB3 = np.sum(g31_LR[0:nxn-1, :] * avg(B3, 'N2LR')[0:nxn-1, :] * avg(avg(B1, 'UD2C'), 'C2LR')[:, :] \
        #                + g32_UD[:,0:nyn-1] * avg(B3, 'N2UD')[:,0:nyn-1] * avg(avg(B2, 'LR2C'), 'C2UD')[:,0:nyn-1] \
        #                + g33_C[:, :] * avg(avg(B3, 'N2UD'), 'UD2C')[:, :] * avg(avg(B3, 'N2UD'), 'UD2C')[:, :])/2.*dx*dy

        # in Logical space -> in the centre yes BC
        #energyE1 = np.sum(J_C * g11_C * avg(E1, 'LR2C') * avg(E1, 'LR2C')
        #                    + J_C * g12_C * avg(E1, 'LR2C') * avg(E2, 'UD2C') \
        #                    + J_C * g13_C * avg(E1, 'LR2C') * E3)/2.*dx*dy
        #energyE2 = np.sum(J_C * g21_C * avg(E2, 'UD2C') * avg(E1, 'LR2C') \
        #                    + J_C * g22_C * avg(E2, 'UD2C') * avg(E2, 'UD2C') \
        #                    + J_C * g23_C * avg(E2, 'UD2C') * E3)/2.*dx*dy
        #energyE3 = np.sum(J_C * g31_C * E3 * avg(E1, 'LR2C') \
        #                    + J_C * g32_C * E3 * avg(E2, 'UD2C') \
        #                    + J_C * g33_C * E3 * E3)/2.*dx*dy
        #energyB1 = np.sum(J_C * g11_C * avg(B1, 'UD2C') * avg(B1, 'UD2C')
        #                    + J_C * g12_C * avg(B1, 'UD2C') * avg(B2, 'LR2C') \
        #                    + J_C * g13_C * avg(B1, 'UD2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy
        #energyB2 = np.sum(J_C * g21_C * avg(B2, 'LR2C') * avg(B1, 'UD2C')\
        #                    + J_C * g22_C * avg(B2, 'LR2C') * avg(B2, 'LR2C') \
        #                    + J_C * g23_C * avg(B2, 'LR2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy
        #energyB3 = np.sum(J_C * g31_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(B1, 'UD2C') \
        #                    + J_C * g32_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(B2, 'LR2C') \
        #                    + J_C * g33_C * avg(avg(B3, 'N2LR'), 'LR2C') * avg(avg(B3, 'N2LR'), 'LR2C'))/2.*dx*dy

        # in Logical space -> Electric energy in Centre & Magnetic energy in Node
        energyE1 = np.sum(J_C * g11_C * avg(E1, 'LR2C') * avg(E1, 'LR2C') \
                        + J_C * g12_C * avg(E1, 'LR2C') * avg(E2, 'UD2C') \
                        + J_C * g13_C * avg(E1, 'LR2C') * E3)/2.*dx*dy  
        energyE2 = np.sum(J_C * g21_C * avg(E2, 'UD2C') * avg(E1, 'LR2C') \
                        + J_C * g22_C * avg(E2, 'UD2C') * avg(E2, 'UD2C') \
                        + J_C * g23_C * avg(E2, 'UD2C') * E3)/2.*dx*dy 
        energyE3 = np.sum(J_C * g31_C * E3 * avg(E1, 'LR2C') \
                        + J_C * g32_C * E3 * avg(E2, 'UD2C') \
                        + J_C * g33_C * E3 * E3)/2.*dx*dy
        energyB1 = np.sum(J_N[0:nxn-1,0:nyn-1] * g11_N[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] \
                        + J_N[0:nxn-1,0:nyn-1] * g12_N[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] \
                        + J_N[0:nxn-1,0:nyn-1] * g13_N[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1])/2.*dx*dy 
        energyB2 = np.sum(J_N[0:nxn-1,0:nyn-1] * g21_N[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] \
                        + J_N[0:nxn-1,0:nyn-1] * g22_N[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] \
                        + J_N[0:nxn-1,0:nyn-1] * g23_N[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1])/2.*dx*dy 
        energyB3 = np.sum(J_N[0:nxn-1,0:nyn-1] * g31_N[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1] * avg(B1, 'UD2N')[0:nxn-1,0:nyn-1] \
                        + J_N[0:nxn-1,0:nyn-1] * g32_N[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1] * avg(B2, 'LR2N')[0:nxn-1,0:nyn-1] \
                        + J_N[0:nxn-1,0:nyn-1] * g33_N[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1] * B3[0:nxn-1,0:nyn-1])/2.*dx*dy
    else:
        energyE1 = np.sum(E1[0:nxn-1,:]**2)/2.*dx*dy
        energyE2 = np.sum(E2[:,0:nyn-1]**2)/2.*dx*dy
        energyE3 = np.sum(E3[:,:]**2)/2.*dx*dy
        energyB1 = np.sum(B1[:,0:nyn-1]**2)/2.*dx*dy
        energyB2 = np.sum(B2[0:nxn-1,:]**2)/2.*dx*dy
        energyB3 = np.sum(B3[0:nxn-1,0:nyn-1]**2)/2.*dx*dy
    
    energyTot = energyP1 + energyP2 + energyE1 + energyE2 + energyE3 + energyB1 + energyB2 + energyB3

    momentumx = np.sum(unew[0:npart]/VT1)
    momentumy = np.sum(vnew[0:npart]/VT2)   
    momentumz = np.sum(wnew[0:npart])
    momentumTot = momentumx + momentumy + momentumz
    mod_vel = np.sqrt(unew**2 + vnew**2 + wnew**2)

    histEnergyP1.append(energyP1)
    histEnergyP2.append(energyP2)
    histEnergyE1.append(energyE1)
    histEnergyE2.append(energyE2)
    histEnergyE3.append(energyE3)
    histEnergyB1.append(energyB1)
    histEnergyB2.append(energyB2)
    histEnergyB3.append(energyB3)
    histEnergyTot.append(energyTot)
    
    histMomentumx.append(momentumx)
    histMomentumy.append(momentumy)
    histMomentumz.append(momentumz)
    histMomentumTot.append(momentumTot)

    energyP[it] = histEnergyP1[it] + histEnergyP2[it]
    energyE[it] = histEnergyE1[it] + histEnergyE2[it] + histEnergyE3[it]
    energyB[it] = histEnergyB1[it] + histEnergyB2[it] + histEnergyB3[it]
    
    err_en[it] = (histEnergyTot[it] - histEnergyTot[it-1])/histEnergyTot[it-1]

    print('cycle',it,'energy =',histEnergyTot[it])
    print('energyP1=',histEnergyP1[it],'energyP2=',histEnergyP2[it])
    print('energyE1=',histEnergyE1[it],'energyE2=',histEnergyE2[it],'energyE3=',histEnergyE3[it])
    print('energyB1=',histEnergyB1[it],'energyB2=',histEnergyB2[it],'energyB3=',histEnergyB3[it])
    print('relative energy change=',(histEnergyTot[it]-histEnergyTot[0])/histEnergyTot[0])
    print('total momentum= ', histMomentumTot[it])
    print('')

    if plot_each:
        plt.figure(figsize=(12, 9))

        plt.subplot(2, 3, 1)
        plt.pcolor(xiN, etaN, B3)
        plt.title('B3 map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

        #plt.subplot(2, 3, 2)
        #plt.pcolor(xiUD, etaUD, E2)
        #plt.title('E_y map')
        #plt.xlabel('x')
        #plt.ylabel('y')
        #plt.colorbar()

        #plt.subplot(2, 3, 3)
        #plt.pcolor(xiC, etaC, E3)
        #plt.title('E_z map')
        #plt.xlabel('x')
        #plt.ylabel('y')
        #plt.colorbar()

        plt.subplot(2, 3, 2)
        plt.pcolor(xiC, etaC, rho)  
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
        plt.xlim((0, Lx))
        plt.ylim((0, Ly))
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

    stop_loop = time.time()
    if plot_save:
        if it == nt:
            myplot_time_series(histEnergyTot, ylabel=r'$U_{tot}$', title='')
            filename1 = PATH1 + '@energy_tot_' + '%04d' % it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            #myplot_time_series((histEnergyTot-histEnergyTot[0])/histEnergyTot[0], ylabel='err(E)', title='')
            #filename1 = PATH1 + '@error_rel_E[0]_' + '%04d' % it + '.png'
            #plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(err_en, ylabel='err(E)', title='')
            filename1 = PATH1 + '@error_rel_E[t-1]_' + '%04d' % it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(energyB, ylabel=r'$U_{mag}$', title='')
            filename1 = PATH1 + '@energy_mag_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(energyE, ylabel=r'$U_{el}$', title='')
            filename1 = PATH1 + '@energy_elec_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)
            
            myplot_time_series(energyP, ylabel=r'$U_{part}$', title='')
            filename1 = PATH1 + '@energy_part_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(histMomentumTot, ylabel='p', title='')
            filename1 = PATH1 + '@momentumTot_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(histMomentumx, ylabel='p', title='')
            filename1 = PATH1 + '@momentumx_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(histMomentumy, ylabel='p', title='')
            filename1 = PATH1 + '@momentumy_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(histMomentumz, ylabel='p', title='')
            filename1 = PATH1 + '@momentumz_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(divE_rho, ylabel='div', title='')
            filename1 = PATH1 + '@div(E)-rho_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(divE, ylabel='div(E)', title='')
            filename1 = PATH1 + '@div(E)_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(divB, ylabel='div(B)', title='')
            filename1 = PATH1 + '@div(B)_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(B1time, ylabel=r'$B^1$', title='')
            filename1 = PATH1 + '@B1_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(B2time, ylabel=r'$B^2$', title='')
            filename1 = PATH1 + '@B2_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(B3time[1:it+1], ylabel=r'$B^3$', title='')
            filename1 = PATH1 + '@B3_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(E1time, ylabel=r'$E^1$', title='')
            filename1 = PATH1 + '@E1_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(E2time, ylabel=r'$E^2$', title='')
            filename1 = PATH1 + '@E2_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(E3time, ylabel=r'$E^3$', title='')
            filename1 = PATH1 + '@E3_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_time_series(histdelta_U, ylabel=r'$\delta_U$', title='')
            filename1 = PATH1 + '@delta_U_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            if (metric) and (skew):
                myplot_time_series(delta_curlB1, ylabel=r'$\delta$', title='')
                filename1 = PATH1 + 'delta_curlB1_' + '%04d' % it + '.png'
                plt.savefig(filename1, dpi=ndpi)
    
                myplot_time_series(delta_curlB2, ylabel=r'$\delta$', title='')
                filename1 = PATH1 + 'delta_curlB2_' + '%04d' % it + '.png'
                plt.savefig(filename1, dpi=ndpi)
    
                myplot_time_series(delta_curlB3, ylabel=r'$\delta$', title='')
                filename1 = PATH1 + 'delta_curlB3_' + '%04d' % it + '.png'
                plt.savefig(filename1, dpi=ndpi)
    
                myplot_time_series(delta_curlE1, ylabel=r'$\delta$', title='')
                filename1 = PATH1 + 'delta_curlE1_' + '%04d' % it + '.png'
                plt.savefig(filename1, dpi=ndpi)
    
                myplot_time_series(delta_curlE2, ylabel=r'$\delta$', title='')
                filename1 = PATH1 + 'delta_curlE2_' + '%04d' % it + '.png'
                plt.savefig(filename1, dpi=ndpi)
    
                myplot_time_series(delta_curlE3, ylabel=r'$\delta$', title='')
                filename1 = PATH1 + 'delta_curlE3_' + '%04d' % it + '.png'
                plt.savefig(filename1, dpi=ndpi)
            
        if (it % every == 0) or (it == 1):
            if nppc!=0:
                #myplot_particle_hystogram(mod_vel)
                #filename1 = PATH1 + 'part_hyst_mod_' + '%04d'%it + '.png'
                #plt.savefig(filename1, dpi=ndpi) 

                myplot_particle_hystogram(unew)
                filename1 = PATH1 + 'part_hyst_u_' + '%04d'%it + '.png'
                plt.savefig(filename1, dpi=ndpi) 

                myplot_particle_hystogram(vnew)
                filename1 = PATH1 + 'part_hyst_v_' + '%04d'%it + '.png'
                plt.savefig(filename1, dpi=ndpi) 

                myplot_particle_hystogram(wnew)
                filename1 = PATH1 + 'part_hyst_w_' + '%04d'%it + '.png'
                plt.savefig(filename1, dpi=ndpi) 

                myplot_particle_map(x, y)
                filename1 = PATH1 + 'part_' + '%04d'%it + '.png'
                plt.savefig(filename1, dpi=ndpi) 

                myplot_phase_space(x, u, limx=(0, Lx), limy=(-2*V0x1, 2*V0x1), xlabel='x', ylabel=r'$v_x$')
                filename1 = PATH1 + 'phase_' + '%04d'%it + '.png'
                plt.savefig(filename1, dpi=ndpi)

                myplot_map(xiC, etaC, rho, xlabel='x', ylabel='y', title='')
                filename1 = PATH1 + 'rho_' + '%04d'%it + '.png'
                plt.savefig(filename1, dpi=ndpi)
            
            myplot_pert_map(xN, yN, B3, xlabel='x', ylabel='y', title='')
            filename1 = PATH1 + 'Bz_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_map(xiN, etaN, B3, xlabel='x', ylabel='y', title='')
            filename1 = PATH1 + 'B3_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_map(xiLR, etaLR, E1, xlabel='x', ylabel='y', title='')
            filename1 = PATH1 + 'E1_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_map(xiUD, etaUD, E2, xlabel='x', ylabel='y', title='')
            filename1 = PATH1 + 'E2_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_map(xiC, etaC, fluxS, xlabel='x', ylabel='y', title='')
            filename1 = PATH1 + 'S_' + '%04d'%it + '.png'
            plt.savefig(filename1, dpi=ndpi)

            myplot_map(xiC, etaC, delta_U, xlabel='x', ylabel='y', title='')
            filename1 = PATH1 + 'delta_U_' + '%04d' % it + '.png'
            plt.savefig(filename1, dpi=ndpi)
                    
    if data_save:
        f = open(PATH1+"iPiC_2D3V_cov_yee.dat", "a")
        print(it, np.sum(E1), np.sum(E2), np.sum(E3), np.sum(B1), np.sum(B2), np.sum(B3),\
                  energyE1, energyE2, energyE3, energyB1, energyB2, energyB3, energyE1+energyE2+energyE3, energyB1+energyB2+energyB3, energyP1, energyP2, histEnergyTot[it],\
                  divE[it], divB[it], file=f)
        f.close()

print('Geometry initialisation cpu time:', stop_geom - start_geom)
print('Time integration cpu time:', stop_loop - start_loop)
