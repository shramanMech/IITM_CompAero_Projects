import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

file1 = 'meshfile.txt'

T_inf = 300
R_air = 287
M_inf = 0.8
p_stat = 1000                          # Freestream static pressure is given to be 0.01 atm ~ 1e3 Pa
gamma = 1.4
rho_inf = p_stat/(R_air*T_inf)
a_inf = np.sqrt(gamma*p_stat/rho_inf)
u_inf = M_inf*a_inf
v_inf = 0
cfl = 0.5
tol = 1e-12
if M_inf < 1:
    mode = 'sub'
else:
    mode = 'sup'
restart = False
if restart:
    last_itr = 145000
    last_time = 0.7649423087785011
high_recon = True

#---------- Function definitions ----------#
# Boundary Conditions
def in_bc(dim, r, mode):
    if mode == 'sub':
        rho_l = rho_inf
        u_l = u_inf
        v_l = v_inf
        p_l = r[3, 0, :]

    if mode == 'sup':
        rho_l = rho_inf
        u_l = u_inf
        v_l = v_inf
        p_l = p_stat

    return rho_l, u_l, v_l, p_l

def out_bc(dim, r, mode):
    if mode == 'sub':
        rho_r = r[0, dim[0]-2, :]
        u_r = r[1, dim[0]-2, :]
        v_r = r[2, dim[0]-2, :]
        p_r = p_stat

    if mode == 'sup':
        rho_r = r[0, dim[0]-2, :]
        u_r = r[1, dim[0]-2, :]
        v_r = r[2, dim[0]-2, :]
        p_r = r[3, dim[0]-2, :]

    return rho_r, u_r, v_r, p_r

def slipwall(dim, r, yfacesA, yfacesN):
    rho_d = r[0, :, 0]
    nx = yfacesN[:,0,0]/yfacesA[:,0]
    ny = yfacesN[:,0,1]/yfacesA[:,0]
    u_d = (r[1, :, 0]*(ny**2-nx**2)-2*r[2, :, 0]*nx*ny)/(nx**2+ny**2)
    v_d = (r[2, :, 0]*(nx**2-ny**2)-2*r[1, :, 0]*nx*ny)/(nx**2+ny**2)
    p_d = r[3, :, 0]
    rho_u = r[0, :, dim[1]-2]
    p_u = r[3, :, dim[1]-2]

    return rho_d, p_d, rho_u, p_u, u_d, v_d

# Defining limitters
def superbee(a,b):
    r1 = np.minimum(2*np.abs(a), np.abs(b))
    r2 = np.minimum(np.abs(a), 2*np.abs(b))
    phi = np.maximum(0, np.maximum(r1, r2))
    return np.sign(a) * phi

def minmod(a,b):
    s = np.sign(a) + np.sign(b)
    return 0.5 * s * np.minimum(np.abs(a), np.abs(b))

# State reconstruction at faces
def muscl_reco(dim, r):

    # Face state arrays
    rl = np.zeros_like(r)           # Left face states
    rr = np.zeros_like(r)           # Right face states
    rd = np.zeros_like(r)           # Lower face states
    ru = np.zeros_like(r)           # Upper face states

    for i in range(1, dim[0]-2):
        drpl0 = r[0,i+1,:] - r[0,i,:]
        drmi0 = r[0,i,:] - r[0,i-1,:]
        drpl1 = r[1,i+1,:] - r[1,i,:]
        drmi1 = r[1,i,:] - r[1,i-1,:]
        drpl2 = r[2,i+1,:] - r[2,i,:]
        drmi2 = r[2,i,:] - r[2,i-1,:]
        drpl3 = r[3,i+1,:] - r[3,i,:]
        drmi3 = r[3,i,:] - r[3,i-1,:]

        rl[0,i,:] = r[0,i,:] + 0.5*minmod(drpl0,drmi0)
        rr[0,i,:] = r[0,i,:] - 0.5*minmod(drpl0,drmi0)
        rl[1,i,:] = r[1,i,:] + 0.5*minmod(drpl1,drmi1)
        rr[1,i,:] = r[1,i,:] - 0.5*minmod(drpl1,drmi1)
        rl[2,i,:] = r[2,i,:] + 0.5*minmod(drpl2,drmi2)
        rr[2,i,:] = r[2,i,:] - 0.5*minmod(drpl2,drmi2)
        rl[3,i,:] = r[3,i,:] + 0.5*minmod(drpl3,drmi3)
        rr[3,i,:] = r[3,i,:] - 0.5*minmod(drpl3,drmi3)

    # Left cells
    drpl0 = r[0,1,:] - r[0,0,:]
    drmi0 = r[0,0,:] - in_bc(dim,r,mode)[0]
    drpl1 = r[1,1,:] - r[1,0,:]
    drmi1 = r[1,0,:] - in_bc(dim,r,mode)[1]
    drpl2 = r[2,1,:] - r[2,0,:]
    drmi2 = r[2,0,:] - in_bc(dim,r,mode)[2]
    drpl3 = r[3,1,:] - r[3,0,:]
    drmi3 = r[3,0,:] - in_bc(dim,r,mode)[3]

    rl[0,0,:] = r[0,0,:] + 0.5*minmod(drpl0,drmi0)
    rr[0,0,:] = r[0,0,:] - 0.5*minmod(drpl0,drmi0)
    rl[1,0,:] = r[1,0,:] + 0.5*minmod(drpl1,drmi1)
    rr[1,0,:] = r[1,0,:] - 0.5*minmod(drpl1,drmi1)
    rl[2,0,:] = r[2,0,:] + 0.5*minmod(drpl2,drmi2)
    rr[2,0,:] = r[2,0,:] - 0.5*minmod(drpl2,drmi2)
    rl[3,0,:] = r[3,0,:] + 0.5*minmod(drpl3,drmi3)
    rr[3,0,:] = r[3,0,:] - 0.5*minmod(drpl3,drmi3)

    # Right cells
    drpl0 = out_bc(dim,r,mode)[0] - r[0,dim[0]-2,:]
    drmi0 = r[0,dim[0]-2,:] - r[0,dim[0]-3,:]
    drpl1 = out_bc(dim,r,mode)[1] - r[1,dim[0]-2,:]
    drmi1 = r[1,dim[0]-2,:] - r[1,dim[0]-3,:]
    drpl2 = out_bc(dim,r,mode)[2] - r[2,dim[0]-2,:]
    drmi2 = r[2,dim[0]-2,:] - r[2,dim[0]-3,:]
    drpl3 = out_bc(dim,r,mode)[3] - r[3,dim[0]-2,:]
    drmi3 = r[3,dim[0]-2,:] - r[3,dim[0]-3,:]

    rl[0,dim[0]-2,:] = r[0,dim[0]-2,:] + 0.5*minmod(drpl0,drmi0)
    rr[0,dim[0]-2,:] = r[0,dim[0]-2,:] - 0.5*minmod(drpl0,drmi0)
    rl[1,dim[0]-2,:] = r[1,dim[0]-2,:] + 0.5*minmod(drpl1,drmi1)
    rr[1,dim[0]-2,:] = r[1,dim[0]-2,:] - 0.5*minmod(drpl1,drmi1)
    rl[2,dim[0]-2,:] = r[2,dim[0]-2,:] + 0.5*minmod(drpl2,drmi2)
    rr[2,dim[0]-2,:] = r[2,dim[0]-2,:] - 0.5*minmod(drpl2,drmi2)
    rl[3,dim[0]-2,:] = r[3,dim[0]-2,:] + 0.5*minmod(drpl3,drmi3)
    rr[3,dim[0]-2,:] = r[3,dim[0]-2,:] - 0.5*minmod(drpl3,drmi3)

    for j in range(1, dim[1]-2):
        drpl0 = r[0,:,j+1] - r[0,:,j]
        drmi0 = r[0,:,j] - r[0,:,j-1]
        drpl1 = r[1,:,j+1] - r[1,:,j]
        drmi1 = r[1,:,j] - r[1,:,j-1]
        drpl2 = r[2,:,j+1] - r[2,:,j]
        drmi2 = r[2,:,j] - r[2,:,j-1]
        drpl3 = r[3,:,j+1] - r[3,:,j]
        drmi3 = r[3,:,j] - r[3,:,j-1]

        rd[0,:,j] = r[0,:,j] + 0.5*minmod(drpl0,drmi0)
        ru[0,:,j] = r[0,:,j] - 0.5*minmod(drpl0,drmi0)
        rd[1,:,j] = r[1,:,j] + 0.5*minmod(drpl1,drmi1)
        ru[1,:,j] = r[1,:,j] - 0.5*minmod(drpl1,drmi1)
        rd[2,:,j] = r[2,:,j] + 0.5*minmod(drpl2,drmi2)
        ru[2,:,j] = r[2,:,j] - 0.5*minmod(drpl2,drmi2)
        rd[3,:,j] = r[3,:,j] + 0.5*minmod(drpl3,drmi3)
        ru[3,:,j] = r[3,:,j] - 0.5*minmod(drpl3,drmi3)

    # Lower cells
    drpl0 = r[0,:,1] - r[0,:,0]
    drmi0 = r[0,:,0] - slipwall(dim,r,yfacesA,yfacesN)[0]
    drpl1 = r[1,:,1] - r[1,:,0]       
    drmi1 = r[1,:,0] - slipwall(dim,r,yfacesA,yfacesN)[4]
    drpl2 = r[2,:,1] - r[2,:,0]       
    drmi2 = r[2,:,0] - slipwall(dim,r,yfacesA,yfacesN)[5]
    drpl3 = r[3,:,1] - r[3,:,0]
    drmi3 = r[3,:,0] - slipwall(dim,r,yfacesA,yfacesN)[1]

    rd[0,:,0] = r[0,:,0] + 0.5*minmod(drpl0,drmi0)
    ru[0,:,0] = r[0,:,0] - 0.5*minmod(drpl0,drmi0)
    rd[1,:,0] = r[1,:,0] + 0.5*minmod(drpl1,drmi1)
    ru[1,:,0] = r[1,:,0] - 0.5*minmod(drpl1,drmi1)
    rd[2,:,0] = r[2,:,0] + 0.5*minmod(drpl2,drmi2)
    ru[2,:,0] = r[2,:,0] - 0.5*minmod(drpl2,drmi2)
    rd[3,:,0] = r[3,:,0] + 0.5*minmod(drpl3,drmi3)
    ru[3,:,0] = r[3,:,0] - 0.5*minmod(drpl3,drmi3)

    # Upper cells
    drpl0 = slipwall(dim,r,yfacesA,yfacesN)[2] - r[0,:,dim[1]-2]
    drmi0 = r[0,:,dim[1]-2] - r[0,:,dim[1]-3]
    drpl1 = 0
    drmi1 = r[1,:,dim[1]-2] - r[1,:,dim[1]-3]
    drpl2 = -2*(r[2,:,dim[1]-2])
    drmi2 = r[2,:,dim[1]-2] - r[2,:,dim[1]-3]
    drpl3 = slipwall(dim,r,yfacesA,yfacesN)[3] - r[3,:,dim[1]-2]
    drmi3 = r[3,:,dim[1]-2] - r[3,:,dim[1]-3]

    rd[0,:,dim[1]-2] = r[0,:,dim[1]-2] + 0.5*minmod(drpl0,drmi0)
    ru[0,:,dim[1]-2] = r[0,:,dim[1]-2] - 0.5*minmod(drpl0,drmi0)
    rd[1,:,dim[1]-2] = r[1,:,dim[1]-2] + 0.5*minmod(drpl1,drmi1)
    ru[1,:,dim[1]-2] = r[1,:,dim[1]-2] - 0.5*minmod(drpl1,drmi1)
    rd[2,:,dim[1]-2] = r[2,:,dim[1]-2] + 0.5*minmod(drpl2,drmi2)
    ru[2,:,dim[1]-2] = r[2,:,dim[1]-2] - 0.5*minmod(drpl2,drmi2)
    rd[3,:,dim[1]-2] = r[3,:,dim[1]-2] + 0.5*minmod(drpl3,drmi3)
    ru[3,:,dim[1]-2] = r[3,:,dim[1]-2] - 0.5*minmod(drpl3,drmi3)
    
    return rl, rr, rd, ru

# AUSM fluxes (with MUSCL reconstruction)
def fluxls_muscl(dim, r, rl, rr, rd, ru):
    
    # Fluxes initialization
    xFluxes = np.zeros((dim[0], dim[1]-1, 4))        # for SM: Same shape as xfacesA + a dimension with value 4 for the 4 eq's
    yFluxes = np.zeros((dim[0]-1, dim[1], 4))        # for SM: Same shape as yfacesA + a dimension with value 4 for the 4 eq's

    # X-face fluxes
    for i in range(1, dim[0]-1):
        area = xfacesA[i,:]
        nx = xfacesN[i,:,0]/area
        ny = xfacesN[i,:,1]/area
        # Left states
        rho_l = rr[0, i-1, :]
        u_l = rr[1, i-1, :]
        v_l = rr[2, i-1, :]
        p_l = rr[3, i-1, :]
        h0_l = (gamma/(gamma - 1))*(p_l/rho_l) + 0.5 * (u_l**2 + v_l**2)
        a_l = np.sqrt(gamma*(p_l/rho_l))
        u_nl = u_l*nx + v_l*ny
        # Right states
        rho_r = rl[0, i, :]
        u_r = rl[1, i, :]
        v_r = rl[2, i, :]
        p_r = rl[3, i, :]
        h0_r = (gamma/(gamma - 1))*(p_r/rho_r) + 0.5 * (u_r**2 + v_r**2)
        a_r = np.sqrt(gamma*(p_r/rho_r))
        u_nr = u_r*nx + v_r*ny

        a_avg = 0.5*(a_l + a_r)
        xma_l = u_nl/a_avg
        xma_r = u_nr/a_avg

        al_l = 0.5*(1.0 + np.sign((xma_l)))
        al_r = 0.5*(1.0 - np.sign((xma_r)))

        be_l = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_l)))
        be_r = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_r)))

        # Defining the van Leer polynomials
        d_pl = 0.25*(((xma_l+1)**2)*(2-xma_l))
        d_mi = 0.25*(((xma_r-1)**2)*(2+xma_r))

        d_l = al_l*(1+be_l) - be_l*d_pl
        d_r = al_r*(1+be_r) - be_r*d_mi

        cvl_pl = al_l * (1+be_l) * xma_l - be_l * 0.25 * (1+xma_l)**2
        cvl_mi = al_r * (1+be_r) * xma_r + be_r * 0.25 * (1-xma_r)**2

        # Using the van Leer polynoials to define the Liou-Stefan polynomials
        cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
        cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

        # Defining the mass-flux using the LS formulation
        fm_pl = area * a_avg * cls_pl * rho_l
        fm_mi = area * a_avg * cls_mi * rho_r
        xFluxes[i,:, 0] = fm_pl + fm_mi
        xFluxes[i,:, 1] = fm_pl * u_l + fm_mi * u_r + (d_l*p_l + d_r*p_r) * nx*area
        xFluxes[i,:, 2] = fm_pl * v_l + fm_mi * v_r + (d_l*p_l + d_r*p_r) * ny*area
        xFluxes[i,:, 3] = fm_pl * h0_l + fm_mi * h0_r

    # Left boundary flux
    area = xfacesA[0,:]
    nx = xfacesN[0,:,0]/area
    ny = xfacesN[0,:,1]/area
    # Left states
    bc_l = in_bc(dim, r, mode)
    rho_l = bc_l[0]
    u_l = bc_l[1]
    v_l = bc_l[2]
    p_l = bc_l[3]
    h0_l = (gamma/(gamma - 1))*(p_l/rho_l) + 0.5 * (u_l**2 + v_l**2)
    a_l = np.sqrt(gamma*(p_l/rho_l))
    u_nl = u_l*nx + v_l*ny
    # Right states
    rho_r = rl[0, 0, :]
    u_r = rl[1, 0, :]
    v_r = rl[2, 0, :]
    p_r = rl[3, 0, :]
    h0_r = (gamma/(gamma - 1))*(p_r/rho_r) + 0.5 * (u_r**2 + v_r**2)
    a_r = np.sqrt(gamma*(p_r/rho_r))
    u_nr = u_r*nx + v_r*ny

    a_avg = 0.5*(a_l + a_r)
    xma_l = u_nl/a_avg
    xma_r = u_nr/a_avg

    al_l = 0.5*(1.0 + np.sign((xma_l)))
    al_r = 0.5*(1.0 - np.sign((xma_r)))

    be_l = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_l)))
    be_r = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_r)))

    # Defining the van Leer polynomials
    d_pl = 0.25*(((xma_l+1)**2)*(2-xma_l))
    d_mi = 0.25*(((xma_r-1)**2)*(2+xma_r))

    d_l = al_l*(1+be_l) - be_l*d_pl
    d_r = al_r*(1+be_r) - be_r*d_mi

    cvl_pl = al_l * (1+be_l) * xma_l - be_l * 0.25 * (1+xma_l)**2
    cvl_mi = al_r * (1+be_r) * xma_r + be_r * 0.25 * (1-xma_r)**2

    # Using the van Leer polynoials to define the Liou-Stefan polynomials
    cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
    cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

    # Defining the mass-flux using the LS formulation
    fm_pl = area * a_avg * cls_pl * rho_l
    fm_mi = area * a_avg * cls_mi * rho_r
    xFluxes[0,:, 0] = fm_pl + fm_mi
    xFluxes[0,:, 1] = fm_pl * u_l + fm_mi * u_r + (d_l*p_l + d_r*p_r) * nx*area
    xFluxes[0,:, 2] = fm_pl * v_l + fm_mi * v_r + (d_l*p_l + d_r*p_r) * ny*area
    xFluxes[0,:, 3] = fm_pl * h0_l + fm_mi * h0_r
    
    # Right boundary flux
    area = xfacesA[dim[0]-1,:]
    nx = xfacesN[dim[0]-1,:,0]/area
    ny = xfacesN[dim[0]-1,:,1]/area
    # Left states
    rho_l = rr[0, dim[0]-2, :]
    u_l = rr[1, dim[0]-2, :]
    v_l = rr[2, dim[0]-2, :]
    p_l = rr[3, dim[0]-2, :]
    h0_l = (gamma/(gamma - 1))*(p_l/rho_l) + 0.5 * (u_l**2 + v_l**2)
    a_l = np.sqrt(gamma*(p_l/rho_l))
    u_nl = u_l*nx + v_l*ny
    # Right states
    bc_r = out_bc(dim, r, mode)
    rho_r = bc_r[0]
    u_r = bc_r[1]
    v_r = bc_r[2]
    p_r = bc_r[3]
    h0_r = (gamma/(gamma - 1))*(p_r/rho_r) + 0.5 * (u_r**2 + v_r**2)
    a_r = np.sqrt(gamma*(p_r/rho_r))
    u_nr = u_r*nx + v_r*ny

    a_avg = 0.5*(a_l + a_r)
    xma_l = u_nl/a_avg
    xma_r = u_nr/a_avg
    
    al_l = 0.5*(1.0 + np.sign((xma_l)))
    al_r = 0.5*(1.0 - np.sign((xma_r)))

    be_l = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_l)))
    be_r = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_r)))

    # Defining the van Leer polynomials
    d_pl = 0.25*(((xma_l+1)**2)*(2-xma_l))
    d_mi = 0.25*(((xma_r-1)**2)*(2+xma_r))

    d_l = al_l*(1+be_l) - be_l*d_pl
    d_r = al_r*(1+be_r) - be_r*d_mi

    cvl_pl = al_l * (1+be_l) * xma_l - be_l * 0.25 * (1+xma_l)**2
    cvl_mi = al_r * (1+be_r) * xma_r + be_r * 0.25 * (1-xma_r)**2

    # Using the van Leer polynoials to define the Liou-Stefan polynomials
    cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
    cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

    # Defining the mass-flux using the LS formulation
    fm_pl = area * a_avg * cls_pl * rho_l
    fm_mi = area * a_avg * cls_mi * rho_r
    xFluxes[dim[0]-1,:, 0] = fm_pl + fm_mi
    xFluxes[dim[0]-1,:, 1] = fm_pl * u_l + fm_mi * u_r + (d_l*p_l + d_r*p_r) * nx*area
    xFluxes[dim[0]-1,:, 2] = fm_pl * v_l + fm_mi * v_r + (d_l*p_l + d_r*p_r) * ny*area
    xFluxes[dim[0]-1,:, 3] = fm_pl * h0_l + fm_mi * h0_r

    # Y-face fluxes
    for j in range(1, dim[1]-1):
        area = yfacesA[:,j]
        nx = yfacesN[:,j,0]/area
        ny = yfacesN[:,j,1]/area
        # Lower states
        rho_d = ru[0, :, j-1]
        u_d = ru[1, :, j-1]
        v_d = ru[2, :, j-1]
        p_d = ru[3, :, j-1]
        h0_d = (gamma/(gamma - 1))*(p_d/rho_d) + 0.5 * (u_d**2 + v_d**2)
        a_d = np.sqrt(gamma*(p_d/rho_d))
        u_nd = u_d*nx + v_d*ny
        # Upper states
        rho_u = rd[0, :, j]
        u_u = rd[1, :, j]
        v_u = rd[2, :, j]
        p_u = rd[3, :, j]
        h0_u = (gamma/(gamma - 1))*(p_u/rho_u) + 0.5 * (u_u**2 + v_u**2)
        a_u = np.sqrt(gamma*(p_u/rho_u))
        u_nu = u_u*nx + v_u*ny

        a_avg = 0.5*(a_d + a_u)
        xma_d = u_nd/a_avg
        xma_u = u_nu/a_avg

        al_d = 0.5*(1.0 + np.sign((xma_d)))
        al_u = 0.5*(1.0 - np.sign((xma_u)))

        be_d = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_d)))
        be_u = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_u)))

        # Defining the van Leer polynomials
        d_pl = 0.25*(((xma_d+1)**2)*(2-xma_d))
        d_mi = 0.25*(((xma_u-1)**2)*(2+xma_u))

        d_d = al_d*(1+be_d) - be_d*d_pl
        d_u = al_u*(1+be_u) - be_u*d_mi

        cvl_pl = al_d * (1+be_d) * xma_d - be_d * 0.25 * (1+xma_d)**2
        cvl_mi = al_u * (1+be_u) * xma_u + be_u * 0.25 * (1-xma_u)**2

        # Using the van Leer polynoials to define the Liou-Stefan polynomials
        cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
        cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

        # Defining the mass-flux using the LS formulation
        fm_pl = area * a_avg * cls_pl * rho_d
        fm_mi = area * a_avg * cls_mi * rho_u
        yFluxes[:,j, 0] = fm_pl + fm_mi
        yFluxes[:,j, 1] = fm_pl * u_d + fm_mi * u_u + (d_d*p_d + d_u*p_u) * nx*area
        yFluxes[:,j, 2] = fm_pl * v_d + fm_mi * v_u + (d_d*p_d + d_u*p_u) * ny*area
        yFluxes[:,j, 3] = fm_pl * h0_d + fm_mi * h0_u

    bc_w = slipwall(dim,r,yfacesA,yfacesN)
    
    # Lower boundary flux
    area = yfacesA[:,0]
    nx = yfacesN[:,0,0]/area
    ny = yfacesN[:,0,1]/area
    # Upper states
    rho_u = rd[0, :, 0]
    u_u = rd[1, :, 0]
    v_u = rd[2, :, 0]
    p_u = rd[3, :, 0]
    h0_u = (gamma/(gamma - 1))*(p_u/rho_u) + 0.5 * (u_u**2 + v_u**2)
    a_u = np.sqrt(gamma*(p_u/rho_u))
    u_nu = u_u*nx + v_u*ny
    
    # Lower states
    u_nd = -u_nu
    a_avg = 0.5*(a_d + a_u)

    xma_d = u_nd/a_avg
    xma_u = u_nu/a_avg
    al_d = 0.5*(1.0 + np.sign((xma_d)))
    al_u = 0.5*(1.0 - np.sign((xma_u)))
    be_d = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_d)))
    be_u = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_u)))

    # Defining the van Leer polynomials
    d_pl = 0.25*(((xma_d+1)**2)*(2-xma_d))
    d_mi = 0.25*(((xma_u-1)**2)*(2+xma_u))

    d_d = al_d*(1+be_d) - be_d*d_pl
    d_u = al_u*(1+be_u) - be_u*d_mi

    cvl_pl = al_d * (1+be_d) * xma_d - be_d * 0.25 * (1+xma_d)**2
    cvl_mi = al_u * (1+be_u) * xma_u + be_u * 0.25 * (1-xma_u)**2

    # Using the van Leer polynoials to define the Liou-Stefan polynomials
    cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
    cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

    # Defining the mass-flux using the LS formulation
    fm_pl = 0 
    fm_mi = 0 
    yFluxes[:,0, 0] = fm_pl + fm_mi
    yFluxes[:,0, 1] = fm_pl * u_d + fm_mi * u_u + (d_d*p_d + d_u*p_u) * nx*area
    yFluxes[:,0, 2] = fm_pl * v_d + fm_mi * v_u + (d_d*p_d + d_u*p_u) * ny*area
    yFluxes[:,0, 3] = fm_pl * h0_d + fm_mi * h0_u
    
    # Upper boundary flux
    area = yfacesA[:,dim[1]-1]
    nx = yfacesN[:,dim[1]-1,0]/area
    ny = yfacesN[:,dim[1]-1,1]/area
    # Lower states
    rho_d = ru[0, :, dim[1]-2]
    u_d = ru[1, :, dim[1]-2]
    v_d = ru[2, :, dim[1]-2]
    p_d = ru[3, :, dim[1]-2]
    h0_d = (gamma/(gamma - 1))*(p_d/rho_d) + 0.5 * (u_d**2 + v_d**2)
    a_d = np.sqrt(gamma*(p_d/rho_d))
    u_nd = u_d*nx + v_d*ny
    # Upper states
    u_nu = -u_nd

    a_avg = 0.5*(a_d + a_u)
    xma_d = u_nd/a_avg
    xma_u = u_nu/a_avg

    al_d = 0.5*(1.0 + np.sign((xma_d)))
    al_u = 0.5*(1.0 - np.sign((xma_u)))

    be_d = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_d)))
    be_u = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_u)))

    # Defining the van Leer polynomials
    d_pl = 0.25*(((xma_d+1)**2)*(2-xma_d))
    d_mi = 0.25*(((xma_u-1)**2)*(2+xma_u))

    d_d = al_d*(1+be_d) - be_d*d_pl
    d_u = al_u*(1+be_u) - be_u*d_mi

    cvl_pl = al_d * (1+be_d) * xma_d - be_d * 0.25 * (1+xma_d)**2
    cvl_mi = al_u * (1+be_u) * xma_u + be_u * 0.25 * (1-xma_u)**2

    # Using the van Leer polynoials to define the Liou-Stefan polynomials
    cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
    cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

    # Defining the mass-flux using the LS formulation
    fm_pl = 0 
    fm_mi = 0 
    yFluxes[:,dim[1]-1, 0] = fm_pl + fm_mi
    yFluxes[:,dim[1]-1, 1] = fm_pl * u_d + fm_mi * u_u + (d_d*p_d + d_u*p_u) * nx*area
    yFluxes[:,dim[1]-1, 2] = fm_pl * v_d + fm_mi * v_u + (d_d*p_d + d_u*p_u) * ny*area
    yFluxes[:,dim[1]-1, 3] = fm_pl * h0_d + fm_mi * h0_u

    res = ((xFluxes[1:, :, :] - xFluxes[:-1, :, :]) + (yFluxes[:, 1:, :] - yFluxes[:, :-1, :]))

    return res

# AUSM fluxes (first order)
def fluxls(dim, r):
    
    # Fluxes initialization
    xFluxes = np.zeros((dim[0], dim[1]-1, 4))        # for SM: Same shape as xfacesA + a dimension with value 4 for the 4 eq's
    yFluxes = np.zeros((dim[0]-1, dim[1], 4))        # for SM: Same shape as yfacesA + a dimension with value 4 for the 4 eq's

    # X-face fluxes
    for i in range(1, dim[0]-1):
        area = xfacesA[i,:]
        nx = xfacesN[i,:,0]/area
        ny = xfacesN[i,:,1]/area
        # Left states
        rho_l = r[0, i-1, :]
        u_l = r[1, i-1, :]
        v_l = r[2, i-1, :]
        p_l = r[3, i-1, :]
        h0_l = (gamma/(gamma - 1))*(p_l/rho_l) + 0.5 * (u_l**2 + v_l**2)
        a_l = np.sqrt(gamma*(p_l/rho_l))
        u_nl = u_l*nx + v_l*ny
        # Right states
        rho_r = r[0, i, :]
        u_r = r[1, i, :]
        v_r = r[2, i, :]
        p_r = r[3, i, :]
        h0_r = (gamma/(gamma - 1))*(p_r/rho_r) + 0.5 * (u_r**2 + v_r**2)
        a_r = np.sqrt(gamma*(p_r/rho_r))
        u_nr = u_r*nx + v_r*ny

        a_avg = 0.5*(a_l + a_r)
        xma_l = u_nl/a_avg
        xma_r = u_nr/a_avg

        al_l = 0.5*(1.0 + np.sign((xma_l)))
        al_r = 0.5*(1.0 - np.sign((xma_r)))

        be_l = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_l)))
        be_r = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_r)))

        # Defining the van Leer polynomials
        d_pl = 0.25*(((xma_l+1)**2)*(2-xma_l))
        d_mi = 0.25*(((xma_r-1)**2)*(2+xma_r))

        d_l = al_l*(1+be_l) - be_l*d_pl
        d_r = al_r*(1+be_r) - be_r*d_mi

        cvl_pl = al_l * (1+be_l) * xma_l - be_l * 0.25 * (1+xma_l)**2
        cvl_mi = al_r * (1+be_r) * xma_r + be_r * 0.25 * (1-xma_r)**2

        # Using the van Leer polynoials to define the Liou-Stefan polynomials
        cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
        cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

        # Defining the mass-flux using the LS formulation
        fm_pl = area * a_avg * cls_pl * rho_l
        fm_mi = area * a_avg * cls_mi * rho_r
        xFluxes[i,:, 0] = fm_pl + fm_mi
        xFluxes[i,:, 1] = fm_pl * u_l + fm_mi * u_r + (d_l*p_l + d_r*p_r) * nx*area
        xFluxes[i,:, 2] = fm_pl * v_l + fm_mi * v_r + (d_l*p_l + d_r*p_r) * ny*area
        xFluxes[i,:, 3] = fm_pl * h0_l + fm_mi * h0_r

    # Left boundary flux
    area = xfacesA[0,:]
    nx = xfacesN[0,:,0]/area
    ny = xfacesN[0,:,1]/area
    # Left states
    bc_l = in_bc(dim, r, mode)
    rho_l = bc_l[0]
    u_l = bc_l[1]
    v_l = bc_l[2]
    p_l = bc_l[3]
    h0_l = (gamma/(gamma - 1))*(p_l/rho_l) + 0.5 * (u_l**2 + v_l**2)
    a_l = np.sqrt(gamma*(p_l/rho_l))
    u_nl = u_l*nx + v_l*ny
    # Right states
    rho_r = r[0, 0, :]
    u_r = r[1, 0, :]
    v_r = r[2, 0, :]
    p_r = r[3, 0, :]
    h0_r = (gamma/(gamma - 1))*(p_r/rho_r) + 0.5 * (u_r**2 + v_r**2)
    a_r = np.sqrt(gamma*(p_r/rho_r))
    u_nr = u_r*nx + v_r*ny

    a_avg = 0.5*(a_l + a_r)
    xma_l = u_nl/a_avg
    xma_r = u_nr/a_avg

    al_l = 0.5*(1.0 + np.sign((xma_l)))
    al_r = 0.5*(1.0 - np.sign((xma_r)))

    be_l = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_l)))
    be_r = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_r)))

    # Defining the van Leer polynomials
    d_pl = 0.25*(((xma_l+1)**2)*(2-xma_l))
    d_mi = 0.25*(((xma_r-1)**2)*(2+xma_r))

    d_l = al_l*(1+be_l) - be_l*d_pl
    d_r = al_r*(1+be_r) - be_r*d_mi

    cvl_pl = al_l * (1+be_l) * xma_l - be_l * 0.25 * (1+xma_l)**2
    cvl_mi = al_r * (1+be_r) * xma_r + be_r * 0.25 * (1-xma_r)**2

    # Using the van Leer polynoials to define the Liou-Stefan polynomials
    cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
    cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

    # Defining the mass-flux using the LS formulation
    fm_pl = area * a_avg * cls_pl * rho_l
    fm_mi = area * a_avg * cls_mi * rho_r
    xFluxes[0,:, 0] = fm_pl + fm_mi
    xFluxes[0,:, 1] = fm_pl * u_l + fm_mi * u_r + (d_l*p_l + d_r*p_r) * nx*area
    xFluxes[0,:, 2] = fm_pl * v_l + fm_mi * v_r + (d_l*p_l + d_r*p_r) * ny*area
    xFluxes[0,:, 3] = fm_pl * h0_l + fm_mi * h0_r
    
    # Right boundary flux
    area = xfacesA[dim[0]-1,:]
    nx = xfacesN[dim[0]-1,:,0]/area
    ny = xfacesN[dim[0]-1,:,1]/area
    # Left states
    rho_l = r[0, dim[0]-2, :]
    u_l = r[1, dim[0]-2, :]
    v_l = r[2, dim[0]-2, :]
    p_l = r[3, dim[0]-2, :]
    h0_l = (gamma/(gamma - 1))*(p_l/rho_l) + 0.5 * (u_l**2 + v_l**2)
    a_l = np.sqrt(gamma*(p_l/rho_l))
    u_nl = u_l*nx + v_l*ny
    # Right states
    bc_r = out_bc(dim, r, mode)
    rho_r = bc_r[0]
    u_r = bc_r[1]
    v_r = bc_r[2]
    p_r = bc_r[3]
    h0_r = (gamma/(gamma - 1))*(p_r/rho_r) + 0.5 * (u_r**2 + v_r**2)
    a_r = np.sqrt(gamma*(p_r/rho_r))
    u_nr = u_r*nx + v_r*ny

    a_avg = 0.5*(a_l + a_r)
    xma_l = u_nl/a_avg
    xma_r = u_nr/a_avg
    
    al_l = 0.5*(1.0 + np.sign((xma_l)))
    al_r = 0.5*(1.0 - np.sign((xma_r)))

    be_l = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_l)))
    be_r = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_r)))

    # Defining the van Leer polynomials
    d_pl = 0.25*(((xma_l+1)**2)*(2-xma_l))
    d_mi = 0.25*(((xma_r-1)**2)*(2+xma_r))

    d_l = al_l*(1+be_l) - be_l*d_pl
    d_r = al_r*(1+be_r) - be_r*d_mi

    cvl_pl = al_l * (1+be_l) * xma_l - be_l * 0.25 * (1+xma_l)**2
    cvl_mi = al_r * (1+be_r) * xma_r + be_r * 0.25 * (1-xma_r)**2

    # Using the van Leer polynoials to define the Liou-Stefan polynomials
    cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
    cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

    # Defining the mass-flux using the LS formulation
    fm_pl = area * a_avg * cls_pl * rho_l
    fm_mi = area * a_avg * cls_mi * rho_r
    xFluxes[dim[0]-1,:, 0] = fm_pl + fm_mi
    xFluxes[dim[0]-1,:, 1] = fm_pl * u_l + fm_mi * u_r + (d_l*p_l + d_r*p_r) * nx*area
    xFluxes[dim[0]-1,:, 2] = fm_pl * v_l + fm_mi * v_r + (d_l*p_l + d_r*p_r) * ny*area
    xFluxes[dim[0]-1,:, 3] = fm_pl * h0_l + fm_mi * h0_r

    # Y-face fluxes
    for j in range(1, dim[1]-1):
        area = yfacesA[:,j]
        nx = yfacesN[:,j,0]/area
        ny = yfacesN[:,j,1]/area
        # Lower states
        rho_d = r[0, :, j-1]
        u_d = r[1, :, j-1]
        v_d = r[2, :, j-1]
        p_d = r[3, :, j-1]
        h0_d = (gamma/(gamma - 1))*(p_d/rho_d) + 0.5 * (u_d**2 + v_d**2)
        a_d = np.sqrt(gamma*(p_d/rho_d))
        u_nd = u_d*nx + v_d*ny
        # Upper states
        rho_u = r[0, :, j]
        u_u = r[1, :, j]
        v_u = r[2, :, j]
        p_u = r[3, :, j]
        h0_u = (gamma/(gamma - 1))*(p_u/rho_u) + 0.5 * (u_u**2 + v_u**2)
        a_u = np.sqrt(gamma*(p_u/rho_u))
        u_nu = u_u*nx + v_u*ny

        a_avg = 0.5*(a_d + a_u)
        xma_d = u_nd/a_avg
        xma_u = u_nu/a_avg

        al_d = 0.5*(1.0 + np.sign((xma_d)))
        al_u = 0.5*(1.0 - np.sign((xma_u)))

        be_d = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_d)))
        be_u = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_u)))

        # Defining the van Leer polynomials
        d_pl = 0.25*(((xma_d+1)**2)*(2-xma_d))
        d_mi = 0.25*(((xma_u-1)**2)*(2+xma_u))

        d_d = al_d*(1+be_d) - be_d*d_pl
        d_u = al_u*(1+be_u) - be_u*d_mi

        cvl_pl = al_d * (1+be_d) * xma_d - be_d * 0.25 * (1+xma_d)**2
        cvl_mi = al_u * (1+be_u) * xma_u + be_u * 0.25 * (1-xma_u)**2

        # Using the van Leer polynoials to define the Liou-Stefan polynomials
        cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
        cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

        # Defining the mass-flux using the LS formulation
        fm_pl = area * a_avg * cls_pl * rho_d
        fm_mi = area * a_avg * cls_mi * rho_u
        yFluxes[:,j, 0] = fm_pl + fm_mi
        yFluxes[:,j, 1] = fm_pl * u_d + fm_mi * u_u + (d_d*p_d + d_u*p_u) * nx*area
        yFluxes[:,j, 2] = fm_pl * v_d + fm_mi * v_u + (d_d*p_d + d_u*p_u) * ny*area
        yFluxes[:,j, 3] = fm_pl * h0_d + fm_mi * h0_u

    bc_w = slipwall(dim,r,yfacesA,yfacesN)
    # Lower boundary flux
    area = yfacesA[:,0]
    nx = yfacesN[:,0,0]/area
    ny = yfacesN[:,0,1]/area
    # Upper states
    rho_u = r[0, :, 0]
    u_u = r[1, :, 0]
    v_u = r[2, :, 0]
    p_u = r[3, :, 0]
    h0_u = (gamma/(gamma - 1))*(p_u/rho_u) + 0.5 * (u_u**2 + v_u**2)
    a_u = np.sqrt(gamma*(p_u/rho_u))
    u_nu = u_u*nx + v_u*ny
    # Lower state
    rho_d = bc_w[0]
    p_d = bc_w[1]
    a_d = np.sqrt(gamma*(p_d/rho_d))
    u_nd = -u_nu
    
    a_avg = 0.5*(a_d + a_u)
    xma_d = u_nd/a_avg
    xma_u = u_nu/a_avg

    al_d = 0.5*(1.0 + np.sign((xma_d)))
    al_u = 0.5*(1.0 - np.sign((xma_u)))

    be_d = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_d)))
    be_u = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_u)))

    # Defining the van Leer polynomials
    d_pl = 0.25*(((xma_d+1)**2)*(2-xma_d))
    d_mi = 0.25*(((xma_u-1)**2)*(2+xma_u))

    d_d = al_d*(1+be_d) - be_d*d_pl
    d_u = al_u*(1+be_u) - be_u*d_mi

    cvl_pl = al_d * (1+be_d) * xma_d - be_d * 0.25 * (1+xma_d)**2
    cvl_mi = al_u * (1+be_u) * xma_u + be_u * 0.25 * (1-xma_u)**2

    # Using the van Leer polynoials to define the Liou-Stefan polynomials
    cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
    cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

    # Defining the mass-flux using the LS formulation
    fm_pl = 0 
    fm_mi = 0 
    yFluxes[:,0, 0] = fm_pl + fm_mi
    yFluxes[:,0, 1] = fm_pl * u_d + fm_mi * u_u + (d_d*p_d + d_u*p_u) * nx*area
    yFluxes[:,0, 2] = fm_pl * v_d + fm_mi * v_u + (d_d*p_d + d_u*p_u) * ny*area
    yFluxes[:,0, 3] = fm_pl * h0_d + fm_mi * h0_u
    
    # Upper boundary flux
    area = yfacesA[:,dim[1]-1]
    nx = yfacesN[:,dim[1]-1,0]/area
    ny = yfacesN[:,dim[1]-1,1]/area
    # Lower states
    rho_d = r[0, :, dim[1]-2]
    u_d = r[1, :, dim[1]-2]
    v_d = r[2, :, dim[1]-2]
    p_d = r[3, :, dim[1]-2]
    h0_d = (gamma/(gamma - 1))*(p_d/rho_d) + 0.5 * (u_d**2 + v_d**2)
    a_d = np.sqrt(gamma*(p_d/rho_d))
    u_nd = u_d*nx + v_d*ny
    # Upper states
    rho_u = bc_w[2]
    p_u = bc_w[3]
    a_u = np.sqrt(gamma*(p_u/rho_u))
    u_nu = -u_nd

    a_avg = 0.5*(a_d + a_u)
    xma_d = u_nd/a_avg
    xma_u = u_nu/a_avg

    al_d = 0.5*(1.0 + np.sign((xma_d)))
    al_u = 0.5*(1.0 - np.sign((xma_u)))

    be_d = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_d)))
    be_u = -np.maximum(0.0, 1.0-np.floor(np.abs(xma_u)))

    # Defining the van Leer polynomials
    d_pl = 0.25*(((xma_d+1)**2)*(2-xma_d))
    d_mi = 0.25*(((xma_u-1)**2)*(2+xma_u))

    d_d = al_d*(1+be_d) - be_d*d_pl
    d_u = al_u*(1+be_u) - be_u*d_mi

    cvl_pl = al_d * (1+be_d) * xma_d - be_d * 0.25 * (1+xma_d)**2
    cvl_mi = al_u * (1+be_u) * xma_u + be_u * 0.25 * (1-xma_u)**2

    # Using the van Leer polynoials to define the Liou-Stefan polynomials
    cls_pl = np.maximum(0.0, cvl_pl + cvl_mi)
    cls_mi = np.minimum(0.0, cvl_pl + cvl_mi)

    # Defining the mass-flux using the LS formulation
    fm_pl = 0
    fm_mi = 0
    yFluxes[:,dim[1]-1, 0] = fm_pl + fm_mi
    yFluxes[:,dim[1]-1, 1] = fm_pl * u_d + fm_mi * u_u + (d_d*p_d + d_u*p_u) * nx*area
    yFluxes[:,dim[1]-1, 2] = fm_pl * v_d + fm_mi * v_u + (d_d*p_d + d_u*p_u) * ny*area
    yFluxes[:,dim[1]-1, 3] = fm_pl * h0_d + fm_mi * h0_u

    res = ((xFluxes[1:, :, :] - xFluxes[:-1, :, :]) + (yFluxes[:, 1:, :] - yFluxes[:, :-1, :]))

    return res

# Roe flux (Flux-Difference Splitting)
def fluxroe(dim, r):

    # Fluxes initialization
    xFluxes = np.zeros((dim[0], dim[1]-1, 4))        # for SM: Same shape as xfacesA + a dimension with value 4 for the 4 eq's
    yFluxes = np.zeros((dim[0]-1, dim[1], 4))        # for SM: Same shape as yfacesA + a dimension with value 4 for the 4 eq's

    # X-face fluxes
    for i in range(1, dim[0]-1):
        area = xfacesA[i,:]
        nx = xfacesN[i,:,0]/area
        ny = xfacesN[i,:,1]/area
        # Left states
        rho_l = r[0, i-1, :]
        u_l = r[1, i-1, :]
        v_l = r[2, i-1, :]
        p_l = r[3, i-1, :]
        h0_l = (gamma/(gamma - 1))*(p_l/rho_l) + 0.5 * (u_l**2 + v_l**2)
        u_nl = u_l*nx + v_l*ny
        # Right states
        rho_r = r[0, i, :]
        u_r = r[1, i, :]
        v_r = r[2, i, :]
        p_r = r[3, i, :]
        h0_r = (gamma/(gamma - 1))*(p_r/rho_r) + 0.5 * (u_r**2 + v_r**2)
        u_nr = u_r*nx + v_r*ny

        # Roe states
        r_factor = np.sqrt(rho_r/rho_l)
        rho_roe = np.sqrt(rho_r*rho_l)
        u_roe = (u_l + u_r*r_factor)/(1+r_factor)
        v_roe = (v_l + v_r*r_factor)/(1+r_factor)
        vel_roe = np.sqrt(u_roe**2 + v_roe**2)
        h0_roe = (h0_l + h0_r*r_factor)/(1+r_factor)
        a_roe = np.sqrt((gamma-1.0)*(h0_roe-0.5*(vel_roe**2)))
        un_roe = u_roe*nx + v_roe*ny

        # Dissipation using Roe states
        alpha0 = area*np.abs(un_roe)*((rho_r-rho_l)-(p_r-p_l)/a_roe**2)
        alpha1 = (area/(2*a_roe**2))*np.abs(un_roe+a_roe)*((p_r-p_l)+(rho_roe*a_roe*(u_nr-u_nl)))
        alpha2 = (area/(2*a_roe**2))*np.abs(un_roe-a_roe)*((p_r-p_l)-(rho_roe*a_roe*(u_nr-u_nl)))
        alpha3 = alpha0 + alpha1 + alpha2
        alpha4 = a_roe*(alpha1-alpha2)
        alpha5 = area*np.abs(un_roe)*((rho_roe*(u_r-u_l))-(nx*rho_roe*(u_nr-u_nl)))
        alpha6 = area*np.abs(un_roe)*((rho_roe*(v_r-v_l))-(ny*rho_roe*(u_nr-u_nl)))

        diss0 = alpha3
        diss1 = u_roe*alpha3 + nx*alpha4 + alpha5
        diss2 = v_roe*alpha3 + ny*alpha4 + alpha6
        diss3 = h0_roe*(alpha3-alpha0) + un_roe*alpha4 + u_roe*alpha5 + v_roe*alpha6 + 0.5*(u_roe**2+v_roe**2)*alpha0

        fm_pl = 0.5*area*rho_l*u_nl
        fm_mi = 0.5*area*rho_r*u_nr

        xFluxes[i,:,0] = (fm_pl + fm_mi) - 0.5*diss0
        xFluxes[i,:,1] = (fm_pl*u_l + fm_mi*u_r) + 0.5*(p_l+p_r)*nx*area - 0.5*diss1
        xFluxes[i,:,2] = (fm_pl*v_l + fm_mi*v_r) + 0.5*(p_l+p_r)*ny*area - 0.5*diss2
        xFluxes[i,:,3] = (fm_pl*h0_l + fm_mi*h0_r) - 0.5*diss3

    # Left boundary flux
    area = xfacesA[0,:]
    nx = xfacesN[0,:,0]/area
    ny = xfacesN[0,:,1]/area
    # Left states
    bc_l = in_bc(dim, r, mode)
    rho_l = bc_l[0]
    u_l = bc_l[1]
    v_l = bc_l[2]
    p_l = bc_l[3]
    h0_l = (gamma/(gamma - 1))*(p_l/rho_l) + 0.5 * (u_l**2 + v_l**2)
    u_nl = u_l*nx + v_l*ny
    # Right states
    rho_r = r[0, 0, :]
    u_r = r[1, 0, :]
    v_r = r[2, 0, :]
    p_r = r[3, 0, :]
    h0_r = (gamma/(gamma - 1))*(p_r/rho_r) + 0.5 * (u_r**2 + v_r**2)
    u_nr = u_r*nx + v_r*ny

    # Roe states
    r_factor = np.sqrt(rho_r/rho_l)
    rho_roe = np.sqrt(rho_r*rho_l)
    u_roe = (u_l + u_r*r_factor)/(1+r_factor)
    v_roe = (v_l + v_r*r_factor)/(1+r_factor)
    vel_roe = np.sqrt(u_roe**2 + v_roe**2)
    h0_roe = (h0_l + h0_r*r_factor)/(1+r_factor)
    a_roe = np.sqrt((gamma-1.0)*(h0_roe-0.5*(vel_roe**2)))
    un_roe = u_roe*nx + v_roe*ny

    # Dissipation using Roe states
    alpha0 = area*np.abs(un_roe)*((rho_r-rho_l)-(p_r-p_l)/a_roe**2)
    alpha1 = (area/(2*a_roe**2))*np.abs(un_roe+a_roe)*((p_r-p_l)+(rho_roe*a_roe*(u_nr-u_nl)))
    alpha2 = (area/(2*a_roe**2))*np.abs(un_roe-a_roe)*((p_r-p_l)-(rho_roe*a_roe*(u_nr-u_nl)))
    alpha3 = alpha0 + alpha1 + alpha2
    alpha4 = a_roe*(alpha1-alpha2)
    alpha5 = area*np.abs(un_roe)*((rho_roe*(u_r-u_l))-(nx*rho_roe*(u_nr-u_nl)))
    alpha6 = area*np.abs(un_roe)*((rho_roe*(v_r-v_l))-(ny*rho_roe*(u_nr-u_nl)))

    diss0 = alpha3
    diss1 = u_roe*alpha3 + nx*alpha4 + alpha5
    diss2 = v_roe*alpha3 + ny*alpha4 + alpha6
    diss3 = h0_roe*(alpha3-alpha0) + un_roe*alpha4 + u_roe*alpha5 + v_roe*alpha6 + 0.5*(u_roe**2+v_roe**2)*alpha0

    fm_pl = 0.5*area*rho_l*u_nl
    fm_mi = 0.5*area*rho_r*u_nr

    xFluxes[0,:,0] = (fm_pl + fm_mi) - 0.5*diss0
    xFluxes[0,:,1] = (fm_pl*u_l + fm_mi*u_r) + 0.5*(p_l+p_r)*nx*area - 0.5*diss1
    xFluxes[0,:,2] = (fm_pl*v_l + fm_mi*v_r) + 0.5*(p_l+p_r)*ny*area - 0.5*diss2
    xFluxes[0,:,3] = (fm_pl*h0_l + fm_mi*h0_r) - 0.5*diss3

    # Right boundary flux
    area = xfacesA[dim[0]-1,:]
    nx = xfacesN[dim[0]-1,:,0]/area
    ny = xfacesN[dim[0]-1,:,1]/area
    # Left states
    rho_l = r[0, dim[0]-2, :]
    u_l = r[1, dim[0]-2, :]
    v_l = r[2, dim[0]-2, :]
    p_l = r[3, dim[0]-2, :]
    h0_l = (gamma/(gamma - 1))*(p_l/rho_l) + 0.5 * (u_l**2 + v_l**2)
    u_nl = u_l*nx + v_l*ny
    # Right states
    bc_r = out_bc(dim, r, mode)
    rho_r = bc_r[0]
    u_r = bc_r[1]
    v_r = bc_r[2]
    p_r = bc_r[3]
    h0_r = (gamma/(gamma - 1))*(p_r/rho_r) + 0.5 * (u_r**2 + v_r**2)
    u_nr = u_r*nx + v_r*ny

    # Roe states
    r_factor = np.sqrt(rho_r/rho_l)
    rho_roe = np.sqrt(rho_r*rho_l)
    u_roe = (u_l + u_r*r_factor)/(1+r_factor)
    v_roe = (v_l + v_r*r_factor)/(1+r_factor)
    vel_roe = np.sqrt(u_roe**2 + v_roe**2)
    h0_roe = (h0_l + h0_r*r_factor)/(1+r_factor)
    a_roe = np.sqrt((gamma-1.0)*(h0_roe-0.5*(vel_roe**2)))
    un_roe = u_roe*nx + v_roe*ny

    # Dissipation using Roe states
    alpha0 = area*np.abs(un_roe)*((rho_r-rho_l)-(p_r-p_l)/a_roe**2)
    alpha1 = (area/(2*a_roe**2))*np.abs(un_roe+a_roe)*((p_r-p_l)+(rho_roe*a_roe*(u_nr-u_nl)))
    alpha2 = (area/(2*a_roe**2))*np.abs(un_roe-a_roe)*((p_r-p_l)-(rho_roe*a_roe*(u_nr-u_nl)))
    alpha3 = alpha0 + alpha1 + alpha2
    alpha4 = a_roe*(alpha1-alpha2)
    alpha5 = area*np.abs(un_roe)*((rho_roe*(u_r-u_l))-(nx*rho_roe*(u_nr-u_nl)))
    alpha6 = area*np.abs(un_roe)*((rho_roe*(v_r-v_l))-(ny*rho_roe*(u_nr-u_nl)))

    diss0 = alpha3
    diss1 = u_roe*alpha3 + nx*alpha4 + alpha5
    diss2 = v_roe*alpha3 + ny*alpha4 + alpha6
    diss3 = h0_roe*(alpha3-alpha0) + un_roe*alpha4 + u_roe*alpha5 + v_roe*alpha6 + 0.5*(u_roe**2+v_roe**2)*alpha0
    
    fm_pl = 0.5*area*rho_l*u_nl
    fm_mi = 0.5*area*rho_r*u_nr

    xFluxes[dim[0]-1,:,0] = (fm_pl + fm_mi) - 0.5*diss0
    xFluxes[dim[0]-1,:,1] = (fm_pl*u_l + fm_mi*u_r) + 0.5*(p_l+p_r)*nx*area - 0.5*diss1
    xFluxes[dim[0]-1,:,2] = (fm_pl*v_l + fm_mi*v_r) + 0.5*(p_l+p_r)*ny*area - 0.5*diss2
    xFluxes[dim[0]-1,:,3] = (fm_pl*h0_l + fm_mi*h0_r) - 0.5*diss3

    # Y-face fluxes
    for j in range(1, dim[1]-1):
        area = yfacesA[:,j]
        nx = yfacesN[:,j,0]/area
        ny = yfacesN[:,j,1]/area
        # Lower states
        rho_d = r[0, :, j-1]
        u_d = r[1, :, j-1]
        v_d = r[2, :, j-1]
        p_d = r[3, :, j-1]
        h0_d = (gamma/(gamma - 1))*(p_d/rho_d) + 0.5 * (u_d**2 + v_d**2)
        u_nd = u_d*nx + v_d*ny
        # Upper states
        rho_u = r[0, :, j]
        u_u = r[1, :, j]
        v_u = r[2, :, j]
        p_u = r[3, :, j]
        h0_u = (gamma/(gamma - 1))*(p_u/rho_u) + 0.5 * (u_u**2 + v_u**2)
        u_nu = u_u*nx + v_u*ny

        # Roe states
        r_factor = np.sqrt(rho_u/rho_d)
        rho_roe = np.sqrt(rho_u*rho_d)
        u_roe = (u_d + u_u*r_factor)/(1+r_factor)
        v_roe = (v_d + v_u*r_factor)/(1+r_factor)
        vel_roe = np.sqrt(u_roe**2 + v_roe**2)
        h0_roe = (h0_d + h0_u*r_factor)/(1+r_factor)
        a_roe = np.sqrt((gamma-1.0)*(h0_roe-0.5*(vel_roe**2)))
        un_roe = u_roe*nx + v_roe*ny

        # Dissipation using Roe states

        alpha0 = area*np.abs(un_roe)*((rho_u-rho_d)-(p_u-p_d)/a_roe**2)
        alpha1 = (area/(2*a_roe**2))*np.abs(un_roe+a_roe)*((p_u-p_d)+(rho_roe*a_roe*(u_nu-u_nd)))
        alpha2 = (area/(2*a_roe**2))*np.abs(un_roe-a_roe)*((p_u-p_d)-(rho_roe*a_roe*(u_nu-u_nd)))
        alpha3 = alpha0 + alpha1 + alpha2
        alpha4 = a_roe*(alpha1-alpha2)
        alpha5 = area*np.abs(un_roe)*((rho_roe*(u_u-u_d))-(nx*rho_roe*(u_nu-u_nd)))
        alpha6 = area*np.abs(un_roe)*((rho_roe*(v_u-v_d))-(ny*rho_roe*(u_nu-u_nd)))

        diss0 = alpha3
        diss1 = u_roe*alpha3 + nx*alpha4 + alpha5
        diss2 = v_roe*alpha3 + ny*alpha4 + alpha6
        diss3 = h0_roe*(alpha3-alpha0) + un_roe*alpha4 + u_roe*alpha5 + v_roe*alpha6 + 0.5*(u_roe**2+v_roe**2)*alpha0

        fm_pl = 0.5*area*rho_d*u_nd
        fm_mi = 0.5*area*rho_u*u_nu

        yFluxes[:,j,0] = (fm_pl + fm_mi) - 0.5*diss0
        yFluxes[:,j,1] = (fm_pl*u_d + fm_mi*u_u) + 0.5*(p_d+p_u)*nx*area - 0.5*diss1
        yFluxes[:,j,2] = (fm_pl*v_d + fm_mi*v_u) + 0.5*(p_d+p_u)*ny*area - 0.5*diss2
        yFluxes[:,j,3] = (fm_pl*h0_d + fm_mi*h0_u) - 0.5*diss3
        
        bc_w = slipwall(dim, r)

    # Lower boundary flux
    area = yfacesA[:,0]
    nx = yfacesN[:,0,0]/area
    ny = yfacesN[:,0,1]/area
    # Lower states
    rho_d = bc_w[0]
    u_d = r[1, :, 0]
    v_d = -r[2, :, 0]
    p_d = bc_w[1]
    h0_d = (gamma/(gamma - 1))*(p_d/rho_d) + 0.5 * (u_d**2 + v_d**2)
    u_nd = u_d*nx + v_d*ny
    # Upper states
    rho_u = r[0, :, 0]
    u_u = r[1, :, 0]
    v_u = r[2, :, 0]
    p_u = r[3, :, 0]
    h0_u = (gamma/(gamma - 1))*(p_u/rho_u) + 0.5 * (u_u**2 + v_u**2)
    u_nu = u_u*nx + v_u*ny

    # Roe states not needed at wall

    fm_pl = 0 
    fm_mi = 0 

    yFluxes[:,0,0] = (fm_pl + fm_mi)
    yFluxes[:,0,1] = (fm_pl*u_d + fm_mi*u_u) + 0.5*(p_d+p_u)*nx*area
    yFluxes[:,0,2] = (fm_pl*v_d + fm_mi*v_u) + 0.5*(p_d+p_u)*ny*area
    yFluxes[:,0,3] = (fm_pl*h0_d + fm_mi*h0_u)

    # Upper boundary flux
    area = yfacesA[:,dim[1]-1]
    nx = yfacesN[:,dim[1]-1,0]/area
    ny = yfacesN[:,dim[1]-1,1]/area
    # Lower states
    rho_d = r[0, :, dim[1]-2]
    u_d = r[1, :, dim[1]-2]
    v_d = r[2, :, dim[1]-2]
    p_d = r[3, :, dim[1]-2]
    h0_d = (gamma/(gamma - 1))*(p_d/rho_d) + 0.5 * (u_d**2 + v_d**2)
    u_nd = u_d*nx + v_d*ny
    vel_d = np.sqrt(u_d**2 + v_d**2)
    # Upper states
    rho_u = bc_w[2]
    u_u = r[1, :, dim[1]-2]
    v_u = -r[2, :, dim[1]-2]
    p_u = bc_w[3]
    h0_u = (gamma/(gamma - 1))*(p_u/rho_u) + 0.5 * (u_u**2 + v_u**2)
    u_nu = u_u*nx + v_u*ny
    vel_u = np.sqrt(u_u**2 + v_u**2)

    # Roe states not needed at wall

    fm_pl = 0 
    fm_mi = 0 

    yFluxes[:,dim[1]-1,0] = (fm_pl + fm_mi)
    yFluxes[:,dim[1]-1,1] = (fm_pl*u_d + fm_mi*u_u) + 0.5*(p_d+p_u)*nx*area
    yFluxes[:,dim[1]-1,2] = (fm_pl*v_d + fm_mi*v_u) + 0.5*(p_d+p_u)*ny*area
    yFluxes[:,dim[1]-1,3] = (fm_pl*h0_d + fm_mi*h0_u)

    res = ((xFluxes[1:, :, :] - xFluxes[:-1, :, :]) + (yFluxes[:, 1:, :] - yFluxes[:, :-1, :]))

    return res

# Time-step calculation
def tcalc(dim, r, facenormals, faceareas, cellvols, cells):
    delta_t = np.zeros((dim[0]-1,dim[1]-1))
    for j in range(dim[1]-1):
        for i in range(dim[0]-1):
            lambda_a = np.zeros(4)
            vel_p = np.array([r[1, i, j], r[2, i, j]])
            p_k = r[3, i, j]
            rho_k = r[0, i, j]
            a_k = np.sqrt(gamma*p_k/rho_k)

            for k in range(4):
                area = faceareas[cells[i,j]][k]
                n = facenormals[cells[i,j]][k]
                vn = np.dot(vel_p,n)
                lambda_a[k] = np.abs(vn) + (a_k * area)


            t_by_v = cfl/np.sum(np.abs(lambda_a))
            delta_t[i,j] = t_by_v*cellvols[cells[i,j]]

    return delta_t
#--------- Function definitions end here -------#

#---------- Meshing ------------#
with open(file1, 'r') as meshfile:
    sortlines = []
    lines = meshfile.readlines()
    for line in lines:
        line = line.strip()
        strnum = line.split()
        fnum = [float(x) for x in strnum]
        sortlines.append(fnum)
dims = sortlines[0]
dim = [int(x) for x in dims]
dim = np.array(dim)

nodes = sortlines[1:-1]

cellsdef = []
for i in range(dim[0]*(dim[1]-1)):
    if (i + 1) % 97 == 0:
        continue
    cellsdef.append([i+1, i+1+dim[0], i+dim[0], i])

cells = np.zeros((dim[0]-1,dim[1]-1), dtype=int)
cellind = 0
for j in range(dim[1]-1):
    for i in range(dim[0]-1):
        cells[i,j] = int(cellind)
        cellind += 1

facenormals = []
for cell in cellsdef:
    nE = np.array(((nodes[cell[1]][1]-nodes[cell[0]][1]),-(nodes[cell[1]][0]-nodes[cell[0]][0])))
    nN = np.array(((nodes[cell[2]][1]-nodes[cell[1]][1]),-(nodes[cell[2]][0]-nodes[cell[1]][0])))
    nW = np.array(((nodes[cell[3]][1]-nodes[cell[2]][1]),-(nodes[cell[3]][0]-nodes[cell[2]][0])))
    nS = np.array(((nodes[cell[0]][1]-nodes[cell[3]][1]),-(nodes[cell[0]][0]-nodes[cell[3]][0])))
    facenormals.append([nE, nN, nW, nS])

faceareas = []
for cell in cellsdef:
    p0 = np.array(nodes[cell[0]])
    p1 = np.array(nodes[cell[1]])
    p2 = np.array(nodes[cell[2]])
    p3 = np.array(nodes[cell[3]])
    aE = np.linalg.norm(p1-p0)
    aN = np.linalg.norm(p2-p1)
    aW = np.linalg.norm(p3-p2)
    aS = np.linalg.norm(p0-p3)
    faceareas.append([aE, aN, aW, aS])
    

cellvols = []
for cell in cellsdef:
    p0 = np.array(nodes[cell[0]])
    p1 = np.array(nodes[cell[1]])
    p2 = np.array(nodes[cell[2]])
    p3 = np.array(nodes[cell[3]])
    a1 = 0.5 * np.abs(p0[0]*(p1[1] - p2[1]) + p1[0]*(p2[1] - p0[1]) + p2[0]*(p0[1] - p1[1]))
    a2 = 0.5 * np.abs(p0[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p0[1]) + p3[0]*(p0[1] - p2[1]))
    v = a1+a2
    cellvols.append(v)

xfacesN = np.zeros((dim[0],dim[1]-1, 2))
for j in range(dim[1]-1):
    for i in range(dim[0]):
        if i == 0:
            xfacesN[i,j] = -facenormals[cells[i,j]][2]
            continue
        xfacesN[i,j] = facenormals[cells[i-1,j]][0]

yfacesN = np.zeros((dim[0]-1,dim[1], 2))
for i in range(dim[0]-1):
    for j in range(dim[1]):
        if j == 0:
            yfacesN[i,j] = -facenormals[cells[i,j]][3]
            continue
        yfacesN[i,j] = facenormals[cells[i,j-1]][1]

xfacesA = np.zeros((dim[0],dim[1]-1))
for j in range(dim[1]-1):
    for i in range(dim[0]):
        if i == 0:
            xfacesA[i,j] = faceareas[cells[i,j]][2]
            continue
        xfacesA[i,j] = faceareas[cells[i-1,j]][0]

yfacesA = np.zeros((dim[0]-1,dim[1]))
for i in range(dim[0]-1):
    for j in range(dim[1]):
        if j == 0:
            yfacesA[i,j] = faceareas[cells[i,j]][3]
            continue
        yfacesA[i,j] = faceareas[cells[i,j-1]][1]

vols_2d = np.reshape(cellvols, (96,48), order='C')
#----------- Meshing ends here ------------#

#----------- Initialization ------------#
if not restart:
    r = np.ones((4,dim[0]-1,dim[1]-1))
    q = np.ones((4,dim[0]-1,dim[1]-1))

    r[0] = r[0] * rho_inf
    r[1] = r[1] * u_inf
    r[2] = r[2] * v_inf
    r[3] = r[3] * p_stat

    q[0] = q[0] * r[0]
    q[1] = q[1] * r[0] * r[1]
    q[2] = q[2] * r[0] * r[2]
    q[3] = q[3] * (r[3]/(gamma-1) + (0.5*r[0]*(r[2]**2+r[3]**2)))

else: 
    res_data = np.load('solfiles/solution_iter_145000.npz')
    r = res_data['r']
    q = res_data['q']
    res_arr = res_data['res']
    last_res = res_arr[-1]
#---------- Initialization ends here -----------#

#---------- Main Loop -----------#
if not restart:
    resnorm = []
else:
    resnorm = res_data['res'].tolist()

if not restart:
    itr = 0
else:
    itr = len(resnorm)

if not restart:
    c_res = 1
else:
    c_res = last_res

if not restart:
    time_itr = 0.0
else: 
    time_itr = last_time

if high_recon:
    rl,rr,rd,ru = muscl_reco(dim,r)

if high_recon:
    res = fluxls_muscl(dim, r, rl, rr, rd, ru)                  # Initialization of residual vector
else:
    res = fluxls(dim, r)

# Scales for convergence residuals
scale1 = rho_inf * u_inf
scale2 = rho_inf * u_inf * u_inf
scale3 = rho_inf * u_inf * u_inf
scale4 = rho_inf * u_inf * (((gamma*p_stat)/(p_stat*rho_inf))+(0.5*u_inf*u_inf))

while np.abs(c_res) > tol and itr <= 500000:

    delta_t = tcalc(dim, r, facenormals, faceareas, cellvols, cells)
    del_t = np.min(delta_t)
    q_new = np.zeros_like(q)
    
    # Updating the conservative variables
    for k in range(4):
        q_new[k] = q[k] - (del_t / vols_2d) * res[:,:,k]

    # Updating the primitive variables
    r[0] = q_new[0]
    r[1] = q_new[1]/q_new[0]
    r[2] = q_new[2]/q_new[0]
    r[3] = (gamma-1) * (q_new[3] - 0.5*((q_new[1]**2 + q_new[2]**2)/q_new[0]))
    
    if high_recon:
        rl,rr,rd,ru = muscl_reco(dim,r)                                         # Updating the face reconstructed states

    q = np.copy(q_new)
    if high_recon:
        res = fluxls_muscl(dim, r, rl, rr, rd, ru)                                  # Recalculating residuals
    else:
        res = fluxls(dim,r)

    resnorm.append(0.0)

    resnorm[itr] = np.sum((res[:,:,0]/scale1)**2+(res[:,:,1]/scale2)**2+(res[:,:,2]/scale3)**2+(res[:,:,3]/scale4)**2)

    if itr == 0:
        resnorm0 = resnorm[itr]
        c_res = 1
    else:
        resnorm0 = resnorm[0]
        stab = 1e-12 
        c_res = (resnorm[itr]) / (resnorm0 + stab)

    time_itr += del_t

    # if itr%50 == 0 or np.abs(c_res) <= tol:                    # Printing residuals every 50 iterations and after convergence
    #     print(f"Residual is {np.abs(c_res)} at time {time_itr}s corresponding to iteration {itr}.")        # SM: Only for debugging. Remove later.
    print(f"Residual is {np.abs(c_res)} at time {time_itr}s corresponding to iteration {itr}.")        # SM: Only for debugging. Remove later.
    itr += 1

    if itr%5000 == 0 or np.abs(c_res) <= tol:                    # Saving every 5000 iterations and after convergence
        resnorm_arr = np.array(resnorm)
        filename = f"solfiles/solution_iter_{itr}.npz"
        np.savez(filename, q=q, r=r, res=resnorm_arr)
        print(f"Saved solution to {filename}")
#---------- Main loop ends here -----------#

#---------- Post processing -----------#
# 1. Extrapolation to nodes
dim = np.array((97,49))

loaded_data = np.load('solfiles_sub_roe/solution_iter_148369.npz')
q_loaded = loaded_data['q']
r_loaded = loaded_data['r']

p_loaded = r_loaded[3]
u_loaded = r_loaded[1]
v_loaded = r_loaded[2]
rho_loaded = r_loaded[0]

# Ghost layer addition
p_final = np.zeros((dim[0]+1, dim[1]+1))
p_final[1:-1, 1:-1] = np.copy(p_loaded)
u_final = np.zeros((dim[0]+1, dim[1]+1))
u_final[1:-1, 1:-1] = np.copy(u_loaded)
v_final = np.zeros((dim[0]+1, dim[1]+1))
v_final[1:-1, 1:-1] = np.copy(v_loaded)
rho_final = np.zeros((dim[0]+1, dim[1]+1))
rho_final[1:-1, 1:-1] = np.copy(rho_loaded)


# Left
if mode == 'sub':
    u_final[0,:] = u_inf
    v_final[0,:] = v_inf
    p_final[0,:] = p_final[1,:]
    rho_final[0,:] = rho_inf
else:
    u_final[0,:] = u_inf
    v_final[0,:] = v_inf
    p_final[0,:] = p_stat
    rho_final[0,:] = rho_inf

# Right
if mode == 'sub':
    u_final[-1,:] = u_final[-2,:]
    v_final[-1,:] = v_final[-2,:]
    p_final[-1,:] = p_stat
    rho_final[-1,:] = rho_final[-2,:]
else:
    u_final[-1,:] = u_final[-2,:]
    v_final[-1,:] = v_final[-2,:]
    p_final[-1,:] = p_final[-2,:]
    rho_final[-1,:] = rho_final[-2,:]

# Bottom
p_final[:,0]  =  p_final[:,1]
rho_final[:,0]  =  rho_final[:,1]
yfacesA_d = np.zeros(dim[0]+1)
yfacesA_d[1:-1] = yfacesA[:,0]
yfacesA_d[0] = yfacesA[0,0]
yfacesA_d[-1] = yfacesA[-1,0]
yfacesN_d = np.zeros((dim[0]+1,2))
yfacesN_d[1:-1,:] = yfacesN[:,0,:]
yfacesN_d[0,:] = yfacesN[0,0,:]
yfacesN_d[-1,:] = yfacesN[-1,0,:]
nx = yfacesN_d[:,0]/yfacesA_d
ny = yfacesN_d[:,1]/yfacesA_d
u_final[:,0] = (u_final[:,1]*(ny**2-nx**2)-2*v_final[:,1]*nx*ny)/(nx**2+ny**2)
v_final[:,0] = (v_final[:,1]*(nx**2-ny**2)-2*u_final[:,1]*nx*ny)/(nx**2+ny**2)

# Top
p_final[:,-1] =  p_final[:,-2]
rho_final[:,-1] =  rho_final[:,-2]
u_final[:,-1] =  u_final[:,-2]
v_final[:,-1] = -v_final[:,-2]

a_final = np.sqrt(gamma*p_final/rho_final)
M_final = np.sqrt(u_final**2+v_final**2)/a_final

p_node = np.zeros((dim[0], dim[1]))
u_node = np.zeros_like(p_node)
v_node = np.zeros_like(p_node)
rho_node = np.zeros_like(p_node)
M_node = np.zeros_like(p_node)

for j in range(dim[1]):
    for i in range(dim[0]):
        p_node[i,j] = 0.25*(p_final[i,j] + p_final[i+1,j] + p_final[i+1,j+1] + p_final[i,j+1])
        u_node[i,j] = 0.25*(u_final[i,j] + u_final[i+1,j] + u_final[i+1,j+1] + u_final[i,j+1])
        v_node[i,j] = 0.25*(v_final[i,j] + v_final[i+1,j] + v_final[i+1,j+1] + v_final[i,j+1])
        rho_node[i,j] = 0.25*(rho_final[i,j] + rho_final[i+1,j] + rho_final[i+1,j+1] + rho_final[i,j+1])
        M_node[i,j] = 0.25*(M_final[i,j] + M_final[i+1,j] + M_final[i+1,j+1] + M_final[i,j+1])

# 2. Plotting
nodes = np.array(nodes)
nodes_arr = np.zeros((dim[0],dim[1],2))
n_itr = 0
for j in range(dim[1]):
    for i in range(dim[0]):
        nodes_arr[i,j] = nodes[n_itr]
        n_itr += 1
x_coords = nodes_arr[:,:,0]
y_coords = nodes_arr[:,:,1]

x_len = x_coords.max() - x_coords.min()
y_len = y_coords.max() - y_coords.min()
fig_width = 8
fig_height = fig_width * (y_len / x_len)

# contour plot
plt.figure(figsize=(fig_width, fig_height))

plt.contourf(x_coords, y_coords, p_node, levels=50, cmap='rainbow')
plt.colorbar(label='P')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(False)
plt.axis('equal') # Ensure a square grid
plt.show()

#-------- Post-processing ends here ---------#

#-------- Code ends here ---------#