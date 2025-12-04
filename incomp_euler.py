import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

file1 = 'meshfile.txt'

rho = 1.293
alpha = 1
beta = 20
u_inf = 20
v_inf = 0
p_stat = 0                          # Since guage pressure is 0
cfl = 0.8
tol = 1e-8
restart = False
if restart:
    last_itr = 100000
    last_res = 0.004307725768801071

resfile = 'solfiles/solution_iter_100000.npz'       # Restart file

#------ Function definitions -----#
# Boundary condition
def applyBC(p, u, v, u_inf, v_inf, p_stat):
    # Left
    u[0,:] = u_inf
    v[0,:] = v_inf
    p[0,:] = 0

    # Right
    u[-1,:] = u[-2,:]
    v[-1,:] = v[-2,:]
    p[-1,:] = p[-2,:]

    # Bottom
    p[:,0] = p[:,1]
    u[:, 0] =  u[:, 1]
    v[:, 0] = -v[:, 1]

    # Top
    p[:, -1] =  p[:, -2]
    u[:, -1] =  u[:, -2]
    v[:, -1] = -v[:, -2]

    return p, u, v

# Expressing the fluxes as functions of cell values
def fluxcalc(dim, p, u, v):

    # Fluxes initialization
    xFluxes = np.zeros((dim[0], dim[1]-1, 3))        # for SM: Same shape as xfacesA + a dimension with value 3 for the 3 eq's
    yFluxes = np.zeros((dim[0]-1, dim[1], 3))        # for SM: Same shape as yfacesA + a dimension with value 3 for the 3 eq's
    
    # X-face fluxes
    for j in range(dim[1]-1):
        for i in range(1, dim[0]-1):
            area = xfacesA[i,j]
            nx = xfacesN[i,j][0]/area 
            ny = xfacesN[i,j][1]/area
            n = np.array([nx, ny])
            qL = np.array([1, u[i-1,j], v[i-1,j]])
            qR = np.array([1, u[i,j], v[i,j]])
            delP = p[i-1,j] - p[i,j]
            p_avg = (p[i-1,j] + p[i,j])/2
            u_avg = 0.5*(u[i-1,j] + u[i,j])
            v_avg = 0.5*(v[i-1,j] + v[i,j])
            v_vec = np.array([u_avg, v_avg])
            v_n = np.dot(v_vec, n)
            lambda_max = 0.5*(np.abs(v_n) + np.sqrt(v_n**2 + 4*beta**2))
            vL = np.array([u[i-1,j], v[i-1,j]])
            vR = np.array([u[i,j], v[i,j]])
            vnL = np.dot(vL, n)
            vnR = np.dot(vR,n)

            # conv_flux = rho * (max(0, vnL) * qL + min(0, vnR) * qR)
            press_dissipation = (alpha / (2 * lambda_max)) * delP
            c_flux = (rho * (max(0, vnL)) + press_dissipation) * qL + (rho*(min(0, vnR) + press_dissipation) * qR)
            p_flux = p_avg * np.array([0, nx, ny])
            
            xFluxes[i,j] = (c_flux + p_flux) * area
    
        # Left face
        area = xfacesA[0,j]
        nx = xfacesN[0,j][0]/area
        ny = xfacesN[0,j][1]/area
        n = np.array([nx, ny])
        qL = np.array([1, u_inf, v_inf])
        qR = np.array([1, u[0,j], v[0,j]])
        delP = 0         # Inlet guage pressure is 0
        p_avg = p[0,j]
        u_avg = 0.5*(u_inf + u[0,j])
        v_avg = 0.5*(v_inf + v[0,j])
        v_vec = np.array([u_avg, v_avg])
        v_n = np.dot(v_vec, n)
        lambda_max = 0.5*(np.abs(v_n) + np.sqrt(v_n**2 + 4*beta**2))
        vL = np.array([u_inf, v_inf])
        vR = np.array([u[0,j], v[0,j]])
        vnL = np.dot(vL, n)
        vnR = np.dot(vR,n)

        press_dissipation = (alpha / (2 * lambda_max)) * delP
        #c_flux = (rho * vnL) * qL 
        c_flux = (rho * (max(0, vnL)) + press_dissipation) * qL + (rho*(min(0, vnR) + press_dissipation) * qR)
        
        p_flux = p_avg * np.array([0, nx, ny])
        xFluxes[0,j] = (c_flux + p_flux) * area
    
        # Right face
        area = xfacesA[dim[0]-1,j]
        nx = xfacesN[dim[0]-1,j][0]/area
        ny = xfacesN[dim[0]-1,j][1]/area
        # nx = facenormals[cells[dim[0]-2,j]][0][0]
        # ny = facenormals[cells[dim[0]-2,j]][0][1]
        # area = faceareas[cells[dim[0]-2,j]][0]
        n = np.array([nx, ny])
        qL = np.array([1, u[dim[0]-2,j], v[dim[0]-2,j]])
        qR = np.array([1, u[dim[0]-2,j], v[dim[0]-2,j]])
        delP = 0
        #delP = p[dim[0]-2,j] - p_stat                       # Check physical interpretation
        p_avg = p[dim[0]-2,j]
        #p_avg = 0                                            # Imposing the right-side condition
        u_avg = u[dim[0]-2,j]
        v_avg = v[dim[0]-2,j]
        v_vec = np.array([u_avg, v_avg])
        v_n = np.dot(v_vec, n)
        lambda_max = 0.5*(np.abs(v_n) + np.sqrt(v_n**2 + 4*beta**2))
        vL = np.array([u[dim[0]-2,j], v[dim[0]-2,j]])
        vR = np.array([u[dim[0]-2,j], v[dim[0]-2,j]])
        vnL = np.dot(vL, n)
        vnR = np.dot(vR,n)

        press_dissipation = (alpha / (2 * lambda_max)) * delP
        #c_flux = (rho * vnL) * qL 
        c_flux = (rho * (max(0, vnL)) + press_dissipation) * qL + (rho*(min(0, vnR) + press_dissipation) * qR)
        
        p_flux = p_avg * np.array([0, nx, ny])
        xFluxes[dim[0]-1,j] = (c_flux + p_flux) * area        
        
    # Y-face fluxes
    for i in range(dim[0]-1):
        for j in range(1, dim[1]-1):
            area = yfacesA[i,j]
            nx = yfacesN[i,j][0]/area
            ny = yfacesN[i,j][1]/area
            n = np.array([nx, ny])
            qD = np.array([1, u[i,j-1], v[i,j-1]])
            qU = np.array([1, u[i,j], v[i,j]])
            delP = p[i,j-1] - p[i,j]
            p_avg = (p[i,j-1] + p[i,j])/2
            u_avg = 0.5*(u[i,j-1] + u[i,j])
            v_avg = 0.5*(v[i,j-1] + v[i,j])
            v_vec = np.array([u_avg, v_avg])
            v_n = np.dot(v_vec, n)
            lambda_max = 0.5*(np.abs(v_n) + np.sqrt(v_n**2 + 4*beta**2))
            vD = np.array([u[i,j-1], v[i,j-1]])
            vU = np.array([u[i,j], v[i,j]])
            vnD = np.dot(vD, n)
            vnU = np.dot(vU, n)
            
            press_dissipation = (alpha / (2 * lambda_max)) * delP
            c_flux = (rho * (max(0, vnD)) + press_dissipation) * qD + (rho*(min(0, vnU) + press_dissipation) * qU)
            
            p_flux = p_avg * np.array([0, nx, ny])
            yFluxes[i,j] = (c_flux + p_flux) * area
    
        # Bottom face
        area = yfacesA[i,0]
        nx = yfacesN[i,0][0]/area
        ny = yfacesN[i,0][1]/area
        n = np.array([nx, ny])
        p_face = p[i,0] 
        yFluxes[i,0] = p_face * np.array([0, nx, ny]) * area
    
        # Top face
        area = yfacesA[i,dim[1]-1]
        nx = yfacesN[i,dim[1]-1][0]/area
        ny = yfacesN[i,dim[1]-1][1]/area
        n = np.array([nx, ny])
        p_face = p[i,dim[1]-2]
        yFluxes[i,dim[1]-1] = p_face * np.array([0, nx, ny]) * area
    
    res = np.zeros((dim[0]-1,dim[1]-1, 3))
    for j in range(dim[1]-1):
        for i in range(dim[0]-1):
            res[i,j] = xFluxes[i+1, j] - xFluxes[i,j] + yFluxes[i,j+1] - yFluxes[i,j]

    return res

# Time-step calaculation
def tcalc(dim, u, v, facenormals, faceareas, cellvols, cells):
    t_step = np.zeros((dim[0]-1,dim[1]-1))
    for j in range(dim[1]-1):
        for i in range(dim[0]-1):
            uP = u[i,j]
            vP = v[i,j]
            velP = np.array([uP,vP])
            a = np.zeros(4)
            nW = np.zeros((4,2))
            vn = np.zeros(4)
            lambda_k = np.zeros(4)
            for k in range(4):
                a[k] = faceareas[cells[i,j]][k]
                nW[k] = np.array(facenormals[cells[i,j]][0])
                vn[k] = np.dot(velP, nW[k])
                lambda_k[k] = 0.5 * (np.abs(vn[k]) + np.sqrt(vn[k]**2 + 4*beta**2))
            t_step[i,j] = cellvols[cells[i,j]] * cfl / (np.dot(a, lambda_k))
    del_t = np.min(t_step)
    return np.abs(del_t)
#------- Functions end here -------#

#-------- Meshing --------#
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
#------- Meshing ends here -------#

#------- Initialization --------#
if not restart:
    p = np.zeros((dim[0]+1,dim[1]+1))
    u = np.zeros((dim[0]+1,dim[1]+1))
    v = np.zeros((dim[0]+1,dim[1]+1))
else: 
    res_data = np.load('solfiles/solution_iter_100000.npz')
    p = res_data['p']
    u = res_data['u']
    v = res_data['v']
#------- Initialization ends here --------#

#------- Main Loop --------#
if not restart:
    itr = 0
else:
    itr = last_itr + 2

if not restart:
    c_res = 1
else:
    c_res = last_res
time_itr = 0.0

res = fluxcalc(dim, p, u, v)        # Initialization of residual vector

# Scales for convergence residuals
scale1 = rho * u_inf
scale2 = rho * u_inf * u_inf
scale3 = rho * u_inf * u_inf

if not restart:
    resnorm = []
else:
    res = res_data['res']

while np.abs(c_res) > tol and itr <= 100000:

    del_t = tcalc(dim, u, v, facenormals, faceareas, cellvols, cells)
    p_new = np.zeros_like(p)
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)
    for j in range(dim[1]-1):
        for i in range(dim[0]-1):
            
            tt = del_t/cellvols[cells[i,j]]
            p_new[i,j] = p[i,j] - (beta**2) * (tt) * res[i,j,0]
            u_new[i,j] = u[i,j] - (tt) * ((res[i,j,1]/rho) - u[i,j]*res[i,j,0])
            v_new[i,j] = v[i,j] - (tt) * ((res[i,j,2]/rho) - v[i,j]*res[i,j,0])

    p = np.copy(p_new)
    u = np.copy(u_new)
    v = np.copy(v_new)

    res = fluxcalc(dim, p, u, v)                              # Recalculating residuals

    resnorm.append(0.0)

    for j in range(dim[1]-1):
        for i in range(dim[0]-1):
            resnorm[itr] += ((res[i,j,0]/scale1)**2+(res[i,j,1]/scale2)**2+(res[i,j,2]/scale3)**2)

    if itr == 0:
        resnorm0 = resnorm[itr]

    if itr == 0:
        resnorm0 = resnorm[itr]
        c_res = 1
    else:
        stab = 1e-12 
        c_res = (resnorm[itr]) / (resnorm0 + stab)
        
    time_itr += del_t
    print(f"Residual is {np.abs(c_res)} at time {time_itr}s corresponding to iteration {itr}.")        # SM: Only for debugging. Remove later.
    itr += 1

    if itr%1000 == 0 or np.abs(c_res) <= tol:                    # Saving every 1000 iterations and after convergence
        resnorm_arr = np.array(resnorm)
        filename = f"solfiles/solution_iter_{itr}.npz"
        np.savez(filename, p=p, u=u, v=v, res=resnorm_arr)
        print(f"Saved solution to {filename}")
#--------- Main loop ends here ----------#

#--------- Post-Process ---------#
# 1. Extrapolation to nodes from cell-centers
dim = np.array((97,49))

loaded_data = np.load('solfiles_NG/solution_iter_17651.npz')
p_loaded = loaded_data['p']
u_loaded = loaded_data['u']
v_loaded = loaded_data['v']

# Ghost layer addition
p_final = np.zeros((dim[0]+1, dim[1]+1))
p_final[1:-1, 1:-1] = np.copy(p_loaded)
u_final = np.zeros((dim[0]+1, dim[1]+1))
u_final[1:-1, 1:-1] = np.copy(u_loaded)
v_final = np.zeros((dim[0]+1, dim[1]+1))
v_final[1:-1, 1:-1] = np.copy(v_loaded)

# Left
u_final[0,:] = u_inf
v_final[0,:] = v_inf
p_final[0,:] = p_final[1,:]

# Right
u_final[-1,:] = u_final[-2,:]
v_final[-1,:] = v_final[-2,:]
p_final[-1,:] = p_final[-2,:]

# Bottom
p_final[:,0]  =  p_final[:,1]
u_final[:,0] =  u_final[:,1]
v_final[:,0] = -v_final[:,1]

# Top
p_final[:,-1] =  p_final[:,-2]
u_final[:,-1] =  u_final[:,-2]
v_final[:,-1] = -v_final[:,-2]

p_node = np.zeros((dim[0], dim[1]))
u_node = np.zeros_like(p_node)
v_node = np.zeros_like(p_node)

for j in range(dim[1]):
    for i in range(dim[0]):
        p_node[i,j] = 0.25*(p_final[i,j] + p_final[i+1,j] + p_final[i+1,j+1] + p_final[i,j+1])
        u_node[i,j] = 0.25*(u_final[i,j] + u_final[i+1,j] + u_final[i+1,j+1] + u_final[i,j+1])
        v_node[i,j] = 0.25*(v_final[i,j] + v_final[i+1,j] + v_final[i+1,j+1] + v_final[i,j+1])

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

plt.contourf(x_coords, y_coords, p_node, levels=50, cmap='seismic')
plt.colorbar(label='P')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(False)
plt.axis('equal') # Ensure a square grid
plt.show()
#---------- Post-processing ends here -----------#

#---------- Code ends here --------------#

    

