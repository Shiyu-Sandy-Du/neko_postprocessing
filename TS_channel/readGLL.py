import numpy as np
from pymech.neksuite import readnek
from mpi4py import MPI
import time
import gll_lib

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

### case parameters setting
datapath = '/scratch/shiyud/nekoexamples/TS_channel/'
fieldname = 'field0'
npl = 1 # number of time steps
nelemy = 4 # number of elements in y direction
nelemz = nelemy # number of elements in z direction
nelemx = nelemy # number of elements in x direction
nelem = nelemx*nelemy*nelemz

### solver parameters setting
Leg_order = 7
GLL_order = Leg_order + 1 # GLL points inside an element

### establish a mapping from 1D field into 3D field
field1d_map = np.array(range(nelem))
field3d_map = np.reshape(field1d_map,[nelemz,nelemy,nelemx])

### MPI partition
start = int(rank*nelemy/size)
stop = int((rank+1)*nelemy/size)
step = int(nelemy/size)

### set up interpolation weights according to polynomial order
[ksi, _] = gll_lib.gLLNodesAndWeights(GLL_order)
if int(2 / (ksi[1] - ksi[0])) == 2 / (ksi[1] - ksi[0]):
    ### in function space, the length of the domain 
    ### is the multiple of the minimal GLL spacing
    n_fine = int(2.0 / (ksi[1] - ksi[0])) + 1
else:
    n_fine = int(2.0 / (ksi[1] - ksi[0])) + 2

wt = np.empty((n_fine, GLL_order))
ksi_fine = np.linspace(-1, 1, n_fine)
for i in range(n_fine):
    wt[i, :] = gll_lib.interp_1D_weights(ksi, Leg_order, ksi_fine[i])

### allocate memory
u_fine_el = np.empty((3, n_fine, n_fine, n_fine))
ux_fine = np.empty((nelem, n_fine, n_fine, n_fine))
uy_fine = np.empty((nelem, n_fine, n_fine, n_fine))
uz_fine = np.empty((nelem, n_fine, n_fine, n_fine))
for ipl in range(npl):
    if rank == 0:
        print("Read field file:",ipl+1,"/",npl)   
    filename = datapath+fieldname+'.f'+str(ipl).zfill(5)
    dsi = readnek(filename)
    element = dsi.elem
    # perform interpolation elementwisely
    # this is a universal solution for all flows
    # however, interpolation in certain direction might not needed for
    # some flow cases such as boundary layer or channel flows 
    for iy_elem in range(start,stop):
        for iz_elem in range(nelemz):
            for ix_elem in range(nelemx):
                u_fine_el = np.tensordot(np.tensordot(np.tensordot( \
                            element[field3d_map[iz_elem,iy_elem,ix_elem]] \
                            .vel, wt, axes = (3, 1)), \
                                  wt, axes = (2, 1)), \
                                  wt, axes = (1, 1))
                ux_fine[field3d_map[iz_elem,iy_elem,ix_elem], :, :, :] \
                    = u_fine_el[0, :, :, :]
                uy_fine[field3d_map[iz_elem,iy_elem,ix_elem], :, :, :] \
                    = u_fine_el[1, :, :, :]
                uz_fine[field3d_map[iz_elem,iy_elem,ix_elem], :, :, :] \
                    = u_fine_el[2, :, :, :]

### All reduce to form a global array
comm.Allreduce(MPI.IN_PLACE, ux_fine, op=MPI.SUM)

                    