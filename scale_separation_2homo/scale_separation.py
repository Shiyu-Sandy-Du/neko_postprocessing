import numpy as np
from mpi4py import MPI
import time
import gll_lib
import matplotlib.pyplot as plt

from pysemtools.io.ppymech.neksuite import pynekread, pynekwrite
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.interpolation.mesh_to_mesh import PMapper
from pysemtools.comm.router import Router

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
rt = Router(comm)
precision = np.float32

### Timing
if rank == 0:
    ts = time.time()

### case parameters setting
datapath = 'data_dardel/LES/'
fieldname = 'field0'
outputname = 'field_lp0'
# kz_low = 31
# kz_high = 170
kz_low = 9
kz_high = 23
# kz_low = 2
# kz_high = 5
kx_low = 1
kx_high = 160
npl = 1 # number of time steps
nelemy = 36 # number of elements in y direction
nelemz = 10 # number of elements in z direction
nelemx = 10 # number of elements in x direction
nelem = nelemx*nelemy*nelemz

### solver parameters setting
Leg_order = 7
GLL_order = Leg_order + 1 # GLL points inside an element

### establish a mapping from 1D field into 3D field
field1d_map = np.array(range(nelem))
field3d_map = np.reshape(field1d_map,[nelemz,nelemy,nelemx])

### set up interpolation weights according to polynomial order
[ksi, _] = gll_lib.gLLNodesAndWeights(GLL_order)
if int(2 / (ksi[1] - ksi[0])) == 2 / (ksi[1] - ksi[0]):
    ### in function space, the length of the domain 
    ### is the multiple of the minimal GLL spacing
    n_fine = int(2.0 / (ksi[1] - ksi[0])) + 1
else:
    n_fine = int(2.0 / (ksi[1] - ksi[0])) + 2
# n_fine = int(GLL_order * 1.5)
### label the index of the overlapping points
index_overlapx_fine = [n_fine*i for i in range(1,nelemx)]
index_overlapy_fine = [n_fine*i for i in range(1,nelemy)]
index_overlapz_fine = [n_fine*i for i in range(1,nelemz)]

#### Partition in z for interpolation
start = rank*(nelemz//size)
stop = (rank+1)*(nelemz//size)
step = nelemz//size

counts = np.array([(nelemz//size)*n_fine* \
                    nelemy*n_fine * \
                    nelemx*n_fine for i in range(size)], dtype=np.longlong)
displs = np.array([sum(counts[:i]) for i in range(size)], dtype=np.longlong)

#### Partition in y for FFT
chunk_sizes_y = np.array([(nelemy*n_fine) // size + \
                        (1 if i < (nelemy*n_fine) % size else 0) for i in range(size)], dtype=np.longlong)
displs_y = np.array([sum(chunk_sizes_y[:i]) * \
                     nelemx * n_fine * \
                     nelemz * n_fine for i in range(size)], dtype=np.longlong)
counts_y = chunk_sizes_y * nelemx * n_fine * nelemz * n_fine
displs_y_fft = np.array([sum(chunk_sizes_y[:i]) * \
                     nelemx * (n_fine-1) * \
                     nelemz * (n_fine-1) for i in range(size)], dtype=np.longlong)
counts_y_fft = chunk_sizes_y * nelemx * (n_fine-1) * nelemz * (n_fine-1)

### allocate memory
if rank == 0:
    ux_fine = np.empty((nelemz*n_fine, nelemy*n_fine, nelemx*n_fine), dtype=precision)
else:
    ux_fine = None

ux_hat2_avg_piece = np.zeros((nelemz*(n_fine - 1), \
                       chunk_sizes_y[rank], \
                       nelemx*(n_fine - 1)), dtype=np.complex64)

### Prepare the time step loop
count_snapshot = 0
msh = Mesh(comm, create_connectivity=False)
fld = FieldRegistry(comm)
eq_fld = FieldRegistry(comm)
## Read the mean flow
for ipl in range(npl):
    if rank == 0:
        tstart = time.time()
        print("Read field file:",ipl+1,"/",npl) 
    ux_fine_piece = np.empty((step*n_fine, nelemy*n_fine, nelemx*n_fine), dtype=precision)
        
    filename = datapath+fieldname+'.f'+str(ipl).zfill(5)
    pynekread(filename, comm, data_dtype=precision, msh=msh, fld=fld)
    ## Take the fluctuation
    fld.add_field(comm, field_name='u_fluc', field=fld.registry['u'], dtype=precision)

    if ipl == 0:
        # initialize the mapper into finer mesh according to the first field
        mapper_l2h = PMapper(n=msh.lx, n_new=n_fine, distribution=['EQ', 'EQ', 'EQ'])
        eq_msh = mapper_l2h.create_mapped_mesh(comm, msh=msh)

    ### interpolate fields separately
    mapped_fields = mapper_l2h.interpolate_from_field_list(comm, field_list=[fld.registry['u_fluc']])

    eq_fld.add_field(comm, field_name='u_fluc', field=mapped_fields[0], dtype = precision)

    ###########################################################################################
    ######## SHOULD BE TURN INTO INDEXING WITH FIELD IN pyNekTools
    ######## Transition from 3 dimensional collapsed into 1d is needed
    ######## should be easy

    for iz_elem in range(step):
        for iy_elem in range(nelemy):
            for ix_elem in range(nelemx):
                local_el_id = field3d_map[iz_elem+start,iy_elem,ix_elem] - eq_msh.offset_el
                ux_fine_piece[iz_elem * n_fine: (iz_elem + 1) * n_fine, \
                        iy_elem * n_fine: (iy_elem + 1) * n_fine, \
                        ix_elem * n_fine: (ix_elem + 1) * n_fine] \
                = eq_fld.registry['u_fluc'][local_el_id, :, :, :]

    # ### All reduce to form a global array
    ## MPI gather: collect chunks splitted along z-axis
    ux_fine, _ = rt.gather_in_root(data = ux_fine_piece.flatten(), root = 0, dtype = ux_fine_piece.dtype)
    if rank == 0:
        ux_fine = np.reshape(ux_fine, (nelemz*n_fine, nelemy*n_fine, nelemx*n_fine))
        ux_fine = np.transpose(ux_fine, (1,0,2)) # shape: y,z,x for scatter

    ## MPI scatter: split chunks along y-axis
    ux_fine_fft_piece = np.empty(chunk_sizes_y[rank]*nelemz*n_fine*nelemx*n_fine, dtype=precision)
    ux_fine_fft_piece = rt.scatter_from_root(data = ux_fine, sendcounts=counts_y, root = 0, dtype = precision)
    ux_fine_fft_piece = np.reshape(ux_fine_fft_piece, (chunk_sizes_y[rank], nelemz*n_fine,  nelemx*n_fine))
    ux_fine_fft_piece = np.transpose(ux_fine_fft_piece, (1,0,2))# shape: z,y,x revert back

    # perform FFT in parallel
    ux_fine_iyel_noovlp = np.delete(ux_fine_fft_piece, [index_overlapz_fine], axis=0)
    ux_fine_iyel_noovlp = np.delete(ux_fine_iyel_noovlp, [index_overlapx_fine], axis=2)

    # Perform FFT on x and z direction
    ux_hat_iy = np.fft.fft2(ux_fine_iyel_noovlp[:-1, :, :-1],axes=(0,2)) # 2d fft

    # # filtering to get the target wavenumbers
    tmp = np.zeros(np.shape(ux_hat_iy), dtype=complex)
    kz_max = np.shape(ux_hat_iy)[0]
    kx_max = np.shape(ux_hat_iy)[2]
    tmp[kz_low:kz_high+1,:,kx_low:kx_high+1] = \
        ux_hat_iy[kz_low:kz_high+1,:,kx_low:kx_high+1]
    tmp[kz_max-kz_high:kz_max-kz_low+1,:,kx_max-kx_high:kx_max-kx_low+1] = \
        ux_hat_iy[kz_max-kz_high:kz_max-kz_low+1,:,kx_max-kx_high:kx_max-kx_low+1]
    tmp[kz_low:kz_high+1,:,kx_max-kx_high:kx_max-kx_low+1] = \
        ux_hat_iy[kz_low:kz_high+1,:,kx_max-kx_high:kx_max-kx_low+1]
    tmp[kz_max-kz_high:kz_max-kz_low+1,:,kx_low:kx_high+1] = \
        ux_hat_iy[kz_max-kz_high:kz_max-kz_low+1,:,kx_low:kx_high+1]
    ux_hat_iy = tmp # shape: kz, y, kx

    # Map back to physical space
    ux_fine_iyel_noovlp = np.fft.ifft2(ux_hat_iy,axes=(0,2)).real

    # # Map to GLL mesh (finer)
    ux_fine_iyel_end = np.concatenate((ux_fine_iyel_noovlp,ux_fine_iyel_noovlp[0,:,:][None,:,:]), axis=0)
    ux_fine_iyel_end = np.concatenate((ux_fine_iyel_end,ux_fine_iyel_end[:,:,0][:,:,None]), axis=2)
    ux_fine_fft_piece = np.empty((nelemz*n_fine, chunk_sizes_y[rank], nelemx*n_fine), dtype=precision)
    countz = 0
    for iz_elem in range(nelemz):
        countx = 0
        for ix_elem in range(nelemx):
            ux_fine_fft_piece[iz_elem * n_fine: (iz_elem + 1) * n_fine, \
                    :, \
                    ix_elem * n_fine: (ix_elem + 1) * n_fine] \
            = ux_fine_iyel_end[iz_elem * n_fine - countz: (iz_elem + 1) * n_fine - countz, \
                                    :, \
                                    ix_elem * n_fine - countx: (ix_elem + 1) * n_fine - countx]
            countx += 1
        countz += 1
    
    ## MPI gather: collect chunks splitted along y-axis
    ux_fine_fft_piece = np.transpose(ux_fine_fft_piece, (1,0,2)) # shape y,z,x
    ux_fine, _ = rt.gather_in_root(data = ux_fine_fft_piece.flatten(), root = 0, dtype = precision)

    if rank == 0:
        ux_fine = np.reshape(ux_fine, (nelemy*n_fine, nelemz*n_fine, nelemx*n_fine)) # shape: y,z,x
        ux_fine = np.transpose(ux_fine, (1,0,2)) # shape: z,y,x for scatter
    
    ## MPI scatter: broadcase into chunks along z-axis
    ux_fine_piece = rt.scatter_from_root(data = ux_fine, sendcounts=counts, \
                                             root = 0, dtype = precision)
    ux_fine_piece = np.reshape(ux_fine_piece, (step*n_fine, nelemy*n_fine, nelemx*n_fine))

    for iz_elem in range(step):
        for iy_elem in range(nelemy):
            for ix_elem in range(nelemx):
                local_el_id = field3d_map[iz_elem+start,iy_elem,ix_elem] - eq_msh.offset_el
                eq_fld.registry['u_fluc'][local_el_id, :, :, :] = \
                ux_fine_piece[iz_elem * n_fine: (iz_elem + 1) * n_fine, \
                        iy_elem * n_fine: (iy_elem + 1) * n_fine, \
                        ix_elem * n_fine: (ix_elem + 1) * n_fine] \

    # Output the filtered fields
    pynekwrite(datapath+outputname+".f"+str(ipl).zfill(5), comm, msh=eq_msh, \
               fld=eq_fld, write_mesh=True, wdsz=4)

    count_snapshot += 1
    
    fld.clear()
    eq_fld.clear()

    tend = time.time()
    if rank == 0:
        print("step time", tend - tstart)