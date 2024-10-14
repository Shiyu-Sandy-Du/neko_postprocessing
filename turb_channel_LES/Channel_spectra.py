import numpy as np
from mpi4py import MPI
import time
import gll_lib

from pynektools.io.ppymech.neksuite import pynekread, pynekwrite
from pynektools.datatypes.msh import Mesh
from pynektools.datatypes.field import FieldRegistry
from pynektools.interpolation.mesh_to_mesh import PMapper
from pynektools.comm.router import Router

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
rt = Router(comm)
precision = np.float32

### Timing
if rank == 0:
    ts = time.time()

### case parameters setting
# datapath = '/scratch/project_465000921/shiyud/chan_550_min_domain/data/'
datapath = '/scratch/shiyud/nekoexamples/turb_channel/DNS_590/tmp/'
fieldname_mean = 'mean_field200_400_xz0'
fieldname = 'field0'
npl = 1 # number of time steps
nelemy = 36 # number of elements in y direction
nelemz = 162 # number of elements in z direction
nelemx = 216 # number of elements in x direction
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
fld_mean = FieldRegistry(comm)
eq_fld = FieldRegistry(comm)
## Read the mean flow
filename_mean = datapath+fieldname_mean+'.f00000'
pynekread(filename_mean, comm, data_dtype=precision, msh=msh, fld=fld_mean)
for ipl in range(npl):
    if rank == 0:
        tstart = time.time()
        print("Read field file:",ipl+1,"/",npl) 
    ux_fine_piece = np.empty((step*n_fine, nelemy*n_fine, nelemx*n_fine), dtype=precision)
        
    filename = datapath+fieldname+'.f'+str(ipl).zfill(5)
    pynekread(filename, comm, data_dtype=precision, msh=msh, fld=fld)
    ## Take the fluctuation
    fld.add_field(comm, field_name='u_fluc', field=fld.registry['u'] - fld_mean.registry['u'], dtype=precision)

    if ipl == 0:
        # initialize the mapper into finer mesh according to the first field
        mapper = PMapper(n=msh.lx, n_new=n_fine, distribution=['EQ', 'EQ', 'EQ'])
        eq_msh = mapper.create_mapped_mesh(comm, msh=msh)
    ### interpolate in one run
    # eq_fld = mapper.create_mapped_field(comm, fld=fld)

    ### interpolate fields separately
    mapped_fields = mapper.interpolate_from_field_list(comm, field_list=[fld.registry['u_fluc']])

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
    ux_fine, _ = rt.gather_in_root(data = ux_fine_piece.flatten(), root = 0, gather="gather", dtype = ux_fine_piece.dtype)

    if rank == 0:
        ux_fine = np.reshape(ux_fine, (nelemz*n_fine, nelemy*n_fine, nelemx*n_fine))
        ux_fine = np.transpose(ux_fine, (1,0,2))

    ## MPI scatter: split chunks along y-axis
    ux_fine_fft_piece = np.empty(chunk_sizes_y[rank]*nelemz*n_fine*nelemx*n_fine, dtype=precision)
    ux_fine_fft_piece = rt.scatter_from_root(data = ux_fine, sendcounts=counts_y, root = 0, dtype = precision)
    ux_fine_fft_piece = np.reshape(ux_fine_fft_piece, (chunk_sizes_y[rank], nelemz*n_fine,  nelemx*n_fine))

    ux_fine_fft_piece = np.transpose(ux_fine_fft_piece, (1,0,2))

    # perform FFT in parallel
    ux_fine_iyel_noovlp = np.delete(ux_fine_fft_piece, [index_overlapz_fine], axis=0)
    ux_fine_iyel_noovlp = np.delete(ux_fine_iyel_noovlp, [index_overlapx_fine], axis=2)

    # Perform FFT on x and z direction
    ux_hat_iy = np.fft.fft(ux_fine_iyel_noovlp[:-1, :, :-1],axis=0)
    ux_hat_iy = np.fft.fft(ux_hat_iy,axis=2)
    ux_hat_iy_abs = np.abs(ux_hat_iy).astype(precision)
    ux_hat2_iy = np.square(ux_hat_iy_abs)

    # Averaging over snapshots (currently only ensemble avg)
    ux_hat2_avg_piece = \
        (ux_hat2_avg_piece * count_snapshot \
        + ux_hat2_iy) / (count_snapshot + 1)

    count_snapshot += 1
    
    fld.clear()
    eq_fld.clear()
    if rank == 0:
        ux_fine = np.transpose(ux_fine, (1,0,2))

    tend = time.time()
    if rank == 0:
        print("step time", tend - tstart)
    
### All reduce to form a global array for the Fourier modes
ux_hat2_avg_piece = np.transpose(ux_hat2_avg_piece, (1,0,2))
if rank == 0:
    ux_hat2_avg = np.empty((nelemy*n_fine, \
                       nelemz*(n_fine - 1), \
                       nelemx*(n_fine - 1)), dtype=precision)
else:
    ux_hat2_avg = None

ux_hat2_avg, _ = rt.gather_in_root(data = ux_hat2_avg_piece.flatten(), root = 0, gather="gather", dtype = ux_hat2_avg_piece.dtype)
if rank == 0:
    ux_hat2_avg = np.reshape(ux_hat2_avg, (nelemy*n_fine, \
                       nelemz*(n_fine - 1), \
                       nelemx*(n_fine - 1)))
    ux_hat2_avg = np.transpose(ux_hat2_avg, (1,0,2))

### Save spectra file
### Timing
if rank == 0:
    np.save(datapath + 'test1.npy', ux_hat2_avg)
    te = time.time()
    print("ellapsed time:", te - ts,"s")
