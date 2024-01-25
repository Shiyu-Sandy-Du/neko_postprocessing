import numpy as np
from pymech.neksuite import readnek
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

### case parameters setting
datapath = ''
fieldname = 'fieldxz0'
npl = 2 # number of time steps
nelemy = 18 # number of elements in y direction
kz = 0 # intended kz to look at
nelemz = nelemy # number of elements in z direction
nelemx = nelemy # number of elements in x direction

### solver parameters setting
GLL_order = 8 # GLL points inside an element
Leg_order = GLL_order-1
nkx = nelemx*GLL_order-(nelemx-1)-1
index_overlapx_Fou = [GLL_order*i for i in range(1,nelemx)]
index_overlapz_Fou = [GLL_order*i for i in range(1,nelemz)]

### MPI partition
start = int(rank*nelemy/size)
stop = int((rank+1)*nelemy/size)
step = int(nelemy/size)

### read the first time step
filename = datapath + fieldname + '.f00000'
dsi = readnek(filename)
element = dsi.elem

# establish a mapping from 1D field into 3D field
field1d_map = np.array(range(len(element)))
field3d_map = np.reshape(field1d_map,[nelemz,nelemy,nelemx])

if rank == 0:
    ## max amp of u1 over y at a certain kz
    growth = np.zeros([npl,nkx])

### following up
if rank == 0:
    ts = time.time()

for ipl in range(npl):
    if rank == 0:
        print("perform fft on time step:",ipl+1,"/",npl)
    if ipl != 0:
        if ipl < 10:
            filename = datapath+fieldname+'.f0000'+str(ipl)
        elif ipl < 100:
            filename = datapath+fieldname+'.f000'+str(ipl)
        elif ipl < 1000:
            filename = datapath+fieldname+'.f00'+str(ipl)
        elif ipl < 10000:
            filename = datapath+fieldname+'.f0'+str(ipl)
        else:
            filename = datapath+fieldname+'.f'+str(ipl)
        dsi = readnek(filename)
        element = dsi.elem
    cand_proc_curr = np.zeros(nkx)
    for iy_elem in range(start,stop):
        for iy in range(GLL_order):
            u1_Fou_iy = np.zeros([nelemz*GLL_order,nelemx*GLL_order])
            for iz_elem in range(nelemz):
                for ix_elem in range(nelemx):
                    u1_Fou_iy[iz_elem*GLL_order:(iz_elem+1)*GLL_order,\
                              ix_elem*GLL_order:(ix_elem+1)*GLL_order] =\
                        element[field3d_map[iz_elem,iy_elem,ix_elem]].vel[0,:,iy,:]
            ### perform fft on u1_Fou_iy
            u1_Fou_iy_noovlp = np.delete(u1_Fou_iy,[index_overlapz_Fou],axis=0)
            u1_Fou_iy_noovlp = np.delete(u1_Fou_iy_noovlp,[index_overlapx_Fou],axis=1)
            ## demean
            u1_avg_z = np.mean(u1_Fou_iy_noovlp[:-1,:-1],axis=0)
            u1_avg_zx = np.mean(u1_avg_z)
            u1_Fou_iy_noovlp = u1_Fou_iy_noovlp - u1_avg_zx

            u1_hat_iy = np.fft.fft(u1_Fou_iy_noovlp[:-1,:-1],axis=0)
            u1_hat_iy = np.fft.fft(u1_hat_iy,axis=1)
            u1_hat_iy /= np.shape(u1_Fou_iy_noovlp)[0]-1
            u1_hat_iy /= np.shape(u1_Fou_iy_noovlp)[1]-1
            for i_kx in range(nkx):
                if i_kx != 0 and (2*abs(u1_hat_iy[kz,i_kx])) > cand_proc_curr[i_kx]: 
                    cand_proc_curr[i_kx] = 2*abs(u1_hat_iy[kz,i_kx]) # kx != 0, consider complex conjugate
                elif i_kx == 0 and abs(u1_hat_iy[kz,i_kx]) > cand_proc_curr[i_kx]:
                    cand_proc_curr[i_kx] = abs(u1_hat_iy[kz,i_kx])

    if rank != 0:
        comm.Send(cand_proc_curr,dest=0,tag=14)
    else:
        cand = np.zeros([size,nkx])
        cand[0,:] = cand_proc_curr
        for ipro in range(1,size):
            receive = np.empty(nkx)
            comm.Recv(receive,source=ipro,tag=14)
            cand[ipro,:] = receive
        growth[ipl,:] = np.max(cand,axis=0)

if rank == 0:
    te = time.time()
    print('total time:', te-ts)
    # dimension np.shape(growth)[0] gives the time step
    # dimension np.shape(growth)[1] gives the kx
    np.save('test_output.npy', growth)