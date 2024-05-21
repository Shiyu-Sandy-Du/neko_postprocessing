import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

###################################
Re = 10953.67
nu = 1/Re
### file path and name
path = "/scratch/shiyud/nekoexamples/turb_channel/LES/"

u_implicit_file = path + "implicit/u_implicit.csv"
u_ds_file = path + "ds/u_ds.csv"
u_sigma_file = path + "sigma/u_sigma.csv"
u_smag_file = path + "smag/u_smag.csv"
u_vreman_file = path + "vreman/u_vreman.csv"

u_grad_implicit_file = path + "implicit/u_grad_implicit.csv"
u_grad_ds_file = path + "ds/u_grad_ds.csv"
u_grad_sigma_file = path + "sigma/u_grad_sigma.csv"
u_grad_smag_file = path + "smag/u_grad_smag.csv"
u_grad_vreman_file = path + "vreman/u_grad_vreman.csv"
### cooridnates dimension
ly = 8
nelem = int(32/2)
####################################
### data loading
index_half = int(ly*nelem)
y= np.genfromtxt(u_implicit_file, delimiter=',')[:index_half,1]
y = y + 1

u_implicit = np.genfromtxt(u_implicit_file, delimiter=',')[:index_half,2]
u_ds = np.genfromtxt(u_ds_file, delimiter=',')[:index_half,2]
u_sigma = np.genfromtxt(u_sigma_file, delimiter=',')[:index_half,2]
u_smag = np.genfromtxt(u_smag_file, delimiter=',')[:index_half,2]
u_vreman = np.genfromtxt(u_vreman_file, delimiter=',')[:index_half,2]

u_grad_w_implicit = np.genfromtxt(u_grad_implicit_file, delimiter=',')[0,2]
u_grad_w_ds = np.genfromtxt(u_grad_ds_file, delimiter=',')[0,2]
u_grad_w_sigma = np.genfromtxt(u_grad_sigma_file, delimiter=',')[0,2]
u_grad_w_smag = np.genfromtxt(u_grad_smag_file, delimiter=',')[0,2]
u_grad_w_vreman = np.genfromtxt(u_grad_vreman_file, delimiter=',')[0,2]

######################################################################################################
### Wall unit
u_tau_wall_implicit = np.sqrt(nu*u_grad_w_implicit)
u_tau_wall_ds = np.sqrt(nu*u_grad_w_ds)
u_tau_wall_sigma = np.sqrt(nu*u_grad_w_sigma)
u_tau_wall_smag = np.sqrt(nu*u_grad_w_smag)
u_tau_wall_vreman = np.sqrt(nu*u_grad_w_vreman)
print("Re_tau")
print("{:<12} {:<10}".format("Implicit:",u_tau_wall_implicit*1/nu))
print("{:<12} {:<10}".format("DS:",u_tau_wall_ds*1/nu))
print("{:<12} {:<10}".format("Sigma:",u_tau_wall_sigma*1/nu))
print("{:<12} {:<10}".format("Smagorinsky:",u_tau_wall_smag*1/nu))
print("{:<12} {:<10}".format("Vreman:",u_tau_wall_vreman*1/nu))

delta_nu_implicit = nu/u_tau_wall_implicit
delta_nu_ds = nu/u_tau_wall_ds
delta_nu_sigma = nu/u_tau_wall_sigma
delta_nu_smag = nu/u_tau_wall_smag
delta_nu_vreman = nu/u_tau_wall_vreman

########################################################################################################
### Shape factor
import gll_lib
def displacement_thickness(u, y, nelem):
    ## u and y are based on GLL points
    delta_1 = 0.0
    nx = int(len(u)/nelem)
    delta_integrand = np.zeros(nelem)
    f = 1 - u/u[-1]
    ## for loop over elements to perform integration
    for i in range(nelem):
        delta_integrand[i] = gll_lib.integrate_1d_using_gll_n(f[i*nx:(i+1)*nx],nx) \
                            *abs(y[(i+1)*nx-1]-y[i*nx])/2.0
    delta_1 = np.sum(delta_integrand)
    return delta_1

def momentum_thickness(u, y, nelem):
    ## u and y are based on GLL points
    delta_2 = 0.0
    nx = int(len(u)/nelem)
    delta_integrand = np.zeros(nelem)
    f = u/u[-1]*(1 - u/u[-1])
    ## for loop over elements to perform integration
    for i in range(nelem):
        delta_integrand[i] = gll_lib.integrate_1d_using_gll_n(f[i*nx:(i+1)*nx],nx) \
                            *abs(y[(i+1)*nx-1]-y[i*nx])/2.0
    delta_2 = np.sum(delta_integrand)
    return delta_2

H_implicit = displacement_thickness(u_implicit, y, nelem)/momentum_thickness(u_implicit, y, nelem)
H_ds       = displacement_thickness(u_ds, y, nelem)/momentum_thickness(u_ds, y, nelem)
H_sigma    = displacement_thickness(u_sigma, y, nelem)/momentum_thickness(u_sigma, y, nelem)
H_smag     = displacement_thickness(u_smag, y, nelem)/momentum_thickness(u_smag, y, nelem)
H_vreman   = displacement_thickness(u_vreman, y, nelem)/momentum_thickness(u_vreman, y, nelem)

print("Shape factor:")
print("{:<12} {:<10}".format("Implicit",H_implicit))
print("{:<12} {:<10}".format("DS",H_ds))
print("{:<12} {:<10}".format("Sigma",H_sigma))
print("{:<12} {:<10}".format("Smagorinsky",H_smag))
print("{:<12} {:<10}".format("Vreman",H_vreman))
# ######################################################################################################
# # ### plot profile
# fig, ax = plt.subplots(figsize=(11,8))
# ax.plot(u_implicit,y,label="implicit LES")
# ax.plot(u_ds,y,alpha=0.7,label="DS")
# ax.plot(u_sigma,y,alpha=0.7,label="Sigma")
# ax.plot(u_smag,y,"--",alpha=0.7,label="Smagorinsky")
# ax.plot(u_vreman,y,"--",alpha=0.7,label="Vreman")
# ax.set_xlabel(r'$<U>$',fontsize=20)
# ax.set_ylabel(r'$y/\delta$',fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend(loc='upper left')

# fig = plt.gcf()
# DPI = fig.get_dpi()
# fig.set_size_inches(600 / float(DPI), 500 / float(DPI))  # size of the saved figure (500,500)dpi
# figDir = path  # directory to save the figure
# figName = 'u_profile' 
# plt.savefig(figDir + figName + '.png', bbox_inches='tight', dpi=200)  # save the image in pdf format
# plt.tight_layout()

# plt.show()

# ######################### in plus unit
# fig, ax = plt.subplots(figsize=(11,8))
# ax.plot(y/delta_nu_implicit,u_implicit/u_tau_wall_implicit,label="implicit LES")
# ax.plot(y/delta_nu_ds,u_ds/u_tau_wall_ds,alpha=0.7,label="DS")
# ax.plot(y/delta_nu_sigma,u_sigma/u_tau_wall_sigma,alpha=0.7,label="Sigma")
# ax.plot(y/delta_nu_smag,u_smag/u_tau_wall_smag,"--",alpha=0.7,label="Smagorinsky")
# ax.plot(y/delta_nu_vreman,u_vreman/u_tau_wall_vreman,"--",alpha=0.7,label="Vreman")
# ax.set_ylabel(r'$<U>+$',fontsize=20)
# ax.set_xlabel(r'$y+$',fontsize=20)
# plt.xscale('log')
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend(loc='upper left')

# fig = plt.gcf()
# DPI = fig.get_dpi()
# fig.set_size_inches(600 / float(DPI), 500 / float(DPI))  # size of the saved figure (500,500)dpi
# figDir = path  # directory to save the figure
# figName = 'u_plus_profile' 
# plt.savefig(figDir + figName + '.png', bbox_inches='tight', dpi=200)  # save the image in pdf format
# plt.tight_layout()

# plt.show()