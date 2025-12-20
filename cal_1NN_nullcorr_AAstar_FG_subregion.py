import numpy as np
import pylab as plt
import os,sys
from itertools import cycle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import time
import argparse

### Present as a separate .py file in this current directory 
import crosscorrelation_utils as cc


'''
#####################################################################
###################    USER INPUT    ##############################
#####################################################################
'''



# Cosmological Parameters from GADGET-2

omega_b=0.04
omega_m=0.308     
h = 0.678
ngrid = 80                              # Number of grid points along each axis 
grid_cell_size = 2                       # Grid cell size in cMpc/h

fullbox_len = ngrid * grid_cell_size   # Total box length in cMpc/h
fullbox_len = np.float32(fullbox_len)  # Convert integer to float32

print("USER INPUT : fullbox_len (cMpc/h) = ",fullbox_len)


z = 7      # redshift
region = 0    # region number (1–8)
tobs=120 ## hours




parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_null",
    type=int,
    default=1000,
    help="Number of fake catalog realizations (default: 1000)",
)
args = parser.parse_args()
n_null = args.n_null


print("USER INPUT : Redshift z = ",z)
print("USER INPUT : n_null = ",n_null)
print("USER INPUT : Total 21cm obs. hrs. = ",tobs)

###### Parameters for kNN-CDF cross correlation function 
rmin=4
rmax=12
delta_r = 1
bin_len = int((rmax-rmin)/delta_r) + 1
n_data = 1              ### number of actual catalog realisation
kNN = [1]
threshold=75

nquery = 100  # number of points along each dimension of the query grid


bins_knn = np.zeros((bin_len, 1))
bins_knn[:,0] =  np.linspace(rmin,rmax,bin_len)
print("USER INPUT : radial_bins = ", bins_knn[:,0])
print("USER INPUT : nquery = ",nquery)
current_dir=os.getcwd()


#####################################################################

'''
#####################################################################
###################    DATA LOADING    ##############################
#####################################################################
'''

##################################################
# array of seeds (for creating random catalogs) 
##################################################

seedlist=np.genfromtxt('./musicseed_list_4001_entries_no_duplicates.txt',dtype=int) ### 4001 entries

# Select the first "n_null" seeds for calculating the null cross-correlations
seed_arr = seedlist[0:n_null]

print("length of seed_arr = ", len(seed_arr))
print("len of unique seeds)   =", len(set(seed_arr)))


###################################################
# galaxy catalog data
###################################################

reioncase="fiducial"
reioncaselabel = r"fiducial"

gal_data_dir = os.path.join(os.getcwd(), './reion_files_mixed_compare_cases/')
infile = os.path.join(gal_data_dir,f'galaxyCatalog_OIII_withfesc_{reioncase}_z{z:06.2f}.npz')

print("------------")
print(f"Processing CASE = {reioncase}, z = {z}")

# Load data if file exists
if os.path.exists(infile):
    galdata = np.load(infile)
    # Extract arrays
    log10_lum = galdata['log10_L_OIII_arr']       # Shape (N,)
    xpos = galdata['xpos_arr']                # Shape (N,)
    ypos = galdata['ypos_arr']                # Shape (N,)
    zpos = galdata['zpos_arr']                # Shape (N,)
else:
    print(f" Warning: file not found for z={z} and case '{case_name}'. Exiting.")
    sys.exit(1)


# Assemble into true galaxies positions (N,3)
positions = np.vstack((xpos, ypos, zpos)).T.astype(np.float32)

print(f"Loaded galaxy data for z={z}: positions shape {positions.shape}, luminosity shape {log10_lum.shape}")


####################################################
# Select galaxies using luminosity threshold
####################################################

# Luminosity cut in log10(L_OIII)

log10_lum_min = 41.5   # lower limit
log10_lum_max = 44.0   # an arbitarily large upper limit

mask = (log10_lum >= log10_lum_min) & (log10_lum <= log10_lum_max)
pos = positions[mask]
pos = pos.astype(np.float32)

print(f"The luminosity cut selects {len(pos)} galaxies over the full box, stored in array of shape {pos.shape}")


###################################################
# reionization field data
###################################################

reion_data_dir = os.path.join(os.getcwd(), './reion_files_mixed_compare_cases/')

# Load field data
field_data = np.load(reion_data_dir + f'reionCube_{reioncase}_z{z:06.2f}.npz')

# Extract the stored arrays
del_HI = field_data['Delta_HI_arr']        # Shape (80,80,80)
del_matter  = field_data['densitycontr_arr']    # Shape (80,80,80)
qion   = field_data['qi_arr']              # Shape (80,80,80)

Tbar = 27 * ((1 + z) / 10)**0.5 * (0.15 / (omega_m * h**2))**0.5 * (omega_b * h**2 / 0.023)     ## units : mk

print("Tbar (mK) at z ="+str(z)+" is "+str(Tbar))
del_T21 = del_HI * Tbar      ## units of mK

print(f"Loaded field data for z={z}: qion shape {qion.shape}, del_matter shape {del_matter.shape}, del_HI shape {del_HI.shape}")
del del_HI

'''
#####################################################################
###################  EXTRACTING REGION  #############################
#####################################################################
'''

# Minimum distance (in cMpc/h) to keep from the subbox boundaries.
# Must be larger than the maximum radius used in the correlation bins.
bufferlength = 14

# size (in cMpc/h) of the galaxy region.
galsubboxsize = 80.0

### IMPORTANT ###
### If you change anything here, remember to change the random galaxy distribution step below (after line 408)

################################################
# Extracting the sub-volume for galaxies 
################################################

def select_galaxies_subbox(positions, boxsize=160.0, subboxsize=80.0, center=None):
    """
    Select galaxies lying inside a cubic sub-volume centered at 'center'.
    If center is None, use the middle of the big box.

    positions  : array of shape (N, 3) containing galaxy coordinates (x, y, z)
    boxsize    : full box size (cMpc/h)
    subboxsize : size of subvolume in cMpc/h
    center     : (cx, cy, cz). If None, it sets to (boxsize/2, boxsize/2, boxsize/2)

    While extracting from the big box, this ensures the requested sub-box lies fully 
    inside the domain [0, boxsize]^3. That is, NO periodic wrapping is done.

    Returns:
        positions_subbox
    """
    # Default centre = midpoint of box
    if center is None:
        cx = cy = cz = boxsize / 2.0
    else:
        cx, cy, cz = center

    half = subboxsize / 2.0

    # Physical boundaries
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half
    zmin, zmax = cz - half, cz + half

    print(f"x-range: [{xmin:.3f}, {xmax:.3f}]")
    print(f"y-range: [{ymin:.3f}, {ymax:.3f}]")
    print(f"z-range: [{zmin:.3f}, {zmax:.3f}]")
    # ----------- Boundary safety check -----------
    if xmin < 0 or ymin < 0 or zmin < 0 or \
       xmax > boxsize or ymax > boxsize or zmax > boxsize:
        raise ValueError(
            f"Sub-box [{xmin},{xmax}]×[{ymin},{ymax}]×[{zmin},{zmax}] "
            f"extends outside the domain [0, {boxsize}]!"
        )

    # Apply mask
    mask = (
        (positions[:, 0] >= xmin) & (positions[:, 0] < xmax) &
        (positions[:, 1] >= ymin) & (positions[:, 1] < ymax) &
        (positions[:, 2] >= zmin) & (positions[:, 2] < zmax)
    )

    subbox_positions = positions[mask]

    # Re-origin positions so that the sub-box starts at (0, 0, 0)
    subbox_positions = subbox_positions - np.array([xmin, ymin, zmin])

    return subbox_positions


## Fetch the (80 cMpc/h)^3 galaxy region 
positions_subbox = select_galaxies_subbox(pos,boxsize=fullbox_len, subboxsize=galsubboxsize)

## shift the galaxies keeping a distance of bufferlength from all the edges
positions_subbox = positions_subbox + bufferlength


print("Number of lum-selected galaxies inside the inner boundary of sub-volume:", len(positions_subbox))
print("x-min, x-max:", positions_subbox[:,0].min(), positions_subbox[:,0].max())
print("y-min, y-max:", positions_subbox[:,1].min(), positions_subbox[:,1].max())
print("z-min, z-max:", positions_subbox[:,2].min(), positions_subbox[:,2].max())

print("**********************")

#### these are the number of true galaxies in the box, we must throw these many number of random galaxies as well 
#### For the null cross-correlations, we dont use the information of true galaxies beyond this point

n_gal_total = len(positions_subbox)  



################################################
# Extracting the sub-volume for 21 cm fields 
################################################


def select_field_subbox(field, boxsize=160.0, subboxsize=80.0, cellsize=2.0, center=None):
    """
    Extract a cubic sub-volume centered at 'center'.
    If center is None, use the middle of the big box.

    field      : 3D array (Ngridx, Ngridy, Ngridz)
    boxsize    : full box size (cMpc/h)
    subboxsize : size of subvolume in cMpc/h
    cellsize   : grid cell size in cMpc/h
    center     : (cx, cy, cz). If None, it sets to (boxsize/2, boxsize/2, boxsize/2)
    
    Returns:
        field_subbox
    """
    # Default centre = midpoint of box
    if center is None:
        cx = cy = cz = boxsize / 2.0
    else:
        cx, cy, cz = center

    half = subboxsize / 2.0

    # Physical boundaries
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half
    zmin, zmax = cz - half, cz + half

    # Convert to grid indices
    i_min = int(xmin / cellsize)
    i_max = int(xmax / cellsize)  # exclusive
    j_min = int(ymin / cellsize)
    j_max = int(ymax / cellsize)
    k_min = int(zmin / cellsize)
    k_max = int(zmax / cellsize)
    print(f"i: [{i_min}, {i_max}),  j: [{j_min}, {j_max}),  k: [{k_min}, {k_max})")

    field_subbox = field[i_min:i_max, j_min:j_max, k_min:k_max]

    return field_subbox #, (i_min, i_max, j_min, j_max, k_min, k_max)



# Total size of the 21cm subbox, including bufferlength padding on both sides.
# This ensures the 21cm field covers the galaxy subbox plus the required boundary margin (viz. twice the bufferlength.
Tb21cm_subboxsize = galsubboxsize + 2.0 * bufferlength

del_T21_subbox = select_field_subbox(del_T21, boxsize=fullbox_len, subboxsize=Tb21cm_subboxsize)
print("Original shape; New shape = ", del_T21.shape,del_T21_subbox.shape)



##########################
# Wedge filter
##########################

'''
omega_l = 1-omega_m
Ez = np.sqrt(omega_m * (1.0 + z)**3 + omega_l)
c_by_H0_100_Mpc = 2997.92458 # c / (H0 / h)
xcom_z = script_fortran_modules.ionization_map.comoving_dist(z)
xz = xcom_z / c_by_H0_100_Mpc # dimensionless
sin_FoV = 1.0 
slope_wedge = sin_FoV* xz * Ez / (1.0 + z)  # wedge slope
'''
sin_FoV = 1.0 #1.0
slope_wedge = sin_FoV*3.27
print("slope_wedge =",slope_wedge)

def wedge_filter(Tb_field, C, cellsize=2.0):
    
    Ngridx, Ngridy, Ngridz = Tb_field.shape
    Tb_k = np.fft.fftn(Tb_field)

    Lx, Ly, Lz = Ngridx*cellsize, Ngridy*cellsize, Ngridz*cellsize

    print("[wedge_filter] boxlength (cMpc/h) of input field  = ",Lx, Ly, Lz)

    ## Conventional Fourier grid
    ## We explicitly multiply by 2*pi/dx because np.fft.fftfreq already multiples by 1/Ngridx --- therefore, the overall factor is 2*pi/(Ngridx * dx) = 2*pi/Lx
    ## See : np.fft.fftfreq --> https://numpy.org/doc/2.3/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq

    kx = 2 * np.pi * np.fft.fftfreq(Ngridx) * Ngridx / Lx  ## 1/dx = Ngridx / Lx
    ky = 2 * np.pi * np.fft.fftfreq(Ngridy) * Ngridy / Ly  ## 1/dy = Ngridx / Ly
    kz = 2 * np.pi * np.fft.fftfreq(Ngridz) * Ngridz / Lz  ## 1/dz = Ngridx / Lz

    print("[wedge_filter] kx, ky, kz shapes = ",kx.shape, ky.shape, kz.shape)


    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    k_perp = np.sqrt(KX**2 + KY**2)
    k_par = np.abs(KZ)

    mask = k_par <= C * k_perp
    Tb_k[mask] = 0

    Tb_filtered = np.real(np.fft.ifftn(Tb_k))

    Tb_filtered = Tb_filtered.astype(np.float32)

    return Tb_filtered, Tb_k, k_perp, k_par, mask
   


#####################################################################


# Create a figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 1, figsize=(8, 7))
fig2, axes2 = plt.subplots(1, 1, figsize=(8, 7))
fig3, axes3 = plt.subplots(1, 1, figsize=(8, 7))


'''
#####################################################################
###################   1NN-CDF CALCULATION ###########################
#####################################################################
'''


# Galaxy count per set for the subbox region (manually)
if log10_lum_min == 41.5:
    n_gal = 943     # 8 sets of 943 galaxies : 2 dropped → n_gal 
elif log10_lum_min == 42.5:
    n_gal = int(n_gal_total)
else:
    raise ValueError(f"Unsupported log10_lum_min value: {log10_lum_min}")


print("[1NN-CDF] n_gal for each set of 1NN =",n_gal)
print("[1NN-CDF] n_gal_total in sub-box = ",n_gal_total)


def get_kNN_from_catalog(bins_knn, boxsize, pos, delta, matter_grid, query_type, query_grid, threshold, kNN,seedvalue):
    # Calculate the number of full sets
    n_full_sets = n_gal_total // n_gal
    # Calculate remainder for the last set
    remainder = n_gal_total % n_gal
    print("[get_kNN_from_catalog] n_full_sets = ",n_full_sets)
    print("[get_kNN_from_catalog] remainder = ",remainder)
    # Create array of set sizes
    if remainder == 0:
        gal_per_set = np.full(n_full_sets, n_gal, dtype=int)
    else:
        gal_per_set = np.full(n_full_sets, n_gal, dtype=int)
        # excluding the excess galaxies
        #gal_per_set = np.append(gal_per_set, remainder)  # last set

    print("[get_kNN_from_catalog] Number of galaxies in each set = ", gal_per_set)
    print(f"[get_kNN_from_catalog] Check: total={n_gal_total}, sum_over_sets={gal_per_set.sum()}")

    if gal_per_set.sum() != n_gal_total:
        print(f"[get_kNN_from_catalog] Total number of galaxies in sets does not match n_gal_total! difference={gal_per_set.sum() - n_gal_total}!")



    delta_threshold = {'type': 'percentile', 'value': threshold}

    '''
    assert sum(gal_per_set) <= len(pos), (
        f"Not enough tracers in 'pos' for unique sampling: "
        f"need {sum(gal_per_set)}, but only {len(pos)} available."
    )
    '''
    remaining = np.arange(len(pos))  # indices of remaining tracers
    subsamples = []

    np.random.seed(seedvalue)

    for size in gal_per_set:
        # Randomly choose indices from the remaining pool
        chosen_idx = np.random.choice(remaining, size=size, replace=False)
        # Select the corresponding rows from pos
        subsamples.append(pos[chosen_idx])
        # Remove chosen indices from the remaining pool
        remaining = np.setdiff1d(remaining, chosen_idx, assume_unique=True)

    # Stack all subsamples into one array
    all_subsamples = np.vstack(subsamples)   # shape = (sum(gal_per_set), 3)
    print("[get_kNN_from_catalog] all_subsamples.shape = ",all_subsamples.shape)
    # Check for uniqueness
    # Convert to tuples for easy row-wise comparison
    unique_rows = np.unique(all_subsamples, axis=0)
    if len(unique_rows) == len(all_subsamples):
        print("[get_kNN_from_catalog] All subsamples are unique (no duplicate tracers).")
    else:
        print("[get_kNN_from_catalog] Duplicate tracers found across subsamples!")

    psi = np.zeros((len(kNN), len(gal_per_set), bin_len))

    for ii in tqdm(range(len(gal_per_set)), desc="Computing kNN CDF for each sub-set"):

        pos_sample = subsamples[ii]  #float array of shape (ntracers_in_set, 3)
        cdf_tracer, cdf_field, cdf_joint = cc.tracer_field_cross_CDF(
            pos_sample,  
            bins_knn,         
            delta,       
            matter_grid,  
            query_type,  
            query_grid,  
            delta_threshold,  
            boxsize, 
            galsubboxsize,     
            kNN, 
            bufferlength,        
            n_threads=32    
        )

        for k in range(len(kNN)):
            psi[k, ii] = cdf_joint[:, k] / (cdf_tracer[:, k] * cdf_field)   ## float array of shape (len(kNN), n_sets, len(bins))

    # Average along the ii axis (axis=1)
    psi_avg = np.mean(psi, axis=1)

    print("[get_kNN_from_catalog] Shape of psi_avg:", psi_avg.shape,"\n")  ## should be a float array of shape (len(kNN), len(bins))

    return psi_avg



start_time = time.time()


# Computing the null cross-correlation for random galaxy positions and ONLY 21cm noise with wedge filtering

psi_null = np.zeros((len(kNN), n_null, bin_len))

for i in tqdm(range(n_null), desc="Computing kNN CDF for random galaxies"):

    seedvalue = seed_arr[i]  # ensure a new seed used for each iteration

    ''' 21cm noise fields ''' 

    data_dir = './tools21cm_datafiles_with_unq_noise_'+str(tobs)+'hrs_ngrid'+str(ngrid)+'/files_seed'+str(seedvalue)
    print("seedvalue = ",seedvalue)
    savefile_21loc = os.path.join(data_dir, 'AAstar_noise21cm_cube_seed'+str(seedvalue) + '.npz')

    ##### only noise being used
    del_T21_noise_sample_full = np.load(savefile_21loc)['noisecube_21cm'].astype(np.float32) 
  
    ##### apply wedge on the full 21cm noise cube
    del_T21_noise_sample_filtered_full, _, _, _, _ = wedge_filter(del_T21_noise_sample_full, slope_wedge)

    ##### mean-subtraction on the full filtered 21cm noise cube
    del_T21_noise_sample_filtered_full = del_T21_noise_sample_filtered_full - np.mean(del_T21_noise_sample_filtered_full)


    ### pick out the noise from the full cube corresponding to the region : [0, bufferlength + galsubboxsize + bufferlength] = [0, Tb21cm_subboxsize] 
    del_T21_noise_sample_filtered_subbox = select_field_subbox(del_T21_noise_sample_filtered_full,boxsize=fullbox_len, subboxsize=Tb21cm_subboxsize)

    del_T21_noise_sample_filtered_subbox = del_T21_noise_sample_filtered_subbox - np.mean(del_T21_noise_sample_filtered_subbox)

    ''' Random galaxies  ''' 
    np.random.seed(seedvalue)

    # Generate random galaxy positions uniformly inside the subbox:
    # the galaxies will span [bufferlength, bufferlength + galsubboxsize] along each axis  --- i.e., inhabit same region as the true galaxies (see line 235)
    pos_sample_subbox = np.random.uniform(low=bufferlength,high=galsubboxsize+bufferlength,size=(n_gal_total, 3)).astype(np.float32)

    # to mimic query points for 1NN, the galaxies should span [bufferlength+bufferlength, bufferlength + galsubboxsize - bufferlength] along each axis.
    # i.e. the galaxies should span [2*bufferlength, galsubboxsize] along each axis.
    #pos_sample_subbox = np.random.uniform(low=2*bufferlength,high=galsubboxsize,size=(n_gal_total, 3)).astype(np.float32)

    num_rand_gal = pos_sample_subbox.shape[0] 
    print("[Random Galaxies] n_gal within boundary of subbox  =",num_rand_gal)
    print("[Random Galaxies] x-min, x-max:", pos_sample_subbox[:,0].min(), pos_sample_subbox[:,0].max())
    print("[Random Galaxies] y-min, y-max:", pos_sample_subbox[:,1].min(), pos_sample_subbox[:,1].max())
    print("[Random Galaxies] z-min, z-max:", pos_sample_subbox[:,2].min(), pos_sample_subbox[:,2].max())



    # The galaxy sub-region spans from [buffer_length, buffer_length + gal_subbox_size].
    # We choose the query grid to span a slightly smaller region,
    # specifically [2 * buffer_length, gal_subbox_size],
    # following the convention used in crosscorrelations_utils.py.

    print(f"[1NN-CDF] Using a {nquery}×{nquery}×{nquery} query grid over the restricted galaxy region.")

    psi_null_thisCatalog = get_kNN_from_catalog(bins_knn = bins_knn, boxsize = Tb21cm_subboxsize, pos = pos_sample_subbox, delta = del_T21_noise_sample_filtered_subbox, matter_grid=del_T21_noise_sample_filtered_subbox.shape[0], query_type='grid', query_grid=nquery, threshold=threshold, kNN=kNN,seedvalue=seedvalue)  #array of shape (len(kNN), len(bins))

    # Store in the preallocated array
    psi_null[:, i, :] = psi_null_thisCatalog  # store along the i axis

    print("##################")


end_time = time.time()
elapsed = end_time - start_time
print(f"Execution time: {elapsed//60:.0f} min {elapsed%60:.2f} sec")

###chisq calculation FOR 1-nn

null = psi_null[0]  ## picks out the 1NN measurements - shape :  (n_null, bin_len)

cov_1nn = np.cov(null,rowvar=False,bias=True)
corrMatrix_1nn = np.corrcoef(null, rowvar=False)
print("[nquery = "+str(nquery)+"] corrMatrix_1NN Min value:", np.min(corrMatrix_1nn))
print("[nquery = "+str(nquery)+"] corrMatrix_1NN Max value:", np.max(corrMatrix_1nn))

hartlap = (n_null-bin_len-2)/(n_null-1)
c_inv_1nn = hartlap*np.linalg.inv(cov_1nn)


#### Computing the "mean" null 
null_1nn = np.mean(psi_null[0], axis=0) #shape :  (bin_len,)


chi_sq_1nn_null = np.zeros(n_null)


for i in range(n_null):
    chi_sq_1nn_null[i] = (np.transpose(psi_null[0,i] - null_1nn)).dot(c_inv_1nn).dot(psi_null[0,i] - null_1nn)



'''
#####################################################################
###################   PLOTTING   ####################################
#####################################################################
'''


###########################################################################################################################
############################## chisquare distribution of the null samples #################################################
###########################################################################################################################

ax = axes

# Histogram of null values
ax.hist(chi_sq_1nn_null, bins=10, color='grey', edgecolor='black', density=True)

chi_sq_1nn_mean = np.mean(chi_sq_1nn_null)

# Vertical line for observed chi-sq value
ax.axvline(chi_sq_1nn_mean, color='red', linestyle='--', linewidth=2,
           label=f'Mean $\chi^2$ = {chi_sq_1nn_mean:,.0f} \n [Percentile = '+str(threshold)+']')



ax.set_ylim(0.0,0.2)
ax.set_xscale('log')
ax.set_xlabel(r'$\chi^2$')
ax.set_ylabel('P($\chi^2$)')
ax.set_title(r'Region = '+str(region)+r' : $\chi^2$ dist. : 1NN with $N_{\rm gal,set}$ = ' + str(n_gal)+"; AA* (" +str(tobs)+r"hrs) ; $m_w$ = "+str(slope_wedge))
ax.legend()
#plt.grid(True, which='both', linestyle='--', alpha=0.6)


###########################################################################################################################
######################### 1NN cross-correlation function of the null samples ##############################################
###########################################################################################################################

ax = axes2
for i in range(n_null):
    ax.plot(bins_knn[:, 0], psi_null[0, i], color='k', alpha=0.1)

ax.plot(bins_knn[:, 0], np.mean(psi_null[0], axis=0), color='k', ls='dashed', label='Random Galaxies X  ONLY 21cm Noise AA* + FG', lw=2)


ax.set_xlabel('r (cMpc/h)')
ax.set_ylabel(r'$\psi_1(r)$')

ax.set_title(r"Region = "+str(region)+r": $N_{\rm gal}$ = " + str(n_gal)+"; AA* (" +str(tobs)+r"hrs) ; $m_w$ = "+str(slope_wedge))
ax.legend(loc='lower right')
ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)


###########################################################################################################################
############################# 1NN correlation matrix of the null samples ##############################################
###########################################################################################################################
ax = axes3


cax = ax.matshow(corrMatrix_1nn, vmax=1, vmin=-1, cmap='coolwarm')
fig3.colorbar(cax, ax=ax)  # attach colorbar to this specific ax

ax.set_title(r"[1NN-region] : $N_{\rm real}$ = " + str(n_null)+"; AA* (" +str(tobs)+r"hrs) ; $m_w$ =  "+str(slope_wedge))

ax.set_xticks(np.arange(bin_len))
ax.set_yticks(np.arange(bin_len))
ax.set_xticklabels(bins_knn[:,0] , rotation=45, ha='left')
ax.set_yticklabels(bins_knn[:,0] , rotation=45, ha='right')
ax.set_xlabel('r (cMpc/h)')
ax.set_ylabel('r (cMpc/h)')




fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()


#plt.show()




# Directory to save all figures as pdf
out_dir_pdf = "./combined_pdf_files_subbox"
os.makedirs(out_dir_pdf, exist_ok=True)  # Create if not present

# Common filename stem
stem = f"OIIIcrossxHI_1NNsigma_Noise{tobs}hrs_AAstar_nreal{n_null}_FGslope{slope_wedge}_region{region}"

# Save figures
fig.savefig(os.path.join(out_dir_pdf, f"chisq_{stem}.pdf"), bbox_inches='tight')
fig2.savefig(os.path.join(out_dir_pdf, f"summary_stats_{stem}.pdf"), bbox_inches='tight')
fig3.savefig(os.path.join(out_dir_pdf, f"corrMatrix_{stem}.pdf"), bbox_inches='tight')


# Full filename
out_dir_npz = "./saved_combined_cov_corr_files_subbox"
filename = os.path.join(out_dir_npz, f"{stem}.npz")

# Ensure directory exists
os.makedirs(out_dir_npz, exist_ok=True)

np.savez(
    filename,
    corrMatrix_1nn=corrMatrix_1nn,
    cov_1nn=cov_1nn,
    psi_null=psi_null[0], ## picks out the 1NN measurements - shape :  (n_null, bin_len)
    chi_sq_1nn_null=chi_sq_1nn_null,
    )
print(f"File saved: {filename}")


#plt.show()



