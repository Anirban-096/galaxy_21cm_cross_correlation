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

#from plotSettings import setFigure 
#setFigure(fontsize=10)

### Present as a separate .py file in the directory : kNN_2PCF_Project_Gadget2/
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
parser.add_argument(
    "--noise_truth",
    type=int,
    default=99295,
    help="Use true noise (1) or simulated noise (0). Default: 1",
)

args = parser.parse_args()
n_null = args.n_null
true_noise_seed = args.noise_truth

print("USER INPUT : Redshift z = ",z)
print("USER INPUT : n_null = ",n_null)
print("USER INPUT : Total 21cm obs. hrs. = ",tobs)
print("USER INPUT : true_noise_seed = ",true_noise_seed)


###### Parameters for 2pt cross correlation function 

thickness = 1 # Mpc/h
rmin=4
rmax=12
delta_r = 1

bin_len = int((rmax-rmin)/delta_r) + 1
bins_xi = np.linspace(rmin,rmax,bin_len)
print("USER INPUT : radial_bins = ", bins_xi)


current_dir=os.getcwd()

#####################################################################


##################################################
# array of seeds (for creating random catalogs) 
##################################################

seedlist=np.genfromtxt('./musicseed_list_4001_entries_no_duplicates.txt',dtype=int) ### 4001 entries

# Select the first "n_null" seeds for calculating the null cross-correlations
seed_arr = seedlist[0:n_null]

print("length of seed_arr = ", len(seed_arr))
print("len of unique seeds)   =", len(set(seed_arr)))



#true_noise_seed = 53827 #52520 #53827#68716 #76231 #537691
# Check if 68716 (the seed used as noise_mock) is in seed_arr
result = np.isin(true_noise_seed, seed_arr)
print("Is seed="+str(true_noise_seed)+" in random_noise_sample_seed_arr?", result)


sin_FoV = 1.0
slope_wedge = sin_FoV*3.27

print("slope_wedge =",slope_wedge)


def process_reion_case(
    reioncase,
    z,
    omega_m,
    omega_b,
    h,
    true_noise_seed,
    n_null,
    ):
    """
    Load galaxy and reionization data for a given reionization scenario, 
    apply observational noise and foreground wedge to the 21 cm brightness temperature field, 
    perform a luminosity-based galaxy selection, and compute the 
    two-point cross-correlation function (2PCF) between galaxies and 
    the filtered noisy 21 cm field.

    Parameters
    ----------
    reioncase : str
        Identifier for the reionization model  case 
    z : float
        Redshift at which to load the data.
    omega_m : float
        Total matter density parameter.
    omega_b : float
        Baryon density parameter.
    h : float
        Dimensionless Hubble parameter (H0 / 100 km/s/Mpc).
    true_noise_seed : int
        Seed used to identify the 21 cm noise realization file.
    n_null : int
        Number of null (randomized) realizations for statistical comparison.
    
    Returns
    -------
    xi : ndarray of shape (n_data, n_bins)
        Computed 2-point cross-correlation function between the galaxy 
        distribution and the (noisy + wedge-filtered) 21 cm brightness temperature field.
    bins_xi : ndarray
        Array of radial bin centers used for the 2pt cross-correlation.
    """

    print("------------")
    print(f"Processing CASE = {reioncase}, z = {z}")

    '''
    #####################################################################
    ###################    DATA LOADING    ##############################
    #####################################################################
    '''
    ###################################################
    # galaxy catalog data
    ###################################################

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
    del_T21_SCRIPT = del_HI * Tbar      ## units of mK

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


    '''
    #####################################################################
    ###################  ADDING NOISE + WEDGE ###########################
    #####################################################################
    '''

    ####################################################
    # Wedge filter
    ####################################################

    '''
    omega_l = 1-omega_m
    Ez = np.sqrt(omega_m * (1.0 + z)**3 + omega_l)
    c_by_H0_100_Mpc = 2997.92458 # c / (H0 / h)
    xcom_z = script_fortran_modules.ionization_map.comoving_dist(z)
    xz = xcom_z / c_by_H0_100_Mpc # dimensionless
    sin_FoV = 1.0 
    slope_wedge = sin_FoV* xz * Ez / (1.0 + z)  # wedge slope
    '''

    def wedge_filter(Tb_field, C, cellsize=2.0):
        
        Ngridx, Ngridy, Ngridz = Tb_field.shape
        Tb_k = np.fft.fftn(Tb_field)

        Lx, Ly, Lz = Ngridx*cellsize, Ngridy*cellsize, Ngridz*cellsize

        print("[wedge_filter] boxlength (cMpc/h) of input field  = ",Lx, Ly, Lz)

        ## Conventional Fourier grid
        ## We explicitly multiply by 2*pi/dx because np.fft.fftfreq already multiples by 1/Ngridx --- therefore, the overall factor is 2*pi/(Ngridx * dx) = 2*pi/Lx
        ## See : np.fft.fftfreq --> https://numpy.org/doc/2.3/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq

        kx = 2 * np.pi * np.fft.fftfreq(Ngridx) * Ngridx / Lx  ## 1/dx = Ngridx / Lx
        ky = 2 * np.pi * np.fft.fftfreq(Ngridy) * Ngridy / Ly  ## 1/dy = Ngridy / Ly
        kz = 2 * np.pi * np.fft.fftfreq(Ngridz) * Ngridz / Lz  ## 1/dz = Ngridz / Lz

        print("[wedge_filter] kx, ky, kz shapes = ",kx.shape, ky.shape, kz.shape)


        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

        k_perp = np.sqrt(KX**2 + KY**2)
        k_par = np.abs(KZ)

        mask = k_par <= C * k_perp
        Tb_k[mask] = 0

        Tb_filtered = np.real(np.fft.ifftn(Tb_k))

        Tb_filtered = Tb_filtered.astype(np.float32)

        return Tb_filtered, Tb_k, k_perp, k_par, mask
       


    ####################################################
    # Add thermal noise and apply the foreground wedge
    ####################################################

    

    ##### noise being added to the SCRIPT field
    file_21loc_mock = os.path.join('./tools21cm_datafiles_with_unq_noise_'+str(tobs)+'hrs_ngrid'+str(ngrid)+'/files_seed'+str(true_noise_seed), 'AAstar_noise21cm_cube_seed'+str(true_noise_seed)+'.npz')
    del_T21_noisy_full = del_T21_SCRIPT.astype(np.float32) + np.load(file_21loc_mock)['noisecube_21cm'].astype(np.float32)   ##### noise being added to SCRIPT fields 
  
    ##### apply wedge on the full 21cm (SCRIPT+noise) cube
    del_T21_noisy_filtered_full, _, _, _, _ = wedge_filter(del_T21_noisy_full, slope_wedge)

    ##### mean-subtraction on the full filtered 21cm data cube
    del_T21_noisy_filtered_full = del_T21_noisy_filtered_full - np.mean(del_T21_noisy_filtered_full)


    ### pick out the final field from the full cube corresponding to the region : [0, bufferlength + galsubboxsize + bufferlength] = [0, Tb21cm_subboxsize] 
    del_T21_noisy_filtered_subbox = select_field_subbox(del_T21_noisy_filtered_full,boxsize=fullbox_len, subboxsize=Tb21cm_subboxsize)

    mean_sub_field = del_T21_noisy_filtered_subbox - np.mean(del_T21_noisy_filtered_subbox)
    
   

    '''
    #####################################################################
    ##################### SUMMARY CALCULATION ###########################
    #####################################################################
    '''

    
    ####################################################
    # Perform 2-point cross correlation
    ####################################################
    n_data = 1
    # Compute cross-correlation 
    xi = np.zeros((n_data, bin_len))
    for i in tqdm(range(n_data), desc="Computing 2PCF for true galaxies"):
        xi[i] = cc.CrossCorr2pt(bins_xi, Tb21cm_subboxsize, positions_subbox, mean_sub_field, thickness)

    print("Finished computing 2PCF")
    print("------------")
    return xi, bins_xi, n_gal_total





xi_fid, bins_xi, n_gal  = process_reion_case("fiducial",z,omega_m, omega_b, h, true_noise_seed, n_null )
xi_low, _ , n_gal  = process_reion_case("alt_case_low",z,omega_m, omega_b, h, true_noise_seed, n_null)
xi_high, _ , n_gal  = process_reion_case("alt_case_high",z,omega_m, omega_b, h, true_noise_seed, n_null)

# Load the covariance matrix of random galaxies x random 21cm noise ONLY
region=0
stem = f"OIIIcrossxHI_2PCFsigma_Noise{tobs}hrs_AAstar_nreal{n_null}_FGslope{slope_wedge}_region{region}"

# Full filename
out_dir_npz = "./saved_combined_cov_corr_files_subbox"
filename = os.path.join(out_dir_npz, f"{stem}.npz")

cov_2pt = np.load(filename)['cov_2pt']
xi_null = np.load(filename)['xi_null']
null_2pt = np.mean(xi_null, axis=0)
print("xi_null.shape =",xi_null.shape)


bin_len=len(bins_xi)
hartlap = (n_null-bin_len-2)/(n_null-1)
c_inv_2pt = hartlap*np.linalg.inv(cov_2pt)

print("Mean null_2pt =",null_2pt)


# Calculate the data vectors for true galaxies x noise-contaminated 21cm fields
data_fid_2pt = np.mean(xi_fid, axis=0)
are_equal = np.array_equal(xi_fid[0], data_fid_2pt)
print(f"Exact equality ?: {are_equal}")
print("data_fid_2pt = ",data_fid_2pt)

data_low_2pt = np.mean(xi_low, axis=0)
are_equal = np.array_equal(xi_low[0], data_low_2pt)
print(f"Exact equality ?: {are_equal}")
print("data_low_2pt = ",data_low_2pt)

data_high_2pt = np.mean(xi_high, axis=0)
are_equal = np.array_equal(xi_high[0], data_high_2pt)
print(f"Exact equality ?: {are_equal}")
print("data_high_2pt = ",data_high_2pt)


outfigdir = "main_plots_paper"
os.makedirs(outfigdir, exist_ok=True)



chi_sq_2pt_fid_low = (np.transpose(data_fid_2pt - data_low_2pt)).dot(c_inv_2pt).dot(data_fid_2pt - data_low_2pt)
chi_sq_2pt_fid_high = (np.transpose(data_fid_2pt - data_high_2pt)).dot(c_inv_2pt).dot(data_fid_2pt - data_high_2pt)
chi_sq_2pt_low_high = (np.transpose(data_low_2pt - data_high_2pt)).dot(c_inv_2pt).dot(data_low_2pt - data_high_2pt)

plt.figure(figsize=(10,8))

plt.plot(bins_xi, data_fid_2pt, color='C0', ls='solid', label=r'Fiducial ($\alpha_{\mathrm{esc}} = -0.3$)', lw=2)
plt.plot(bins_xi, data_low_2pt, color='C2', ls='dashed', label=r"$\alpha_{\mathrm{esc}} = -0.45 : \chi^2_{\rm 2pt;(fid,M1)} =$"+str(np.round(chi_sq_2pt_fid_low,3)), lw=2)
plt.plot(bins_xi, data_high_2pt, color='C1', ls='dashed', label=r"$\alpha_{\mathrm{esc}} = -0.15 : \chi^2_{\rm 2pt;(fid,M2)} =$"+str(np.round(chi_sq_2pt_fid_high,3)), lw=2)
for xi in xi_null:
    plt.plot(bins_xi, xi, color='black', alpha=0.05, lw=0.5)
plt.plot(bins_xi, null_2pt, color='black', ls='solid',label="Random galaxies x (21cm Noise + Wedge) \n"+r"($N_{\mathrm{null,real}} = $"+str(n_null)+")", lw=2)
plt.xlabel('r (cMpc/h)')
plt.ylabel(r'$\xi(r)~$[mK]')
plt.title('2-point cross-correlation \n '+str(n_gal)+' galaxies , SKA-AA* ('+str(tobs)+'hrs) + Foreground Wedge ($m_w$ = '+str(slope_wedge)+')',fontsize=14)
plt.legend()
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(outfigdir, f"2pcf_stat_comparision_nreal{n_null}_region{region}.pdf"), dpi=300)


# Print results 
print("===================================")
print("[SUB-BOX ",region," ] Chi-square comparison results:")
print("-----------------------------------")
print(f"χ²(fiducial vs low) = {chi_sq_2pt_fid_low:.4f}")
print(f"χ²(fiducial vs high) = {chi_sq_2pt_fid_high:.4f}")
print(f"χ²(low vs high) = {chi_sq_2pt_low_high:.4f}")
print("===================================")



fig3, axes3 = plt.subplots(1, 1, figsize=(8, 7))
ax=axes3

corrMatrix_2pt = np.load(filename)['corrMatrix_2pt']

cax = ax.matshow(corrMatrix_2pt, vmax=1, vmin=-1, cmap='coolwarm')
fig3.colorbar(cax, ax=ax)  # attach colorbar to this specific ax

ax.set_title("2-point cross-correlation  \n"+ r" $N_{\mathrm{null,real}} = $"+str(n_null)+' , SKA-AA* ('+str(tobs)+r'hrs) + Wedge ($m_w$ = '+str(slope_wedge)+')', pad=10)


ax.set_xticks(np.arange(bin_len))
ax.set_yticks(np.arange(bin_len))
ax.set_xticklabels(bins_xi , rotation=45, ha='left')
ax.set_yticklabels(bins_xi, rotation=45, ha='right')
ax.set_xlabel('r (cMpc/h)')
ax.set_ylabel('r (cMpc/h)')
plt.tight_layout()


plt.savefig(os.path.join(outfigdir, f"2pcf_corrMat_comparision_nreal{n_null}_region{region}.pdf"), dpi=300)
print(f"Figures saved to : ./{outfigdir}")
plt.show()


