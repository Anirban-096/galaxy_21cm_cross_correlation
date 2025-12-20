###############################################################################################################################
#### Code developed by Kwanit 
#### Modified by Anirban
####################


#-------------------------------------------  Importing the required libraries  -----------------------------------------------
from tqdm import tqdm

import numpy as np
import pyfftw
from scipy import interpolate
import scipy.spatial
import time
import MAS_library as MASL
import smoothing_library as SL

###############################################################################################################################

#-------------------------------------------------  Function definitions  -----------------------------------------------------

# Function that returns the the 2-point Cross-Correlation Function between a set of discrete tracers and a continuous field using the stacking method

def CrossCorr2pt(bins, boxsize, pos, delta, thickness, threads=32):
    '''
    Parameters
    ----------
    bins : float array of shape (m,)
        Set of m radial distances at which the 2PCF will be computed

    boxsize : float
        The length (in Mpc/h) of the cubic box containing the tracers and the field

    pos : float array of shape (n, 3)
        3D positions of the n tracers (e.g., galaxies) inside the box, given in Cartesian coordinates (x, y, z) within the range [0, boxsize]

    delta : float array of shape (ngrid, ngrid, ngrid)
        Overdensity field defined on a uniform grid with ngrid³ points

    thickness : float
        Thickness (in Mpc/h) of the spherical shell used for stacking to compute the 2PCF

    threads : number of cores to parallelize over, default number of threads = 8

    Returns
    -------
    xi : float array of shape (m,)
        The 2-point cross-correlation function (2PCF) between the tracer positions and the field, computed at each of the m radial bins.
    '''
    print(" ~~~~~~~~~~~~~~ New Version [AC] ~~~~~~~~~~~~~~~~~~~~ ")
    # Calculating the number of grid points along each axis of the field delta
    shape = np.shape(delta)
    if len(shape) != 3 or shape[0] != shape[1] or shape[1] != shape[2]:
        raise ValueError("Error: Input array is not cubical (n, n, n).")
    ngrid = shape[0]
    print("[CrossCorr2pt] delta.shape = ",delta.shape)
    # Calculating the grid cell size
    grid_cell_size = boxsize / ngrid  
    print("[CrossCorr2pt] grid_cell_size = ",grid_cell_size)

    # Fourier Transforming the delta overdensity field
    pyfftw.interfaces.cache.enable()
    delta_k = pyfftw.interfaces.numpy_fft.rfftn(delta, threads=threads)
    print("[CrossCorr2pt] delta_k.shape = ",delta_k.shape)
    # Initialize output array
    delta_smooth = np.zeros((len(bins), ngrid, ngrid, ngrid), dtype=np.float32)

    # Compute smoothed field for each R value in bins
    for i, R in enumerate(bins):
        # Defining a spherical shell window function in real space
        W = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
        coords = (np.arange(ngrid) - ngrid // 2) * grid_cell_size
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        r_grid = np.sqrt(X**2 + Y**2 + Z**2)

        W[(r_grid >= R - thickness / 2) & (r_grid <= R + thickness / 2)] = 1.0
        W /= np.sum(W)  # Normalize before FFT (avoids NaNs)
        W_shifted = np.fft.ifftshift(W)  # Center to corner for FFT alignment

        # Taking convolution of window function with overdensity field delta
        W_k = pyfftw.interfaces.numpy_fft.rfftn(W_shifted, threads=threads)
        delta_k_smooth = delta_k * W_k
        delta_smooth[i] = pyfftw.interfaces.numpy_fft.irfftn(delta_k_smooth, threads=threads) / np.sum(W)

    # Interpolating the field at the tracer (galaxy) positions
    delta_interp = np.zeros((len(bins), len(pos)))  # Shape (number_of_bins, number_of_tracers)

    # Perform interpolation for each smoothed field
    for i, R in enumerate(bins):
        density_interpolated = np.zeros(pos.shape[0], dtype=np.float32)
        MASL.CIC_interp(delta_smooth[i], boxsize, pos.astype(np.float32), density_interpolated)
        delta_interp[i] = density_interpolated

    # Computing the 2-point Cross-Correlation Function by averaging over the interpolated field at the tracer positions
    xi = np.zeros_like(bins)
    for i, R in enumerate(bins):
        xi[i] = np.mean(delta_interp[i])

    return xi
#------------------------------------------------------------------------------------------------------------------------------

# Function that returns interpolating functions for the empirical CDFs of the given k-nearest neighbour distances

def cdf_vol_knn(vol):
    '''    
    Parameters
    ----------
    
    vol: float array of shape (n, l) where 'n' is the number of query points and 'l' is the number of nearest neighbours queried
         Sorted array of nearest neighbour distances

    Returns
    -------

    cdf: list
        list of interpolated empirical CDF functions that can be evaluated at desired distance bins
    '''
    
    cdf = []
    n = vol.shape[0]
    l = vol.shape[1]
    gof = ((np.arange(0, n) + 1) / (n*1.0))
    for c in range(l):
        ind = np.argsort(vol[:, c])
        s_vol= vol[ind, c]
        cdf.append(interpolate.interp1d(s_vol, gof, kind='linear', bounds_error=False))
    return cdf

#------------------------------------------------------------------------------------------------------------------------------

# Function that returns the kNN-CDFs for the given nearest-neighbour distances, evaluated at the given distance bins    

def calc_kNN_CDF(vol, kNN, bins):
    '''
    Parameters
    ----------
    
    vol: float array of shape (n, l) where 'n' is the number of query points and 'l' is the number of nearest neighbours queried
        Sorted array of nearest neighbour distances
        
    kNN: int list
        the list of nearest neighbours to calculate the distances to

    bins: float array of shape (n_bins, len(kNN)) where n_bins is the number of radial bins
        Distance bins at which we need to evaluate the CDFs for each nearest neighbour

    Returns
    -------

    data: float array of shape (bins.shape[0], len(kNN))
        kNN-CDFs evaluated at the desired distance bins
    '''
    
    data = (np.zeros((bins.shape[0], len(kNN))))
    cdfs = cdf_vol_knn(vol)
    
    for i in range(len(kNN)):
        
        min_dist = np.min(vol[:, i])
        max_dist = np.max(vol[:, i])
        bin_mask = np.searchsorted(bins[:, i], [min_dist, max_dist])
        if bin_mask[1]!=bins.shape[0]:
            if bins[bin_mask[1], i] == max_dist:
                bin_mask[1] += 1
        NNcdf = np.zeros(bins.shape[0])
        NNcdf[:bin_mask[0]] = 0 #Setting the value of the CDFs at scales smaller than the smallest NN distance to 0
        NNcdf[bin_mask[0]:bin_mask[1]] = cdfs[i](bins[bin_mask[0]:bin_mask[1], i])
        NNcdf[bin_mask[1]:] = 1 #Setting the value of the CDFs at scales larger than the largest NN distance to 1
        data[:, i] = NNcdf
        
    return data

#------------------------------------------------------------------------------------------------------------------------------

# Function to compute quantities associated with the tracer-field cross kNN CDFs

def tracer_field_cross_CDF(data_pos, bins, delta_matter, matter_grid, query_type, query_grid, delta_threshold, boxsize, subboxsize, kNN, bufferlength,n_threads=32):
    '''
    Returns the probabilities P_{>k}, P_{>dt} and P_{>k,>dt} that measure the extent of the spatial cross-correlation between the     given discrete tracer positions ('data_pos') and the given continuous field ('delta_matter'), evaluated at the given array of     distances ('bins'):

        1. P_{>k}: 
            the CDF_{kNN} of the discrete tracers

        2. P_{>dt}: 
            the probability of the smoothed density field exceeding the given constant percentile density threshold
            [haven't implemented the constant mass threshold yet :( ]

        3. P_{>k, dt}:
            the joint probability of finding at least 'k' tracers within the given spherical bin and the smoothed density field               exceeding the given density threshold

    The excess cross-correlation can be computed trivially from these quatities:

        Psi_{k, dt} = P_{>k, dt}/(P_{>k}*P_{>dt})
    
    Parameters
    ----------
    
    data_pos: float array of shape (n, 3) where n is the number of data points (tracers)
        array of locations for the discrete tracers in the cosmological box
        
    bins: float array of shape (n_bins, len(kNN)) where n_bins is the number of radial bins
        Distance bins at which we need to evaluate the CDFs for each nearest neighbour
        
    delta_matter: float array of shape (matter_grid, matter_grid, matter_grid)
        the continuous density field interpolated to a 'matter_grid'**3 grid
        
    matter_grid: int
        the dimensions of the density field grid

    query_type: string
        the type of query points to be generated ('grid' or 'random')

    query_grid: int
        the dimension of the query grid (or the cube root of the number of random query points)

    delta_threshold: float
        the percentile value for the constant percentile threshold to be used for the continuous field
        for example, delta_threshold = 75 represents a 75th percentile threshold
        
    boxsize: float:
        the size of the periodic simulation box
        
    kNN: int list
        the list of nearest neighbours to calculate the distances to

    n_threads: int
        the number of threads used in multi-threading while smoothing the density field

    Returns
    -------

    p_gtr_k: float array of shape (bins.shape[0], len(kNN))
        kNN-CDFs of the discrete tracers evaluated at the desired distance bins
        
    p_gtr_dt: float array of shape (len(bins),)
        the probability of the smoothed density field exceeding the given constant percentile density threshold

    p_gtr_k_dt: float array of shape ((len(bins), len(kNN)))
        the joint probability of finding at least 'k' tracers within the given spherical bin and the smoothed density field               exceeding the given density threshold
    '''
    print(" ~~~~~~~~~~~~~~ New Version [AC] ~~~~~~~~~~~~~~~~~~~~ ")    
    #-------------------------------------------------------------------------------------------------------
    #Step 0: Making the box containing the tracers (galaxies) periodic, if not already
    data_pos = data_pos % boxsize
    
    #Step 1: Creating a set of query points  
    
    if query_type == 'grid':
        print(" bufferlength (cMpc/h) = ",bufferlength,"\n")
        print(" galsubboxsize (cMpc/h) = ",subboxsize,"\n")
        #Creating a grid of query points
        # The galaxy region spans [bufferlength, subboxsize + bufferlength] along each axis.
        # To ensure that query points lie strictly inside this region (not touching boundaries and being atleast away by 'bufferlength'),
        # we offset the lower limit by an additional bufferlength, giving:

        x_ = np.linspace(0.+2*bufferlength, subboxsize, query_grid)
        y_ = np.linspace(0.+2*bufferlength, subboxsize, query_grid)
        z_ = np.linspace(0.+2*bufferlength, subboxsize, query_grid)

        x, y, z = np.array(np.meshgrid(x_, y_, z_, indexing='xy'))

        query_pos = np.zeros((query_grid**3, 3))
        query_pos[:, 0] = np.reshape(x, query_grid**3)
        query_pos[:, 1] = np.reshape(y, query_grid**3)
        query_pos[:, 2] = np.reshape(z, query_grid**3)

        print("Query positions range:")
        print("  x: {:.4f} to {:.4f}".format(query_pos[:,0].min(), query_pos[:,0].max()))
        print("  y: {:.4f} to {:.4f}".format(query_pos[:,1].min(), query_pos[:,1].max()))
        print("  z: {:.4f} to {:.4f}".format(query_pos[:,2].min(), query_pos[:,2].max()))

    elif query_type == 'random':
        
        #Creating a set of randomly distributed query points
        np.random.seed()
        query_pos = np.random.rand(query_grid**3, 3)*boxsize
        
    else:        
        raise Exception('Please provide a valid query type')
        
    #-------------------------------------------------------------------------------------------------------
        
    #Step 2: Calculate nearest neighbour distances of query points, and the kNN-CDFs for the Halos
    
    #Building the tree
    xtree = scipy.spatial.cKDTree(data_pos, boxsize=boxsize)
    #Calculating the NN distances
    vol, disi = xtree.query(query_pos, k=kNN, workers=-1)
    #Calculating the kNN-CDFs
    p_gtr_k = calc_kNN_CDF(vol, kNN, bins)
    
    #-------------------------------------------------------------------------------------------------------
        
    #Steps 3, 4 & 5: Smooth the matter density field over a smoothing scale equal to a particular radial 
    #bin and interpolate the smoothened density to the query points. Calculate a density threshold for 
    #each radial bin. Calculate the fraction of query points with nearest neighbour distance less than 
    #the radial distance and smoothened density greater than the density threshold.
        
    p_gtr_k_dt = np.zeros((len(bins), len(kNN)))
    p_gtr_dt = np.zeros(len(bins))

    for j, k in enumerate(kNN):
        for i, rs in enumerate(bins[:, j]):    

            #Compute FFT of the filter
            W_k = SL.FT_filter(boxsize, rs, matter_grid, 'Top-Hat', n_threads)
            #Smooth the field
            delta_matter_smooth = SL.field_smoothing(delta_matter, W_k, n_threads)

            #Define the array containing the value of the interpolated density field
            delta_matter_smooth_interp = np.zeros(query_pos.shape[0], dtype = np.float32)

            #Find the value of the interpolated density field at the query positions
            MASL.CIC_interp(delta_matter_smooth, boxsize, query_pos.astype(np.float32), delta_matter_smooth_interp)

            #Calculate the density threshold
            if delta_threshold['type'] == 'const_mass':
                delta_star_rs = ((delta_threshold['value']/rs)**3)-1
            elif delta_threshold['type'] == 'percentile':
                delta_star_rs = np.percentile(delta_matter_smooth_interp, delta_threshold['value'])            
            else:        
                raise Exception('Please provide a valid threshold type')

            ind_gtr_k_dt = np.where((vol[:, j]<rs)&(delta_matter_smooth_interp>delta_star_rs))
            p_gtr_k_dt[i, j] = len(ind_gtr_k_dt[0])/(query_grid**3)

            ind_gtr_dt = np.where(delta_matter_smooth_interp>delta_star_rs)
            p_gtr_dt[i] = len(ind_gtr_dt[0])/(query_grid**3)
    
    return p_gtr_k, p_gtr_dt, p_gtr_k_dt

#------------------------------------------------------------------------------------------------------------------------------

# Function that downsamples galaxies from a given realization, and computes the 2-point cross-correlation and null-correlation

def Sampling2ptCC(bins, boxsize, pos, delta, thickness, n_gal, n_data, n_null, seed_for_true_gal= 1496, seed_for_random_gal = 7203):
    '''     
    Parameters
    ----------
    bins : float array of shape (m,)
        Set of m radial distances at which the 2PCF will be computed
        
    boxsize : float
        The length (in Mpc/h) of the cubic box containing the tracers and the field

    pos : float array of shape (n, 3)
        3D positions of the n tracers (e.g., galaxies) inside the box, given in Cartesian coordinates (x, y, z) within the range [0, boxsize]

    delta : float array of shape (ngrid, ngrid, ngrid)
        Overdensity field defined on a uniform grid with ngrid³ points

    thickness : float
        Thickness (in Mpc/h) of the spherical shell used for stacking to compute the 2PCF
    
    n_gal : int
        number of galaxies to be downsampled from the pos array
    
    n_data : int
        number of independent samples to be drawn from the true galaxy data
    
    n_null : int
        number of independent samples to be drawn from random positions characterizing the null 

    seed_for_true_gal : int, optional (default=1496)
        Random seed to pick "true" galaxies for a particular sample.

    seed_for_random_gal : int, optional (default=7203)
        Random seed to pick "random" galaxies for a particular sample.
    
    Returns
    -------
    xi : float array of shape (n_data, bin_length)
        The 2-point cross-correlation function (2PCF) between the true galaxy positions and the field, computed at each radial bin.
    
    xi_null : float array of shape (n_null, bin_length)
        The 2-point cross-correlation function (2PCF) between randomly generated tracer positions and the field, computed at each radial bin.
    '''
    
    bin_len = len(bins)

    # prepare the list of seeds for "real" galaxies to picked when you form a particular independent sample
    np.random.seed(seed_for_true_gal)  
    seedvalue_arr_real_gal = np.random.randint(0, 100000, size=n_data)

    # prepare the list of seeds for "random" galaxies to be picked when you form a particular independent sample
    np.random.seed(seed_for_random_gal)  
    seedvalue_arr_random_gal = np.random.randint(0, 100000, size=n_null)


    # Computing the 2-point cross-correlation for real data
    xi = np.zeros((n_data, bin_len))
    for i in tqdm(range(n_data), desc="Computing 2PCF for true galaxies"):
        np.random.seed(seedvalue_arr_real_gal[i])  # ensure a new seed each iteration
        indices = np.random.choice(pos.shape[0], size=n_gal, replace=False)
        pos_sample = pos[indices]
        xi[i] = CrossCorr2pt(bins, boxsize, pos_sample, delta, thickness)

    # Computing the null cross-correlation for random positions
    xi_null = np.zeros((n_null, bin_len))
    for i in tqdm(range(n_null), desc="Computing 2PCF for random galaxies"):
        np.random.seed(seedvalue_arr_random_gal[i])  # ensure a new seed each iteration
        pos_sample = np.random.uniform(0, boxsize, size=(n_gal, 3)).astype(np.float32)
        xi_null[i] = CrossCorr2pt(bins, boxsize, pos_sample, delta, thickness)
    
    return xi, xi_null


#------------------------------------------------------------------------------------------------------------------------------

# Function that downsamples galaxies from a given realization, and computes the kNN cross-correlations and null-correlations

def Sampling_kNN_CC(bins, boxsize, pos, delta, matter_grid, query_type, query_grid, threshold, kNN, n_gal, n_data, n_null, seed_for_true_gal= 1496, seed_for_random_gal = 7203, n_threads=24):
    ''' 
    Parameters
    ----------
    bins : float array of shape (n_bins, len(kNN))
        Radial distance bins at which the kNN-CDFs will be computed.
        Each row corresponds to a distance bin, and each column corresponds to a different kNN order.

    boxsize : float
        The size of the periodic simulation box. The coordinates of all tracers
        lie within the range [0, boxsize] in each dimension.

    pos : float array of shape (n, 3)
        3D positions of n galaxies (tracers) in Cartesian coordinates.

    delta : float array of shape (matter_grid, matter_grid, matter_grid)
        The continuous density field, interpolated onto a cubic grid of `matter_grid³` voxels.

    matter_grid : int
        The resolution of the density field grid (i.e., the number of grid points per dimension).

    query_type : string
        The type of query points to be generated for kNN calculations.
        Options:
        - "grid": Query points are placed on a uniform grid.
        - "random": Query points are randomly distributed.

    query_grid : int
        The number of query points per dimension if `query_type='grid'`,
        or the cube root of the total number of random query points if `query_type='random'`.

    threshold : float
        The percentile threshold for defining overdense regions in the density field.
        For example, `threshold = 90` selects the top 10% densest regions.

    kNN : list of int
        A list of k-values for which kNN distances will be computed.
        For example, `kNN = [1, 2, 3]` computes distances to the 1st, 2nd, and 3rd nearest neighbors.

    n_gal : int
        The number of galaxies to randomly select from `pos` in each sampling.
        This allows for statistical resampling.

    n_data : int
        number of independent samples to be drawn from the true galaxy data.
    
    n_null : int
        number of independent samples to be drawn from random positions characterizing the null. 
        
    seed_for_true_gal : int, optional (default=1496)
        Random seed to pick "true" galaxies for a particular sample.

    seed_for_random_gal : int, optional (default=7203)
        Random seed to pick "random" galaxies for a particular sample.

    n_threads : int, optional (default=8)
        The number of threads used for parallel computation, speeding up kNN queries.

    Returns
    -------
    psi : float array of shape (len(kNN), n_samples, len(bins))
        The kNN cross-correlation function Psi_{k, dt}, which quantifies 
        the spatial correlation between the galaxy positions and the density field.

    psi_null : float array of shape (len(kNN), n_samples, len(bins))
        The null cross-correlation function, computed using randomly placed tracers instead of galaxies.
        This serves as a statistical baseline to compare against `psi`.
    ''' 
    
    delta_threshold = {'type': 'percentile', 'value': threshold}
    bin_len = len(bins)

    # prepare the list of seeds for "real" galaxies to picked when you form a particular independent sample
    np.random.seed(seed_for_true_gal)  
    seedvalue_arr_real_gal = np.random.randint(0, 100000, size=n_data)

    # prepare the list of seeds for "random" galaxies to be picked when you form a particular independent sample
    np.random.seed(seed_for_random_gal)  
    seedvalue_arr_random_gal = np.random.randint(0, 100000, size=n_null)
    
    # Computing the kNN cross-correlations
    psi = np.zeros((len(kNN), n_data, bin_len))

    for i in tqdm(range(n_data), desc="Computing kNN CDF for true galaxies"):
        np.random.seed(seedvalue_arr_real_gal[i])  # ensure a new seed each iteration
        indices = np.random.choice(pos.shape[0], size=n_gal, replace=False)
        pos_sample = pos[indices]

        cdf_tracer, cdf_field, cdf_joint = tracer_field_cross_CDF(
            pos_sample,  
            bins,         
            delta,       
            matter_grid,  
            query_type,  
            query_grid,  
            delta_threshold,  
            boxsize,      
            kNN,         
            n_threads    
        )

        for k in range(len(kNN)):
            psi[k, i] = cdf_joint[:, k] / (cdf_tracer[:, k] * cdf_field)

    # Computing the null cross-correlations
    psi_null = np.zeros((len(kNN), n_null, bin_len))

    for i in tqdm(range(n_null), desc="Computing kNN CDF for random galaxies"):
        np.random.seed(seedvalue_arr_random_gal[i])  # ensure a new seed each iteration
        pos_sample = np.random.uniform(0, boxsize, size=(n_gal, 3)).astype(np.float32)

        cdf_tracer, cdf_field, cdf_joint = tracer_field_cross_CDF(
            pos_sample,  
            bins,      
            delta,        
            matter_grid,  
            query_type,   
            query_grid,   
            delta_threshold,  
            boxsize,     
            kNN,          
            n_threads    
        )

        for k in range(len(kNN)):
            psi_null[k, i] = cdf_joint[:, k] / (cdf_tracer[:, k] * cdf_field)

    return psi, psi_null

#------------------------------------------------------------------------------------------------------------------------------
