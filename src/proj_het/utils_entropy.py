import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.stats as spstats
import seaborn as sns
import pandas as pd
import itertools
import os
import pickle as pkl
import altair as alt

from itertools import product
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD, PCA
from numba import jit, njit
from collections import Counter
from scipy import sparse


from .utils_io import load_data, filter_data_by_modrate, get_sizes
from .utils_analysis import run_BMM, run_HBBMM, run_louvain, run_leiden
from .utils_arcs import plot_arcs_from_list

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def get_start_end_coords(p_length, p_offset=0, p_window=100, p_step_size=1):
    p_coords = [ (i, i+p_window) for i in list(range(p_offset+0, p_offset+p_length-p_window, 
                                                     p_step_size)) ]
    return p_coords


def get_start_end_coords_v2(p_genes, dict_sizes, p_start=None, p_end=None, p_window=100, p_step_size=1):
    if p_start is None and p_end is None:
        p_length_of_gene = dict_sizes[p_genes[0]]
        p_coords = [ (i, i+p_window) for i in list(range(0, p_length_of_gene-p_window, 
                                                        p_step_size)) ]
    else:
        p_coords = [ (i, i+p_window) for i in list(range(p_start, p_end-p_window, 
                                                         p_step_size)) ]
    return p_coords


def get_coords(p_coords):
    return np.array([ (x+y)//2 for x, y in p_coords ])

def generate_indpt_reads(df, p_reject_limit=1000000, p_seed=386, p_verbose=False):
    """ Generate randomized reads that respect the column-wise and row-wise modification rates

    Args:
        df (pd.dataframe): sparse dataframe of binary matrix
        p_reject_limit (int): maximum number of iterations
        p_seed (int): seed

    Return:
        df_new (pd.dataframe): row-wise and column-wise randomized 
    """

    p_modrate_per_pos = df.values.sum(axis=0)/df.shape[0]
    counter_mr_per_read = [ (k, v) for k, v in sorted(Counter(df.values.sum(axis=1)).items()) ]
    p_modrate_per_read = np.zeros(counter_mr_per_read[-1][0]+1, dtype=int)
    for k, v in counter_mr_per_read:
        p_modrate_per_read[k] = v

    # perform rejection sampling
    accepted = rejection_sampling(p_modrate_per_pos, p_modrate_per_read, p_reject_limit=p_reject_limit, p_seed=p_seed)
    accepted = np.array(accepted)

    total_mods_sim = accepted.sum().sum()
    total_mods_real = df.sum().sum()

    # calculate the R^2 for column
    ssr_col = ((accepted.sum(axis=0)/accepted.shape[0] - p_modrate_per_pos)**2).sum()
    sst_col = ((p_modrate_per_pos-p_modrate_per_pos.mean())**2).sum()
    R2_col = 1 - ssr_col/sst_col

    # calculate the R^2 for row
    try:
        counter_mr_per_read_final = [ (k, v) for k, v in sorted(Counter(accepted.sum(axis=1)).items()) ]
        p_modrate_per_read_final = np.zeros(counter_mr_per_read[-1][0]+1, dtype=int)
        for k, v in counter_mr_per_read_final:
            p_modrate_per_read_final[k] = v

        ssr_row = ((p_modrate_per_read-p_modrate_per_read_final)**2).sum()
        sst_row = ((p_modrate_per_read_final-p_modrate_per_read_final.mean())**2).sum()
        R2_row = 1 - ssr_row/sst_row

        if p_verbose:
            print("R2 pos: %.3f" % R2_col)
            print("R2 read: %.3f" % R2_row)
            print(total_mods_sim, total_mods_real, total_mods_sim/total_mods_real)
    except IndexError:
        pass

    # convert numpy array to dataframe
    sparse_accepted = sparse.coo_matrix(accepted)
    #df_new = pd.DataFrame(accepted, index=df.index[:accepted.shape[0]], columns=df.columns)
    df_new = pd.DataFrame.sparse.from_spmatrix(sparse_accepted)
    df_new.index = df.index[:accepted.shape[0]]
    df_new.columns = df.columns

    return df_new

@jit(nopython=False, parallel=False)
def rejection_sampling(p_modrate_per_pos, p_modrate_per_read,
                       p_reject_limit=1000000, p_seed=386):
    """ Rejection sampling routine

    Args:
        p_modrate_per_pos (float): target modification read per position to respect
        p_modrate_per_read (float): target modification rate per read to respect
        p_reject_limit (int): maximum number of iterations
        p_seed (int): seed
    Return:
        accepted (list of list of int): binary matrix
    """

    np.random.seed(p_seed)

    p_modrate_per_read_copy = p_modrate_per_read.copy()

    p_length = len(p_modrate_per_pos)
    p_depth = p_modrate_per_read_copy.sum()
    counter, rejected = 0, 0
    accepted = []
    while counter < p_depth:
        
        #read = np.zeros(p_length, dtype=int) #dtype="int64")
        read = [ 0 for _ in range(p_length) ]
        for no, pr in enumerate(p_modrate_per_pos):
            if np.random.random() <= pr:
                read[no] = 1

        total_count = int(sum(read))
        if p_modrate_per_read_copy[total_count] != 0:
            accepted.append(read)
            p_modrate_per_read_copy[total_count] -= 1
            counter += 1
        else:
            rejected += 1

        if rejected >= p_reject_limit:
            break

    return accepted

### <<< ADDED
# https://stackoverflow.com/questions/62199851/how-to-shuffle-a-2d-binary-matrix-preserving-marginal-distributions
def flip1(m):
    """
    Chooses a single (i0, j0) location in the matrix to 'flip'
    Then randomly selects a different (i, j) location that creates
    a quad [(i0, j0), (i0, j), (i, j0), (i, j) in which flipping every
    element leaves the marginal distributions unaltered.  
    Changes those elements, and returns 1.

    If such a quad cannot be completed from the original position, 
    does nothing and returns 0.
    """
    i0 = np.random.randint(m.shape[0])
    j0 = np.random.randint(m.shape[1])

    level = m[i0, j0]
    flip = 0 if level == 1 else 1  # the opposite value

    for i in np.random.permutation(range(m.shape[0])):  # try in random order
        if (i != i0 and  # don't swap with self
            m[i, j0] != level):  # maybe swap with a cell that holds opposite value
            for j in np.random.permutation(range(m.shape[1])):
                if (j != j0 and  # don't swap with self
                    m[i, j] == level and  # check that other swaps work
                    m[i0, j] != level):
                    # make the swaps
                    m[i0, j0] = flip
                    m[i0, j] = level
                    m[i, j0] = level
                    m[i, j] = flip
                    return 1

    return 0

def shuffle_old(m1, n=1000):
    m2 = m1.copy()
    f_success = np.mean([flip1(m2) for _ in range(n)])

    # f_success is the fraction of flip attempts that succeed, for diagnostics
    #print(f_success)

    # check the answer
    assert(all(m1.sum(axis=1) == m2.sum(axis=1)))
    assert(all(m1.sum(axis=0) == m2.sum(axis=0)))

    return m2


@jit(nopython=False, parallel=False)
def shuffle(m, n=1000, p_seed=386):

    np.random.seed(p_seed)

    for _ in range(n):
        i0 = np.random.randint(m.shape[0])
        j0 = np.random.randint(m.shape[1])

        level = m[i0, j0]
        flip = 0 if level == 1 else 1  # the opposite value

        for i in np.random.permutation(np.arange(m.shape[0])):  # try in random order
            if (i != i0 and  # don't swap with self
                m[i, j0] != level):  # maybe swap with a cell that holds opposite value
                for j in np.random.permutation(np.arange(m.shape[1])):
                    if (j != j0 and  # don't swap with self
                        m[i, j] == level and  # check that other swaps work
                        m[i0, j] != level):
                        # make the swaps
                        m[i0, j0] = flip
                        m[i0, j] = level
                        m[i, j0] = level
                        m[i, j] = flip
                        break
                else:
                    continue
                break
        else:
            continue

    return m


def generate_indpt_reads_v2(df, p_limit=1000, p_seed=386):

    # perform rejection sampling
    m = df.to_numpy().copy()
    accepted = shuffle(m, n=p_limit, p_seed=p_seed)

    # convert numpy array to dataframe
    sparse_accepted = sparse.coo_matrix(accepted)
    #df_new = pd.DataFrame(accepted, index=df.index[:accepted.shape[0]], columns=df.columns)
    df_new = pd.DataFrame.sparse.from_spmatrix(sparse_accepted)
    df_new.index = df.index[:accepted.shape[0]]
    df_new.columns = df.columns

    return df_new

def calc_svd_v2(f_pkl, p_genes, dict_sizes, p_start=None, p_end=None, no_of_components=30, p_window=100, p_seed=386, 
                p_step_size=1, p_obs=True):

    
    with open(f_pkl, "rb") as f:
        dict_pkl = pkl.load(f)

    if p_obs:
        df_WT, = dict_pkl["obs"][386]
    else:
        df_WT, = dict_pkl["sim"][p_seed]
    
    #print(X.shape, df_WT.shape, df_MT.shape)
    p_coords = get_start_end_coords_v2(p_genes, dict_sizes, p_start=p_start, p_end=p_end, p_window=p_window, p_step_size=p_step_size)

    data_entropy_svd_wt = []
    
    for start, end in p_coords:
        
        """
        cc = np.corrcoef(df_WT.loc[:,start:end-1].values+1e-10)
        ncc = cc/np.trace(cc)
        """
        #ncc = df_WT.loc[:,start:end-1].values+1e-10

        ncc = df_WT.iloc[:,start:end].values
        if no_of_components == -1:
            U, S, Vh = randomized_svd(ncc, n_components=min(ncc.shape[0], ncc.shape[1]))
            #entropy_svd_wt = spstats.entropy(S)
            entropy_svd_wt = spstats.entropy(S/S.sum())
        else:
            U, S, Vh = randomized_svd(ncc, n_components=no_of_components)
            entropy_svd_wt = spstats.entropy(S/S.sum())
            #model = PCA(n_components=no_of_components, random_state=386)
            #model.fit(ncc)
            #S = model.singular_values_
            #entropy_svd_wt = spstats.entropy(S/S.sum())
            #entropy_svd_wt = spstats.entropy(model.explained_variance_ratio_)
        
        data_entropy_svd_wt.append(entropy_svd_wt)

    return data_entropy_svd_wt

def draw_samples(f_pkl, p_genes, list_of_seeds, p_step_size, dict_sizes, p_first_nth_seeds=100, f_pkl_svd=None, p_start=None, p_end=None, p_no_of_components=30, p_window=100):

    data_entropy_svd_wt_sim = [ calc_svd_v2(f_pkl, p_genes, dict_sizes, p_start=p_start, p_end=p_end,
                                            no_of_components=p_no_of_components, p_window=p_window, 
                                            p_seed=p_seed, p_step_size=p_step_size, 
                                            p_obs=False) for p_seed in list_of_seeds[:p_first_nth_seeds] ]

    data_entropy_svd_wt_obs = [ calc_svd_v2(f_pkl, p_genes, dict_sizes, p_start=p_start, p_end=p_end,
                                            no_of_components=p_no_of_components, p_window=p_window, 
                                            p_seed=p_seed, p_step_size=p_step_size, 
                                            p_obs=True) for p_seed in [386,] ]

    data_entropy_svd_wt_sim = np.array(data_entropy_svd_wt_sim)
    data_entropy_svd_wt_obs = np.array(data_entropy_svd_wt_obs)

    print(data_entropy_svd_wt_sim.shape, data_entropy_svd_wt_obs.shape)

    # PUT THIS BACK IN THE FUTURE!!
    #if f_pkl_svd is not None:

    dict_pkl_svd = {"sim": data_entropy_svd_wt_sim, "obs": data_entropy_svd_wt_obs}
    #f_pkl_svd = "NB_003_draw_samples_permtest_%s_svd_%s.pkl" % (p_genes[0], p_step_size)
    with open(f_pkl_svd, "wb") as f:
        pkl.dump(dict_pkl_svd, f)

    return data_entropy_svd_wt_obs, data_entropy_svd_wt_sim
### >>> ADDED

def get_data(dfs, p_genes, p_length=-1, p_start=None, p_end=None,
             p_min_samples=-1, p_sample="ribosxitch",
             p_pos_modrate_low=0.0, p_pos_modrate_high=1.0,
             p_read_modrate_low=0.0075, p_read_modrate_high=0.025, 
             p_seed=386, p_threshold_len_prop=0.0):
    """ Get dict of data frames from the respective input files. Filter by reads only 

    Args:
        dfs (list of pandas df):
        p_genes (list of str): 
    Return:
        dict_of_dfs (dict of panda dataframes): 
    """

    
    print([ df.shape for df in dfs ])
    print([ df.shape[0] for df in dfs ])
    # Filter data by modrate on reads only
    dfs_flt = [ filter_data_by_modrate(df, 
                                       p_modrate_low=p_read_modrate_low, 
                                       p_modrate_high=p_read_modrate_high, 
                                       p_axis=1) for df in dfs ]
    
    np.random.seed(386) #386, 472, 123, 823

    if p_min_samples == -1:
        min_samples = min([ df.shape[0] for df in dfs_flt ])
    else:
        min_samples = p_min_samples
    print([ df.shape[0] for df in dfs_flt ])
    print(min_samples)
    
    dfs_flt2 = [ df.sample(min_samples, random_state=p_seed) for df in dfs_flt ]
    new_X = pd.concat(dfs_flt2)
    
    dict_of_dfs = {"complete": new_X, "individual": dfs_flt2}

    return dict_of_dfs

def get_simulated_data(orig_dfs, p_genes, p_length=-1, p_start=None, p_end=None,
                       p_min_samples=-1, p_sample="ribosxitch",
                       p_overshoot=5.0, p_seed=386,
                       p_pos_modrate_low=0.0, p_pos_modrate_high=1.0,
                       p_read_modrate_low=0.0075, p_read_modrate_high=0.025,
                       p_reject_limit=5000000):
    
    dict_of_dfs = get_data(orig_dfs, p_genes, p_length=p_length,
                           p_start=p_start, p_end=p_end,
                           p_sample=p_sample,
                           p_min_samples=-1, 
                           p_pos_modrate_low=0.0, p_pos_modrate_high=1.0,
                           p_read_modrate_low=0.0, p_read_modrate_high=1.0,
                           p_seed=p_seed)
        
    new_X = dict_of_dfs["complete"]
    dfs = dict_of_dfs["individual"]
    print(new_X.shape, [ df.shape for df in dfs ])

    #new_X, df_WT, df_MT = get_data(p_genes, p_min_samples=p_min_samples)
    new_X_sim = generate_indpt_reads(new_X, p_reject_limit=p_reject_limit, p_seed=p_seed)
    dfs_sim = [ generate_indpt_reads(df, p_reject_limit=p_reject_limit, p_seed=p_seed) for df in dfs ]

    print("Sim Shape")
    print(new_X_sim.shape, [ df.shape for df in dfs_sim ])

    # Filter data by modrate on reads
    new_X_flt = filter_data_by_modrate(new_X_sim, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    dfs_flt = [ filter_data_by_modrate(df, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1) 
               for df in dfs_sim ]

    print("Sim Filter")
    print(new_X_flt.shape, [ df.shape for df in dfs_flt ])

    # Downsample to specified amount
    if p_min_samples == -1:
        min_samples = min([ df.shape[0] for df in dfs_flt ])
    else:
        min_samples = p_min_samples
    
    new_X_flt2 = new_X_flt.sample(min_samples*(len(dfs_flt)), random_state=386)
    dfs_flt2 = [ df.sample(min_samples, random_state=386) for df in dfs_flt ]

    # Final!
    print("Sim Final")
    print(new_X_flt2.shape, [ df.shape for df in dfs_flt2 ])

    dict_of_dfs_sim = {"complete": new_X_flt2, "individual": dfs_flt2}

    return dict_of_dfs_sim

def run_local_svd(f_pkl, f_pkl_svd, p_genes, p_step_size, dict_sizes, list_of_seeds, p_first_nth_seeds=100, p_window=100):
    
    if os.path.isfile(f_pkl_svd):
        with open(f_pkl_svd, "rb") as f:
            dict_pkl_svd = pkl.load(f)
            data_entropy_svd_wt_vs_mt_obs = dict_pkl_svd["obs"]
            data_entropy_svd_wt_vs_mt_sim = dict_pkl_svd["sim"]
            data_entropy = (data_entropy_svd_wt_vs_mt_obs, data_entropy_svd_wt_vs_mt_sim)
    else:
        data_entropy = draw_samples(f_pkl, p_genes, list_of_seeds, p_step_size, dict_sizes, p_first_nth_seeds=p_first_nth_seeds, f_pkl_svd=f_pkl_svd, p_window=p_window)
        data_entropy_svd_wt_vs_mt_obs, data_entropy_svd_wt_vs_mt_sim = data_entropy
    return data_entropy

def get_simulated_data_v2(new_X, p_length=-1, p_start=None, p_end=None, 
                          p_min_samples=-1, p_overshoot=5.0, p_seed=386,
                          p_pos_modrate_low=0.0, p_pos_modrate_high=1.0,
                          p_read_modrate_low=0.0075, p_read_modrate_high=0.025,
                          p_reject_limit=5000000, p_flip_limit=50000):

    new_X_sim = generate_indpt_reads_v2(new_X, p_limit=p_flip_limit, p_seed=p_seed)

    return new_X_sim


def calc_svd(dict_pkl, p_genes, no_of_components=30, p_window=100, p_seed=386, 
             p_step_size=1, p_obs=True, p_target="complete"):

    if p_obs:
        dict_of_dfs = dict_pkl["data"]["obs"][386]
    else:
        dict_of_dfs = dict_pkl["data"]["sim"][p_seed]

    if p_target == "complete":
        target_X = dict_of_dfs["complete"]
    else:
        target_X = dict_of_dfs["individual"][p_target]

    p_length = target_X.shape[1]
    p_coords = get_start_end_coords(p_length, p_window=p_window, p_step_size=p_step_size)

    data_entropy_svd = []

    for start, end in p_coords:
        U, S, Vh = randomized_svd(target_X.iloc[:,start:end].values, n_components=no_of_components)
        entropy_svd = spstats.entropy(S/S.sum())
        data_entropy_svd.append(entropy_svd)

    return data_entropy_svd


def collapse_into_intervals(i):
    for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1], b[-1][1]

def call_heterogeneous_locations_permtest(data, x_coords, p_step_size, p_pval_threshold=0.05):

    data_entropy_svd_wt_obs, data_entropy_svd_wt_sim = data

    list_of_significant_sites = []

    p_vals = 1 - (np.count_nonzero(data_entropy_svd_wt_obs[0] < data_entropy_svd_wt_sim, axis=0) / len(data_entropy_svd_wt_sim))
    print(p_vals)

    for no, (x, p) in enumerate(zip(x_coords, p_vals)):
        if p < p_pval_threshold:
            list_of_significant_sites.append(x)

    list_of_significant_intervals = list(collapse_into_intervals([ x//p_step_size for x in list_of_significant_sites ]))
    return list_of_significant_sites, list_of_significant_intervals, p_vals

def call_heterogeneous_locations(data, x_coords, p_step_size, 
                                 p_pval_threshold=0.05):

    data_entropy_svd_obs, data_entropy_svd_sim = data

    list_of_significant_sites = []

    datum_entropy_svd_obs_mean = data_entropy_svd_obs[0]
    datum_entropy_svd_sim_mean = data_entropy_svd_sim.mean(axis=0)
    datum_entropy_svd_sim_stdv = data_entropy_svd_sim.std(axis=0)

    p_vals = np.array([ spstats.norm.cdf(obs, mean, stdv) for obs, mean, stdv in zip(datum_entropy_svd_obs_mean, datum_entropy_svd_sim_mean, datum_entropy_svd_sim_stdv) ])

    for no, (x, p) in enumerate(zip(x_coords, p_vals)):
        if p < (p_pval_threshold / len(p_vals)):
            list_of_significant_sites.append(x)

    list_of_significant_intervals = list(collapse_into_intervals([ x//p_step_size for x in list_of_significant_sites ]))
    return list_of_significant_sites, list_of_significant_intervals, p_vals






# Plotting Functions
# Inherited from analysis/localentropy_simulation
# --------------------------------------------------

def plot_heterogeneity(f_pkl, p_genes, data, p_step_size, dict_sizes, sim, p_pval_threshold=0.05, p_start=None, p_end=None, p_label="Combined", p_window=100, p_corr_threshold=0.5):

    data_entropy_svd_wt_obs, data_entropy_svd_wt_sim = data

    p_coords = get_start_end_coords_v2(p_genes, dict_sizes, p_window=p_window, p_step_size=p_step_size, p_start=p_start, p_end=p_end)
    x_coords = get_coords(p_coords)

    with open(f_pkl, "rb") as f:
        dict_pkl = pkl.load(f)

    df_WT, = dict_pkl["obs"][386]

    prop_WT = df_WT.sum(axis=0)/df_WT.shape[0]

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[1, 2, 1]}, dpi=100)
    #ax[0].plot(prop_WT, label=p_label)

    data_entropy_svd_wt_obs_mean = data_entropy_svd_wt_obs[0]
    data_entropy_svd_wt_sim_mean = data_entropy_svd_wt_sim.mean(axis=0)
    data_entropy_svd_wt_sim_std = data_entropy_svd_wt_sim.std(axis=0)

    ax[1].plot(x_coords, data_entropy_svd_wt_obs[0], color="black", label="Entropy (obs)")
    ax[1].plot(x_coords, data_entropy_svd_wt_sim_mean, linestyle="--", color=colors[0], label="Entropy (sim), mean")
    ax[1].fill_between(x_coords, 
                    data_entropy_svd_wt_sim_mean-data_entropy_svd_wt_sim_std,
                    data_entropy_svd_wt_sim_mean+data_entropy_svd_wt_sim_std, 
                    color=colors[0], alpha=0.3, 
                    label="Entropy (sim), +/-std")

    ax[0].set_ylabel("Mod Rate")
    ax[0].set_title("%s" % p_genes[0])
    ax[2].set_xlabel("Position")
    ax[1].set_ylabel("Entropy")
    #ax[0].legend()
    ax[1].legend()

    diff_entropy = data_entropy_svd_wt_sim_mean-data_entropy_svd_wt_obs_mean
    ax[2].plot(x_coords, diff_entropy, color=colors[0], label="Entropy Difference")
    ax[2].set_ylabel("Entropy Diff\n(Sim-Obs)")
    ax[2].legend(loc="upper left")

    list_of_significant_sites, list_of_significant_intervals, p_vals = call_heterogeneous_locations(data, x_coords, p_step_size, p_pval_threshold=p_pval_threshold)
    q_vals = (-1*ma.log10(p_vals)).filled(318)

    """
    for x in list_of_significant_sites:
        ax[1].axvspan(x-p_step_size/2, x+p_step_size/2, color="gold", alpha=0.1, zorder=-1)

    print(list_of_significant_sites)
    """

    for x in list_of_significant_intervals:
        ax[1].axvspan((x[0]-0.5)*p_step_size, (x[1]+0.5)*p_step_size, color="gold", alpha=0.1, zorder=-1)
    print(list_of_significant_intervals)

    ax22 = ax[2].twinx()
    ax22.plot(x_coords, q_vals, color="black", label="Q value: -log10(pval)")
    ax22.set_ylabel("Q value")
    ax22.legend()

    if sim is not None:
        # Add simulation information
        # ----------------------------
        sim_wt = sim.library[sim.truths==0].sum(axis=0)/sim.library[sim.truths==0].shape[0]
        sim_mt = sim.library[sim.truths==1].sum(axis=0)/sim.library[sim.truths==1].shape[0]

        corr_x_coords = []
        corrs = []
        for i in range(0, len(sim_wt)-p_window, p_step_size):
            corr_x_coords.append(i+(p_window//2))
            r, pval = spstats.pearsonr(sim_wt[i:i+p_window], sim_mt[i:i+p_window])
            corrs.append(r)
        corrs = np.array(corrs)

        ax[0].plot(sim_wt, linewidth=0.5)
        ax[0].plot(sim_mt, linewidth=0.5)
        ax2 = ax[0].twinx()
        ax2.plot(corr_x_coords, corrs, color="black")

        print(diff_entropy.shape)
        print(len(corrs))

        if len(list_of_significant_intervals) > 0:

            # Predictions
            # --------------
            fig_auc, ax_auc = plt.subplots(1, 2, figsize=(9, 4), dpi=100, sharex=False, sharey=False)

            # Prediction Method 1
            # --------------------
            predicted = np.ones(diff_entropy.shape[0], dtype=int)
            for x in list_of_significant_intervals:
                predicted[x[0]-((p_window//2)//p_step_size):x[1]-((p_window//2)//p_step_size)+1] = 0

            from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, f1_score, recall_score
            auc = roc_auc_score(predicted, corrs)
            fpr, tpr, thresholds = roc_curve(predicted, corrs)

            best_threshold_index = np.argmax((1-fpr)*tpr)
            best_threshold = thresholds[best_threshold_index]
            best_fpr = fpr[best_threshold_index]
            best_tpr = tpr[best_threshold_index]

            ax2.axhline(best_threshold, linestyle="--", color="black", linewidth=0.5)
            ax2.set_ylabel("Correlation")

            
            ax_auc[0].plot(fpr, tpr, label="AUC: %.3f" % auc)
            ax_auc[0].plot([0, 1], [0, 1], linewidth=1.0, color="black", linestyle="--", label="Random")
            ax_auc[0].scatter(best_fpr, best_tpr, color="red", s=20, label="Best Threshold: %.3f" % best_threshold, zorder=5)
            ax_auc[0].legend()
            ax_auc[0].set_xlabel("FPR")
            ax_auc[0].set_ylabel("TPR")
            

            # Prediction Method 2
            # --------------------
            truths = np.zeros(corrs.shape[0], dtype=int)
            truths[corrs<p_corr_threshold] = 1

            if not (np.all(truths==1) or (np.all(truths==0))):
                auc = roc_auc_score(truths, q_vals)
                fpr, tpr, thresholds = roc_curve(truths, q_vals)

                best_threshold_index = np.argmax((1-fpr)*tpr)
                best_threshold = thresholds[best_threshold_index]
                best_fpr = fpr[best_threshold_index]
                best_tpr = tpr[best_threshold_index]
                
                ax2.axhline(p_corr_threshold, linestyle="--", color="purple", linewidth=0.5)
                ax2.set_ylabel("Correlation")

                ax_auc[1].plot(fpr, tpr, label="AUC: %.3f" % auc)
                ax_auc[1].plot([0, 1], [0, 1], linewidth=1.0, color="black", linestyle="--", label="Random")
                ax_auc[1].scatter(best_fpr, best_tpr, color="red", s=20, label="Best Threshold: %.3f" % best_threshold, zorder=5)
                ax_auc[1].legend()
                ax_auc[1].set_xlabel("FPR")
                ax_auc[1].set_ylabel("TPR")

                ax_auc[0].set_title("Predicted Regions vs Correlation")
                ax_auc[1].set_title("Correlation Cut-off vs Q-value")
            
                acc = accuracy_score(truths, 1-predicted)
                ppv = precision_score(truths, 1-predicted)
                f1 = f1_score(truths, 1-predicted)
                recall = recall_score(truths, 1-predicted)
                

                print("Accuracy: %.3f" % acc)
                print("Recall: %.3f" % recall)
                print("PPV: %.3f" % ppv)
                print("F1: %.3f" % f1)

    return fig, ax

def plot_heterogeneity_with_splash(f_pkl, p_genes, data, p_step_size, dict_sizes, sim, p_pval_threshold=0.05, p_start=None, p_end=None, p_label="Combined", p_window=100, p_corr_threshold=0.5, p_splash=False, p_offset=0):

    data_entropy_svd_wt_obs, data_entropy_svd_wt_sim = data

    p_coords = get_start_end_coords_v2(p_genes, dict_sizes, p_window=p_window, p_step_size=p_step_size, p_start=p_start, p_end=p_end)
    x_coords = get_coords(p_coords)

    with open(f_pkl, "rb") as f:
        dict_pkl = pkl.load(f)

    df_WT, = dict_pkl["obs"][386]

    prop_WT = df_WT.sum(axis=0)/df_WT.shape[0]

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[1, 2, 1]}, dpi=100)
    #ax[0].plot(prop_WT, label=p_label)

    data_entropy_svd_wt_obs_mean = data_entropy_svd_wt_obs[0]
    data_entropy_svd_wt_sim_mean = data_entropy_svd_wt_sim.mean(axis=0)
    data_entropy_svd_wt_sim_std = data_entropy_svd_wt_sim.std(axis=0)

    ax[1].plot(x_coords, data_entropy_svd_wt_obs[0], color="black", label="Entropy (obs)")
    ax[1].plot(x_coords, data_entropy_svd_wt_sim_mean, linestyle="--", color=colors[0], label="Entropy (sim), mean")
    ax[1].fill_between(x_coords, 
                    data_entropy_svd_wt_sim_mean-data_entropy_svd_wt_sim_std,
                    data_entropy_svd_wt_sim_mean+data_entropy_svd_wt_sim_std, 
                    color=colors[0], alpha=0.3, 
                    label="Entropy (sim), +/-std")

    ax[0].set_ylabel("Mod Rate")
    ax[0].set_title("%s" % p_genes[0])
    ax[2].set_xlabel("Position")
    ax[1].set_ylabel("Entropy")
    #ax[0].legend()
    ax[1].legend()

    diff_entropy = data_entropy_svd_wt_sim_mean-data_entropy_svd_wt_obs_mean
    ax[2].plot(x_coords, diff_entropy, color=colors[0], label="Entropy Difference")
    ax[2].set_ylabel("Entropy Diff\n(Sim-Obs)")
    ax[2].legend(loc="upper left")

    list_of_significant_sites, list_of_significant_intervals, p_vals = call_heterogeneous_locations(data, x_coords, p_step_size, p_pval_threshold=p_pval_threshold)
    q_vals = (-1*ma.log10(p_vals)).filled(318)

    """
    for x in list_of_significant_sites:
        ax[1].axvspan(x-p_step_size/2, x+p_step_size/2, color="gold", alpha=0.1, zorder=-1)

    print(list_of_significant_sites)
    """

    for x in list_of_significant_intervals:
        ax[1].axvspan((x[0]-0.5)*p_step_size, (x[1]+0.5)*p_step_size, color="gold", alpha=0.1, zorder=-1)
    print(list_of_significant_intervals)

    """
    ax22 = ax[2].twinx()
    ax22.plot(x_coords, q_vals, color="black", label="Q value: -log10(pval)")
    ax22.set_ylabel("Q value")
    ax22.legend()
    """

    if p_splash:
        from intervaltree import IntervalTree

        tree = IntervalTree()

        f_center = "/home/ubuntu/projects/proj_het/analysis/splash_altstruct/results/pline_paris/ribosxitch_wt/overlap_10/%s_ABB618+ABB619.center" % p_genes[0]

        # Highlight regions with structure
        list_of_coords = []
        list_of_arcs = []
        list_of_counts = []

        counter_altstruct = np.zeros(df_WT.shape[1])

        with open(f_center, "r") as f:
            for iid, line in enumerate(f):
                row = line.strip("\r\n").split("\t")

                s1, e1 = int(row[1]), int(row[2])
                s2, e2 = int(row[4]), int(row[5])

                mid1 = (int(row[1])+int(row[2]))//2
                mid2 = (int(row[4])+int(row[5]))//2
                counts = int(row[12].split("_")[-1])

                if abs(mid1 - mid2) >= 300:
                    continue

                #if (mid1 - p_offset) >= x_coords[0] and (mid2 - p_offset) <= x_coords[-1]:
                if (mid1 - p_offset) >= p_start and (mid2 - p_offset) <= p_end:
                    list_of_coords.append((mid1-p_offset-5, mid1-p_offset+5))
                    list_of_coords.append((mid2-p_offset-5, mid2-p_offset+5))

                    list_of_arcs.append((mid1-p_offset, mid2-p_offset))
                    list_of_counts.append(counts)


                    """
                    if mid1 < mid2:
                        tree[mid1-p_offset:mid2-p_offset] = counts
                    elif mid2 > mid1:
                        tree[mid2-p_offset:mid1-p_offset] = counts
                    """

                    tree[mid1-p_offset-5:mid1-p_offset+5] = (counts, iid)
                    tree[mid2-p_offset-5:mid2-p_offset+5] = (counts, iid)

                    #counter_altstruct[int(mid1-p_offset):int(mid2-p_offset)] += counts #counts
                    #counter_altstruct[s1-p_offset:e1-p_offset] += 1 #1
                    #counter_altstruct[s2-p_offset:e2-p_offset] += 1 #1

                    counter_altstruct[s1-p_offset:e2-p_offset] += 1 #1
                    
                    #counter_altstruct[mid1-p_offset-5:mid1-p_offset+5] += 1 #1
                    #counter_altstruct[mid2-p_offset-5:mid2-p_offset+5] += 1 #1

        #counter_altstruct = get_counter_altstruct()
        p_splash_window = p_window  #100
        moving_sum = []
        entropy_splash = []

        for x in x_coords:

            # moving sum calculation
            moving_sum.append(counter_altstruct[x-p_splash_window//2:x+p_splash_window//2].sum())

            # entropy splash calculation
            results = tree[x-p_splash_window//2:x+p_splash_window//2]
            if len(results) > 0:
                entropy_splash.append(spstats.entropy([ r.data[0] for r in results ]))
            else:
                entropy_splash.append(0.0)


        plot_arcs_from_list(list_of_arcs, p_figax=(fig, ax[0]), list_of_counts=list_of_counts, p_show_count=True)


        ax[0].set_ylabel("AltStructs\n(SPLASH)")
        ax[0].set_title("Heterogeneity of %s" % p_genes[0])
        ax[1].set_ylabel("Entropy")

        ax[-1].set_xlabel("Position")

        """
        ax2t = ax[2].twinx()
        ax2t.plot(x_coords, moving_sum, color=colors[1], label="Moving Sum of AltStructs")
        ax2t.set_ylabel("AltStruct MS")
        ax2t.legend(loc="upper right")

        as_corr, as_pval = spstats.pearsonr(diff_entropy, moving_sum)
        ax[2].set_title("Correlation between delta Entropy vs Density of AltStructs: %.3f" % as_corr)
        """

        ax2t = ax[2].twinx()
        ax2t.plot(x_coords, entropy_splash, color=colors[1], label="Entropy AltStructs")
        ax2t.set_ylabel("AltStruct Entropy")
        ax2t.legend(loc="upper right")

        as_corr, as_pval = spstats.pearsonr(diff_entropy, entropy_splash)
        ax[2].set_title("Correlation between delta Entropy vs Entropy of AltStructs: %.3f" % as_corr)
        print(as_corr, as_pval)

    return fig, ax


def plot_heterogeneity_with_BMM(f_pkl, p_genes, data, p_step_size, dict_sizes, sim, p_pval_threshold=0.05, p_start=None, p_end=None, p_label="Combined", p_window=100, p_corr_threshold=0.5, p_splash=False, p_offset=0):

    data_entropy_svd_wt_obs, data_entropy_svd_wt_sim = data

    p_coords = get_start_end_coords_v2(p_genes, dict_sizes, p_window=p_window, p_step_size=p_step_size, p_start=p_start, p_end=p_end)
    x_coords = get_coords(p_coords)

    with open(f_pkl, "rb") as f:
        dict_pkl = pkl.load(f)

    df_WT, = dict_pkl["obs"][386]

    prop_WT = df_WT.sum(axis=0)/df_WT.shape[0]

    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios':[1, 2, 1, 1]}, dpi=100)
    #ax[0].plot(prop_WT, label=p_label)

    data_entropy_svd_wt_obs_mean = data_entropy_svd_wt_obs[0]
    data_entropy_svd_wt_sim_mean = data_entropy_svd_wt_sim.mean(axis=0)
    data_entropy_svd_wt_sim_std = data_entropy_svd_wt_sim.std(axis=0)

    ax[1].plot(x_coords, data_entropy_svd_wt_obs[0], color="black", label="Entropy (obs)")
    ax[1].plot(x_coords, data_entropy_svd_wt_sim_mean, linestyle="--", color=colors[0], label="Entropy (sim), mean")
    ax[1].fill_between(x_coords, 
                    data_entropy_svd_wt_sim_mean-data_entropy_svd_wt_sim_std,
                    data_entropy_svd_wt_sim_mean+data_entropy_svd_wt_sim_std, 
                    color=colors[0], alpha=0.3, 
                    label="Entropy (sim), +/-std")

    ax[0].set_ylabel("Mod Rate")
    ax[0].set_title("%s" % p_genes[0])
    ax[3].set_xlabel("Position")
    ax[1].set_ylabel("Entropy")
    #ax[0].legend()
    ax[1].legend()

    diff_entropy = data_entropy_svd_wt_sim_mean-data_entropy_svd_wt_obs_mean
    ax[2].plot(x_coords, diff_entropy, color=colors[0], label="Entropy Difference")
    ax[2].set_ylabel("Entropy Diff\n(Sim-Obs)")
    ax[2].legend(loc="upper left")

    list_of_significant_sites, list_of_significant_intervals, p_vals = call_heterogeneous_locations(data, x_coords, p_step_size, p_pval_threshold=p_pval_threshold)
    q_vals = (-1*ma.log10(p_vals)).filled(318)

    """
    for x in list_of_significant_sites:
        ax[1].axvspan(x-p_step_size/2, x+p_step_size/2, color="gold", alpha=0.1, zorder=-1)

    print(list_of_significant_sites)
    """

    for x in list_of_significant_intervals:
        ax[1].axvspan((x[0]-0.5)*p_step_size, (x[1]+0.5)*p_step_size, color="gold", alpha=0.1, zorder=-1)
        ax[3].axvspan((x[0]-0.5)*p_step_size, (x[1]+0.5)*p_step_size, color="gold", alpha=0.1, zorder=-1)
    print(list_of_significant_intervals)

    
    ax22 = ax[2].twinx()
    ax22.plot(x_coords, q_vals, color="black", label="Q value: -log10(pval)")
    ax22.set_ylabel("Q value")
    ax22.legend()

    if sim is not None:
        # Add simulation information
        # ----------------------------
        sim_wt = sim.library[sim.truths==0].sum(axis=0)/sim.library[sim.truths==0].shape[0]
        sim_mt = sim.library[sim.truths==1].sum(axis=0)/sim.library[sim.truths==1].shape[0]

        corr_x_coords = []
        corrs = []
        for i in range(0, len(sim_wt)-p_window, p_step_size):
            corr_x_coords.append(i+(p_window//2))
            r, pval = spstats.pearsonr(sim_wt[i:i+p_window], sim_mt[i:i+p_window])
            corrs.append(r)
        corrs = np.array(corrs)

        ax[0].plot(sim_wt, linewidth=0.5)
        ax[0].plot(sim_mt, linewidth=0.5)
        ax2 = ax[0].twinx()
        ax2.plot(corr_x_coords, corrs, color="black")

        print(diff_entropy.shape)
        print(len(corrs))

        if len(list_of_significant_intervals) > 0:

            # Predictions
            # --------------
            fig_auc, ax_auc = plt.subplots(1, 2, figsize=(9, 4), dpi=100, sharex=False, sharey=False)

            # Prediction Method 1
            # --------------------
            predicted = np.ones(diff_entropy.shape[0], dtype=int)
            for x in list_of_significant_intervals:
                predicted[x[0]-((p_window//2)//p_step_size):x[1]-((p_window//2)//p_step_size)+1] = 0

            from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, f1_score, recall_score
            auc = roc_auc_score(predicted, corrs)
            fpr, tpr, thresholds = roc_curve(predicted, corrs)

            best_threshold_index = np.argmax((1-fpr)*tpr)
            best_threshold = thresholds[best_threshold_index]
            best_fpr = fpr[best_threshold_index]
            best_tpr = tpr[best_threshold_index]

            ax2.axhline(best_threshold, linestyle="--", color="black", linewidth=0.5)
            ax2.set_ylabel("Correlation")

            
            ax_auc[0].plot(fpr, tpr, label="AUC: %.3f" % auc)
            ax_auc[0].plot([0, 1], [0, 1], linewidth=1.0, color="black", linestyle="--", label="Random")
            ax_auc[0].scatter(best_fpr, best_tpr, color="red", s=20, label="Best Threshold: %.3f" % best_threshold, zorder=5)
            ax_auc[0].legend()
            ax_auc[0].set_xlabel("FPR")
            ax_auc[0].set_ylabel("TPR")
            

            # Prediction Method 2
            # --------------------
            truths = np.zeros(corrs.shape[0], dtype=int)
            truths[corrs<p_corr_threshold] = 1

            if not (np.all(truths==1) or (np.all(truths==0))):
                auc = roc_auc_score(truths, q_vals)
                fpr, tpr, thresholds = roc_curve(truths, q_vals)

                best_threshold_index = np.argmax((1-fpr)*tpr)
                best_threshold = thresholds[best_threshold_index]
                best_fpr = fpr[best_threshold_index]
                best_tpr = tpr[best_threshold_index]
                
                ax2.axhline(p_corr_threshold, linestyle="--", color="purple", linewidth=0.5)
                ax2.set_ylabel("Correlation")

                ax_auc[1].plot(fpr, tpr, label="AUC: %.3f" % auc)
                ax_auc[1].plot([0, 1], [0, 1], linewidth=1.0, color="black", linestyle="--", label="Random")
                ax_auc[1].scatter(best_fpr, best_tpr, color="red", s=20, label="Best Threshold: %.3f" % best_threshold, zorder=5)
                ax_auc[1].legend()
                ax_auc[1].set_xlabel("FPR")
                ax_auc[1].set_ylabel("TPR")

                ax_auc[0].set_title("Predicted Regions vs Correlation")
                ax_auc[1].set_title("Correlation Cut-off vs Q-value")
            
                acc = accuracy_score(truths, 1-predicted)
                ppv = precision_score(truths, 1-predicted)
                f1 = f1_score(truths, 1-predicted)
                recall = recall_score(truths, 1-predicted)
                

                print("Accuracy: %.3f" % acc)
                print("Recall: %.3f" % recall)
                print("PPV: %.3f" % ppv)
                print("F1: %.3f" % f1)

    return fig, ax, list_of_significant_sites