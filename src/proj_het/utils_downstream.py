import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spstats
import pandas as pd
import umap
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from matplotlib import cm
from matplotlib.colors import rgb2hex
from collections import Counter
from sklearn.metrics import accuracy_score, rand_score, adjusted_rand_score
from pyfaidx import Fasta

from .utils_io import load_data, filter_data_by_modrate
from .utils_analysis import run_BMM_v2
from .utils_suboptimal_structure import get_structs, parse_struct_into_contactmap, parse_structs_into_vector, find_centroid

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def projection_of_candidate_matrix_into_orignal_matrix(new_X, results, params, p_unknown=0.0):

    (X_orig, df_orig_WT, df_orig_MT, p_gene, p_start, p_end, p_local) = params

    # definition of candidate matrices: 
    #   - input data with plausibly reduced number of positions or reduced number of reads 

    # split candidate matrix into candidate matrix (WT) and candidate matrix (MT)
    new_X1 = new_X[new_X.index.isin(df_orig_WT.index)]   
    new_X2 = new_X[new_X.index.isin(df_orig_MT.index)]

    # project candaidate matrices into the original matrices
    new_X1 = X_orig[X_orig.index.isin(new_X1.index)]
    new_X2 = X_orig[X_orig.index.isin(new_X2.index)]

    return new_X1, new_X2

def run_2sample_clustering_analysis(f_mtxs, f_ids, p_genes, f_sizes, 
                                    p_depth=-1, p_length=-1, p_start=None, p_end=None, 
                                    p_threshold_len_prop=0.85, p_verbose=True, 
                                    f_likelihood=None, p_clusters=2, p_no_of_runs=1,
                                    p_threshold=0.5, #0.0001,
                                    p_min_iters=20, #300 #300,
                                    p_seed=386,
                                    p_min_samples=-1,
                                    p_unknown=0.00, 
                                    p_pos_modrate_low=0.01, p_pos_modrate_high=1.0,
                                    p_read_modrate_low=0.0075, p_read_modrate_high=0.025, 
                                    p_visualize=True, p_impute=False, p_max_iters=100, p_local=False,
                                    p_prop=None):



    X, (df_WT, df_MT) = load_data(f_mtxs, f_ids, p_genes, f_sizes=f_sizes, 
                                  p_depth=p_depth, p_length=p_length, 
                                  p_start=p_start, p_end=p_end, 
                                  p_threshold_len_prop=p_threshold_len_prop, 
                                  p_verbose=p_verbose)
    print(X.shape)
    print(df_WT.shape)
    print(df_MT.shape)
    # Plot original

    if p_local:

        X_full = X
        df_WT_full = df_WT
        df_MT_full = df_MT

        print(X_full.shape)
        print(df_WT_full.shape)
        print(df_MT_full.shape)

    else:

        X_full, (df_WT_full, df_MT_full) = load_data(f_mtxs, f_ids, p_genes, f_sizes=f_sizes, 
                                                    p_depth=p_depth, p_length=-1, 
                                                    p_start=None, p_end=None, 
                                                    p_threshold_len_prop=p_threshold_len_prop, 
                                                    p_verbose=p_verbose)
        
        print(X_full.shape)
        print(df_WT_full.shape)
        print(df_MT_full.shape)

    
    modrate_WT = (df_WT.sum().sum()*100)/(df_WT.shape[0]*df_WT.shape[1])
    modrate_MT = (df_MT.sum().sum()*100)/(df_MT.shape[0]*df_MT.shape[1])

    prop_WT = df_WT.sum(axis=0)/df_WT.shape[0]
    prop_MT = df_MT.sum(axis=0)/df_MT.shape[0]

    if p_visualize:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2), sharex=True, sharey=False, dpi=100)

        ax.plot(prop_WT, label="WT (n=%s, %.2f%%)" % (df_WT.shape[0], modrate_WT), linewidth=1.0)
        ax.plot(prop_MT, label="MT (n=%s, %.2f%%)" % (df_MT.shape[0], modrate_MT), linewidth=1.0)

        ax.legend()

    """
    # 1st option
    # Filter data by modrate on positions and reads
    df_WT_flt = filter_data_by_modrate(df_WT, p_modrate_low=p_pos_modrate_low, p_modrate_high=p_pos_modrate_high, p_axis=0)
    df_MT_flt = filter_data_by_modrate(df_MT, p_modrate_low=p_pos_modrate_low, p_modrate_high=p_pos_modrate_high, p_axis=0)
    
    df_WT_flt = filter_data_by_modrate(df_WT_flt, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    df_MT_flt = filter_data_by_modrate(df_MT_flt, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    
    print(df_WT_flt.shape, df_MT_flt.shape)

    # Subset positions to smaller data size
    index_retained = np.union1d(np.where(df_WT_flt.sum(axis=0)!=0)[0], np.where(df_MT_flt.sum(axis=0)!=0)[0])
    print(np.where(df_WT_flt.sum(axis=0)!=0)[0].shape)
    print(np.where(df_MT_flt.sum(axis=0)!=0)[0].shape)
    print(index_retained.shape)

    df_WT_flt = df_WT_flt.iloc[:,index_retained]
    df_MT_flt = df_MT_flt.iloc[:,index_retained]
    """

    """
    # 2nd option
    # Filter data by modrate on positions and reads
    df_WT_copy = df_WT.copy()
    df_MT_copy = df_MT.copy()

    df_WT_copy = filter_data_by_modrate(df_WT_copy, p_modrate_low=p_pos_modrate_low, p_modrate_high=p_pos_modrate_high, p_axis=0)
    df_MT_copy = filter_data_by_modrate(df_MT_copy, p_modrate_low=p_pos_modrate_low, p_modrate_high=p_pos_modrate_high, p_axis=0)

    print(df_WT_copy.shape, df_MT_copy.shape)

    print(np.where(df_WT_copy.sum(axis=0)!=0)[0].shape)
    print(np.where(df_MT_copy.sum(axis=0)!=0)[0].shape)


    # Subset positions to smaller data size
    index_retained = np.union1d(np.where(df_WT_copy.sum(axis=0)!=0)[0], np.where(df_MT_copy.sum(axis=0)!=0)[0])

    print(index_retained.shape)

    mask = np.ones(df_WT_copy.shape[1], dtype=bool)
    mask[index_retained] = False
    index_skipped = np.arange(df_WT_copy.shape[1])[mask]

    print(df_WT.shape, df_MT.shape)
    df_WT.iloc[:,index_skipped] = 0
    df_MT.iloc[:,index_skipped] = 0

    df_WT_flt = filter_data_by_modrate(df_WT, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    df_MT_flt = filter_data_by_modrate(df_MT, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)

    df_WT_flt = df_WT_flt.iloc[:,index_retained]
    df_MT_flt = df_MT_flt.iloc[:,index_retained]

    print(df_WT.shape, df_MT.shape)
    """

    """
    # 3rd Option
    # Independent row rate and column rate
    # Filter data by modrate on positions and reads
    df_WT_flt = filter_data_by_modrate(df_WT, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    df_MT_flt = filter_data_by_modrate(df_MT, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    
    df_WT_flt3 = filter_data_by_modrate(df_WT, p_modrate_low=p_pos_modrate_low, p_modrate_high=p_pos_modrate_high, p_axis=0)
    df_MT_flt3 = filter_data_by_modrate(df_MT, p_modrate_low=p_pos_modrate_low, p_modrate_high=p_pos_modrate_high, p_axis=0)

    print(df_WT_flt.shape, df_MT_flt.shape)

    # Subset positions to smaller data size
    index_retained = np.union1d(np.where(df_WT_flt3.sum(axis=0)!=0)[0], np.where(df_MT_flt3.sum(axis=0)!=0)[0])

    df_WT_flt = df_WT_flt.iloc[:,index_retained]
    df_MT_flt = df_MT_flt.iloc[:,index_retained]
    """

    
    # 3.5th Option
    # Independent row rate and column rate
    # Filter data by modrate on positions and reads
    df_WT_flt = filter_data_by_modrate(df_WT, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    df_MT_flt = filter_data_by_modrate(df_MT, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    
    df_WT_MT = pd.concat([df_WT, df_MT])
    df_WT_MT_flt = filter_data_by_modrate(df_WT_MT, p_modrate_low=p_pos_modrate_low, p_modrate_high=p_pos_modrate_high, p_axis=0)

    # Subset positions to smaller data size
    index_retained = np.where(df_WT_MT_flt.sum(axis=0)!=0)[0]
    df_WT_flt = df_WT_flt.iloc[:,index_retained]
    df_MT_flt = df_MT_flt.iloc[:,index_retained]

    print(df_WT_flt.shape, df_MT_flt.shape)
    

    """
    # 4th Option
    # Filter data by modrate on positions and reads
    df_WT_MT = pd.concat([df_WT, df_MT])
    df_WT_MT_flt = filter_data_by_modrate(df_WT_MT, p_modrate_low=p_pos_modrate_low, p_modrate_high=p_pos_modrate_high, p_axis=0)

    index_retained = np.where(df_WT_MT_flt.sum(axis=0)!=0)[0]
    df_WT_flt = df_WT.iloc[:,index_retained]
    df_MT_flt = df_MT.iloc[:,index_retained]

    #print(index_retained)
    print(df_WT_flt.shape, df_MT_flt.shape)
    df_WT_flt = filter_data_by_modrate(df_WT_flt, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    df_MT_flt = filter_data_by_modrate(df_MT_flt, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    """

    """
    # 5th Option
    # Filter data by modrate on positions and reads
    df_WT_MT = pd.concat([df_WT, df_MT])
    df_WT_MT_flt = filter_data_by_modrate(df_WT_MT, p_modrate_low=p_pos_modrate_low, p_modrate_high=p_pos_modrate_high, p_axis=0)

    index_retained = np.where(df_WT_MT_flt.sum(axis=0)!=0)[0]
    index_skipped = np.where(df_WT_MT_flt.sum(axis=0)==0)[0]

    print(df_WT.sum().sum())
    df_WT.iloc[:,index_skipped] = 0
    df_MT.iloc[:,index_skipped] = 0
    print(df_WT.sum().sum())

    df_WT_flt = filter_data_by_modrate(df_WT, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    df_MT_flt = filter_data_by_modrate(df_MT, p_modrate_low=p_read_modrate_low, p_modrate_high=p_read_modrate_high, p_axis=1)
    
    
    df_WT_flt = df_WT_flt.iloc[:,index_retained]
    df_MT_flt = df_MT_flt.iloc[:,index_retained]

    print(df_WT_flt.shape, df_MT_flt.shape)
    """

    # Show routine
    modrate_WT_flt = (df_WT_flt.sum().sum()*100)/(df_WT_flt.shape[0]*df_WT_flt.shape[1])
    modrate_MT_flt = (df_MT_flt.sum().sum()*100)/(df_MT_flt.shape[0]*df_MT_flt.shape[1])

    prop_WT_flt = df_WT_flt.sum(axis=0)/df_WT_flt.shape[0]
    prop_MT_flt = df_MT_flt.sum(axis=0)/df_MT_flt.shape[0]

    if p_visualize:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2), sharex=True, sharey=False, dpi=100)

        ax.plot(prop_WT_flt, label="WT (n=%s, %.2f%%)" % (df_WT_flt.shape[0], modrate_WT_flt), linewidth=1.0)
        ax.plot(prop_MT_flt, label="MT (n=%s, %.2f%%)" % (df_MT_flt.shape[0], modrate_MT_flt), linewidth=1.0)

        ax.legend()

    print(df_WT_flt.shape, df_MT_flt.shape)


    # Set parameters
    np.random.seed(386) #386, 472, 123, 823

    if p_min_samples == -1:
        min_samples = min(df_WT_flt.shape[0], df_MT_flt.shape[0])
    else:
        min_samples = p_min_samples

    if p_prop is None:
        p_min_samples_WT = min_samples
        p_min_samples_MT = min_samples
    else:
        p_min_samples_all = min_samples * 2
        p_min_samples_WT = int(p_min_samples_all * p_prop)
        p_min_samples_MT = p_min_samples_all - p_min_samples_WT

    df_WT_flt2 = df_WT_flt.sample(p_min_samples_WT, random_state=386)
    df_MT_flt2 = df_MT_flt.sample(p_min_samples_MT, random_state=386)

    print(df_WT_flt2.shape, df_MT_flt2.shape)

    new_X = pd.concat([df_WT_flt2, df_MT_flt2])

    # Prepare extra parameters for downstream analysis
    #params = [X_full, df_WT_full, df_MT_full, p_genes[0], p_start, p_end, p_local]
    params = [pd.concat([df_WT_full, df_MT_full]), df_WT_full, df_MT_full, p_genes[0], p_start, p_end, p_local]
    #params = [X, df_WT, df_MT, p_genes[0], p_start, p_end]

    # Run BMM analysis
    model, results = run_BMM_v2(new_X, p_clusters, params, p_seed=p_seed, p_unknown=p_unknown, p_no_of_runs=p_no_of_runs, p_threshold=p_threshold, p_min_iters=p_min_iters, f_likelihood=f_likelihood, p_visualize=p_visualize, p_impute=p_impute, p_max_iters=p_max_iters)

    return model, results, params, new_X

def get_flip_order(new_X, results, params, p_unknown=0.0):

    (X_orig, df_orig_WT, df_orig_MT, p_gene, p_start, p_end, p_local) = params

    new_X1, new_X2 = projection_of_candidate_matrix_into_orignal_matrix(new_X, results, params, p_unknown=p_unknown)
    total_reads = new_X1.shape[0] + new_X2.shape[0]
    baseline = [new_X1.shape[0]/total_reads, new_X2.shape[0]/total_reads]

    flip_order = []  # 0: WT, 1: MT
    for i, r in enumerate(results):

        r = np.array(r, dtype=int)

        # Project back to original matrix
        X_pred = X_orig[X_orig.index.isin(new_X.iloc[r,:].index)]

        # Selected Best PPV
        # Calculate Precision
        c1 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_WT.index))
        c2 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_MT.index))

        # Calculate PPV
        ppv = [(c1/(c1+c2)), (c2/(c1+c2))]

        best_ppv = np.argmax(np.array(ppv) - np.array(baseline))
        flip_order.append(best_ppv)

    return flip_order


def run_visualization(new_X, results, params, p_unknown=0.05):
    pass


def plot_inferred_proportions(new_X, results, params, flip_order, p_unknown=0.0):

    (X_orig, df_orig_WT, df_orig_MT, p_gene, p_start, p_end, p_local) = params

    new_X1, new_X2 = projection_of_candidate_matrix_into_orignal_matrix(new_X, results, params, p_unknown=p_unknown)
    total_reads = new_X1.shape[0] + new_X2.shape[0]
    baseline = [new_X1.shape[0]/total_reads, new_X2.shape[0]/total_reads]
    no_of_clusters = len(results)

    # for each predicted module
    #ppvs_prop = [ [] for _ in range(2) ]
    #ppvs_vals = [ [] for _ in range(2) ]

    ppvs_prop = np.zeros((2, 2))
    ppvs_vals = np.zeros((2, 2)) #[ [0.0, 0.0] for _ in range(2) ]

    for i, (r, order) in enumerate(zip(results, flip_order)):

        # convert irregular object-typed array
        r = np.array(r, dtype=int)

        
        # Selected Best PPV
        # Calculate Precision
        c1 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_WT.index))
        c2 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_MT.index))

        # Calculate PPV
        ppv = [(c1/(c1+c2)), (c2/(c1+c2))]
        ppv_val = [c1, c2]

        #best_ppv = np.argmax(np.array(ppv) - np.array(baseline))

        """
        ppvs_prop[order].append(ppv[order]*100)
        ppvs_prop[1-order].append(ppv[1-order]*100)
        ppvs_vals[order].append(ppv_val[order])
        ppvs_vals[1-order].append(ppv_val[1-order])
        """
        
        ppvs_prop[order,order] = ppv[order]*100
        ppvs_prop[1-order,order] = ppv[1-order]*100
        ppvs_vals[order,order] = ppv_val[order]
        ppvs_vals[1-order,order] = ppv_val[1-order]

    ppvs_prop = np.array(ppvs_prop)
    ppvs_vals = np.array(ppvs_vals)

    print(ppvs_prop)
    # Plot Barplot
    # -------------
    fig, ax = plt.subplots(figsize=(6, 1.5), dpi=100)
    labels = ["Cluster %s\n(n=%s, %.1f%%)" % (i, len(results[i]), (len(results[i])/total_reads)*100) for i in range(no_of_clusters) ]
    ax.barh(labels[::-1], ppvs_prop[0][::-1], height=0.9, label="WT")
    ax.barh(labels[::-1], ppvs_prop[1][::-1], height=0.9, left=ppvs_prop[0][::-1], label="MT")

    print(ppvs_prop.T[::-1])

    for no, ppv in enumerate(ppvs_prop.T[::-1]):
        for p, x in zip(ppv, np.cumsum(ppv)):
            print(p, x, no+0.5)
            ax.annotate("%.1f%%" % p, xy=(x, no), xytext=(-40,0), 
                        textcoords="offset points", ha="left", va="center")


        #ax.annotate("%.2f%%" % p.get_x(), xy=(-2.5, p.get_y()+p.get_height()/2),
        #    xytext=(0, 0), textcoords='offset points', ha="left", va="center")

    ax.set_xlabel("Percentage (%)")
    #ax.set_ylim(-0.5, 2.5)
    #ax.legend(loc="upper right")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05))
    ax.set_title("%s\nPercentage of WT/MT reads per Predicted Cluster" % p_gene)
    ax.set_xlim(0, 100)
    

    return ppvs_prop, ppvs_vals

def plot_predicted_cluster_profile(new_X, results, params, p_unknown=0.0):

    (X_orig, df_orig_WT, df_orig_MT, p_gene, p_start, p_end, p_local) = params

    new_X1, new_X2 = projection_of_candidate_matrix_into_orignal_matrix(new_X, results, params, p_unknown=p_unknown)
    total_reads = new_X1.shape[0] + new_X2.shape[0]
    baseline = [new_X1.shape[0]/total_reads, new_X2.shape[0]/total_reads]
    no_of_clusters = len(results)

    dict_assignments = {"WT": [], "MT": [], "UN": []}
    for i, r in enumerate(results):

        r = np.array(r, dtype=int)

        # Project back to original matrix
        X_pred = X_orig[X_orig.index.isin(new_X.iloc[r,:].index)]

        # Selected Best PPV
        # Calculate Precision
        c1 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_WT.index))
        c2 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_MT.index))

        # Calculate PPV
        ppv = [(c1/(c1+c2)), (c2/(c1+c2))]

        best_ppv = np.argmax(np.array(ppv) - np.array(baseline))
        color = colors[best_ppv]
        if ppv[best_ppv] < baseline[best_ppv] + p_unknown:
            color = "black"

        if ppv[best_ppv] < baseline[best_ppv] + p_unknown:
            dict_assignments["UN"].append(i)
        else:
            if best_ppv == 0:
                dict_assignments["WT"].append(i)
            elif best_ppv == 1:
                dict_assignments["MT"].append(i)


    total = sum([ len(r) for r in results ])
    print(total)
    
    def get_results(results, dict_assignments, category):
        R = results[dict_assignments[category]]
        if len(R) == 0:
            return np.array([])
        else:
            return np.concatenate(R)


    WT = get_results(results, dict_assignments, "WT")
    MT = get_results(results, dict_assignments, "MT")
    UN = get_results(results, dict_assignments, "UN")

    print("WT: %s | MT: %s | UN: %s" % (len(WT), len(MT), len(UN)))
    
    np.random.seed(386)
    seq_UN = np.arange(len(UN))
    UN_WT_random = np.random.choice(seq_UN, size=int(baseline[0]*len(seq_UN)), replace=False)
    UN_WT = UN[UN_WT_random]
    UN_MT = UN[np.setdiff1d(seq_UN, UN_WT_random)]

    print("UN WT: %s | UN MT: %s"% (len(UN_WT), len(UN_MT)))

    WT = np.concatenate([WT, UN_WT])
    MT = np.concatenate([MT, UN_MT])

    X_pred_WT = X_orig[X_orig.index.isin(new_X.iloc[WT,:].index)]
    X_pred_MT = X_orig[X_orig.index.isin(new_X.iloc[MT,:].index)]

    # Correlation
    corr_WT, pval_WT = spstats.pearsonr(X_pred_WT.sum(axis=0)/X_pred_WT.shape[0], new_X1.sum(axis=0)/new_X1.shape[0])
    corr_MT, pval_MT = spstats.pearsonr(X_pred_MT.sum(axis=0)/X_pred_MT.shape[0], new_X2.sum(axis=0)/new_X2.shape[0])

    fig, ax = plt.subplots(2, 1, figsize=(13, 4+1), sharex=True, sharey=True)

    ax[0].plot(range(new_X1.shape[1]), new_X1.sum(axis=0)/new_X1.shape[0], 
            label="WT (n=%s, %.1f%%)" % (new_X1.shape[0], baseline[0]*100))
    ax[0].plot(range(new_X2.shape[1]), new_X2.sum(axis=0)/new_X2.shape[0], 
            label="MT (n=%s, %.1f%%)" % (new_X2.shape[0], baseline[1]*100))

    ax[0].legend()
    ax[0].set_title("Observed", fontsize=14)

    #total = X_pred_WT.shape[0] + X_pred_MT.shape[0]
    label_WT = "Cluster 0 (n=%s, %.1f%%) | Corr: %.3f" % (X_pred_WT.shape[0], (X_pred_WT.shape[0]/total)*100, corr_WT)
    label_MT = "Cluster 1 (n=%s, %.1f%%) | Corr: %.3f" % (X_pred_MT.shape[0], (X_pred_MT.shape[0]/total)*100, corr_MT)

    
    counter_WT = Counter(X_pred_WT.index.isin(new_X1.index))
    counter_MT = Counter(X_pred_MT.index.isin(new_X2.index))
    print(counter_WT)
    print(counter_MT)
    
    print((counter_WT[True] + counter_MT[True])/total)
    ACC = (counter_WT[True] + counter_MT[True])/total

    ax[1].plot(range(X_pred_WT.shape[1]), X_pred_WT.sum(axis=0)/X_pred_WT.shape[0], label=label_WT)
    ax[1].plot(range(X_pred_WT.shape[1]), X_pred_MT.sum(axis=0)/X_pred_MT.shape[0], label=label_MT)
    ax[1].legend()

    ax[0].set_ylabel("Mod Rate", fontsize=14)
    ax[1].set_xlabel("Position", fontsize=14), ax[1].set_ylabel("Mod Rate", fontsize=14)

    for i in range(2):
        ax[i].tick_params(axis='both', which='major', labelsize=12)

    predictions = np.zeros(sum([ len(r) for r in results ]), dtype=int)

    truths = np.zeros(new_X.shape[0], dtype=int)
    truths[new_X.index.isin(df_orig_WT.index)] = 0
    truths[new_X.index.isin(df_orig_MT.index)] = 1
    truths = truths[sorted(np.concatenate([ list(r) for r in results]))]

    for i, r in enumerate(results):
        r = np.array(r, dtype=int)
        predictions[r] = i

    acc = accuracy_score(truths, predictions)
    if acc < 1-acc:
        acc = accuracy_score(truths, 1-predictions)
        rand_index = rand_score(truths, 1-predictions)
        ari = adjusted_rand_score(truths, 1-predictions)
    else:
        acc = accuracy_score(truths, predictions)
        rand_index = rand_score(truths, predictions)
        ari = adjusted_rand_score(truths, predictions)
    
    ax[1].set_title("Predicted | ACC: %.3f | Rand Index: %.3f | ARI: %.3f" % (ACC, rand_index, ari), fontsize=14)

    return ACC, rand_index, ari, corr_WT, corr_MT



def plot_condition_stratified_cluster_profile(new_X, results, params, p_unknown=0.05):

    (X_orig, df_orig_WT, df_orig_MT, p_gene, p_start, p_end, p_local) = params

    new_X1, new_X2 = projection_of_candidate_matrix_into_orignal_matrix(new_X, results, params, p_unknown=0.05)
    total_reads = new_X1.shape[0] + new_X2.shape[0]
    baseline = [new_X1.shape[0]/total_reads, new_X2.shape[0]/total_reads]
    no_of_clusters = len(results)

    dict_assignments = {"WT": [], "MT": [], "UN": []}
    for i, r in enumerate(results):

        # convert irregular object-typed array
        r = np.array(r, dtype=int)

        # Project back to original matrix
        X_pred = X_orig[X_orig.index.isin(new_X.iloc[r,:].index)]

        # Selected Best PPV
        # Calculate Precision
        c1 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_WT.index))
        c2 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_MT.index))

        # Calculate PPV
        ppv = [(c1/(c1+c2)), (c2/(c1+c2))]
        ppv_val = [c1, c2]

        best_ppv = np.argmax(np.array(ppv) - np.array(baseline))
        
        if ppv[best_ppv] < baseline[best_ppv] + p_unknown:
            color = "black"

        if ppv[best_ppv] < baseline[best_ppv] + p_unknown:
            dict_assignments["UN"].append(i)
        else:
            if best_ppv == 0:
                dict_assignments["WT"].append(i)
            elif best_ppv == 1:
                dict_assignments["MT"].append(i)


        # Calculate Correlation
        obs = [new_X1.sum(axis=0)/new_X1.shape[0], new_X2.sum(axis=0)/new_X2.shape[0]]
        pre = X_pred.sum(axis=0)/X_pred.shape[0]
        corr, pval = spstats.pearsonr(obs[best_ppv], pre)

        # Prepare Labels
        labels = ["WT", "MT"]
        label_ppv = "PPV to %s: %.3f" % (labels[best_ppv], ppv[best_ppv])
        label_corr = "Corr: %.3f" % (corr)
        label = "%s | %s" % (label_ppv, label_corr)


def predict_secondary_structures(new_X, results, params, flip_order, p_unknown=0.05,
                                 f_fasta="/mnt/projects/chengyza/projects/proj_het/data/reference/reference_of_ribosxitch_v2.fa",
                                 p_ss_mod_threshold=0.01, p_temp=25.0, p_slope=2.6, p_intercept=-0.8):

    (X_orig, df_orig_WT, df_orig_MT, p_gene, p_start, p_end, p_local) = params

    # Plot figure
    # -------------
    new_X1 = new_X[new_X.index.isin(df_orig_WT.index)]
    new_X2 = new_X[new_X.index.isin(df_orig_MT.index)]
    new_X1 = X_orig[X_orig.index.isin(new_X1.index)]
    new_X2 = X_orig[X_orig.index.isin(new_X2.index)]

    total_reads = new_X1.shape[0] + new_X2.shape[0]
    baseline = [new_X1.shape[0]/total_reads, new_X2.shape[0]/total_reads]

    no_of_clusters = len(results)


    #ppvs_prop = [ [] for _ in range(2) ]
    #ppvs_vals = [ [] for _ in range(2) ]
    ppvs_vals_wtmt = [ [] for _ in range(2) ]

    ppvs_prop = []
    ppvs_vals = []
    candidates = []
    for i, (r, o) in enumerate(zip(results, flip_order)):

        r = np.array(r, dtype=int)

        # Project back to original matrix
        X_pred = X_orig[X_orig.index.isin(new_X.iloc[r,:].index)]

        # Selected Best PPV
        # Calculate Precision
        c1 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_WT.index))
        c2 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_MT.index))

        #X_pred_wt = X_pred[X_pred.index.isin(new_X1.index)]
        #X_pred_mt = X_pred[X_pred.index.isin(new_X2.index)]

        #candidates.append([X_pred_wt.sum(axis=0)/X_pred_wt.shape[0], 
        #                   X_pred_mt.sum(axis=0)/X_pred_mt.shape[0]])

        candidates.append(X_pred.sum(axis=0)/X_pred.shape[0])

        # Calculate PPV
        ppv = [(c1/(c1+c2)), (c2/(c1+c2))]
        ppv_val = [c1, c2]

        #ppvs_prop[0].append(ppv[0]*100)
        #ppvs_prop[1].append(ppv[1]*100)
        ppvs_vals_wtmt[o].append(ppv_val[o])
        ppvs_vals_wtmt[1-o].append(ppv_val[1-o])

        ppvs_prop.append((ppv[o]+ppv[1-o])*100)
        ppvs_vals.append(ppv_val[o]+ppv_val[1-o])

    # Convert modrate into reactivities
    tmp = []
    X_props = []
    for order in np.argsort(flip_order):
        #candidate_wt, candidate_mt = candidate
        #tmp.append(candidate_wt)
        #tmp.append(candidate_mt)
        tmp.append(candidates[order])
        X_props.append(candidates[order].values)

    tmp1 = pd.concat(tmp)
    norm_factor = max(spstats.iqr(tmp1) * 1.5, np.percentile(tmp1, 90))
    print(norm_factor)

    #step1 = tmp1[tmp1<=spstats.iqr(tmp1) * 1.5]
    #norm_factor = step1[step1>=np.percentile(step1, 90)].mean()
    #print(norm_factor)

    fig, ax = plt.subplots(len(tmp), 1, figsize=(16, 2 * len(tmp)))


    X_norms = []
    for i, X_modrate in enumerate(tmp):

        X_norm = X_modrate/norm_factor
        print(X_norm.shape)
        ax[i].plot(range(X_norm.shape[0]), X_norm)
        X_norms.append(X_norm)

    # Get sequence
    o_fasta = Fasta(f_fasta)

    """
    if p_start is not None and p_end is not None:
        seq = str(o_fasta[p_gene][p_start:p_end])
    else:
        seq = str(o_fasta[p_gene])
    """

    if p_local:
        seq = str(o_fasta[p_gene][p_start:p_end])
    else:
        seq = str(o_fasta[p_gene])
    
    print(len(seq))

    total_structures = np.array(ppvs_vals).T.flatten()
    total_structures_cumsum = np.concatenate([[0], total_structures.cumsum()])
    print(total_structures)
    print(total_structures_cumsum)

    struct_ens = []
    for no_of_structs, X_norm, X_prop in zip(total_structures, X_norms, X_props):

        X_norm[X_prop<p_ss_mod_threshold] = -999
        #no_of_structs = 10000
        #struct = get_structs(seq[:-1], constraints=None, SHAPE=X_norm.values[1:], p_shape_type="deigan", temp=p_temp, subopt=True, no_of_folds=no_of_structs)
        #struct = get_structs(seq[:-3], constraints=None, SHAPE=X_norm.values[3:], p_shape_type="deigan", temp=p_temp, subopt=True, no_of_folds=no_of_structs)
        struct = get_structs(seq, constraints=None, SHAPE=X_norm.values, p_shape_type="deigan", temp=p_temp, subopt=True, no_of_folds=no_of_structs, p_slope=p_slope, p_intercept=p_intercept)
        #struct = get_structs(seq, constraints=None, SHAPE=X_norm.values, p_shape_type="zarringhalam", temp=37.0, subopt=True, no_of_folds=no_of_structs)
        struct_ens.append(struct)

    struct_mfe = []
    for X_norm in X_norms:
        struct = get_structs(seq[:-1], constraints=None, SHAPE=X_norm.values[1:], p_shape_type="deigan", temp=p_temp, subopt=False, no_of_folds=1, p_slope=p_slope, p_intercept=p_intercept)
        #struct = get_structs(seq, constraints=None, SHAPE=X_norm.values, p_shape_type="deigan", temp=25.0, subopt=False, no_of_folds=250)
        struct_mfe.append(struct)

    return seq, struct_ens, struct_mfe, ppvs_vals_wtmt


def plot_structural_ensemble_visualization(struct_ens, no_of_clusters, p_viz_method="pca"):

    total_structures = np.array([ len(s) for s in struct_ens ])
    total_structures_cumsum = np.concatenate([[0], total_structures.cumsum()])

    vectors_sg = []
    for struct in struct_ens:
        vector_sg = np.array(parse_structs_into_vector(struct, p_encode="single"))
        vectors_sg.append(vector_sg)

    if p_viz_method == "umap":
        #embeddings = umap.UMAP(n_components=2, n_neighbors=10, metric="kulsinski", random_state=386).fit_transform(np.concatenate(vectors_sg))
        embeddings = umap.UMAP(n_components=2, metric="manhattan", random_state=386).fit_transform(np.concatenate(vectors_sg))
    elif p_viz_method == "mds":
        embeddings = MDS(n_components=2).fit_transform(np.concatenate(vectors_sg))
    elif p_viz_method == "pca":
        model = PCA(n_components=2)
        embeddings = model.fit_transform(np.concatenate(vectors_sg))


    cmap_blue = cm.get_cmap("Blues", no_of_clusters+2)
    cmap_blue_hex = [ rgb2hex(cmap_blue(i)) for i in range(2, cmap_blue.N) ]

    cmap_orange = cm.get_cmap("Oranges", no_of_clusters+2)
    cmap_orange_hex = [ rgb2hex(cmap_orange(i)) for i in range(2, cmap_orange.N) ]

    new_colors = [cmap_blue_hex, cmap_orange_hex]

    markers = ["v" , "," , "o" , "<" , "^" , ".", ">"]

    # Plot embedding
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    for i in range(len(total_structures_cumsum)-1):

        #if i % 2 == 0:
        #    lab = "WT"
        #else:
        #    lab = "MT"
        
        ax.scatter(embeddings[total_structures_cumsum[i]:total_structures_cumsum[i+1],0], 
                   embeddings[total_structures_cumsum[i]:total_structures_cumsum[i+1],1], 
                   label="Cluster %s (n=%s, %.1f%%)" % (i, total_structures[i], (total_structures[i]/sum(total_structures))*100),
                   #color=colors[i%2],
                   color=new_colors[i%2][i//2],
                   marker=markers[i//2],
                   s=3, 
                   alpha=0.8)
    ax.legend()

    if p_viz_method == "umap":
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
    elif p_viz_method == "mds":
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
    elif p_viz_method == "pca":
        pc1 = model.explained_variance_ratio_[0] * 100
        pc2 = model.explained_variance_ratio_[1] * 100
        ax.set_xlabel("PC-1 (%.1f%%)" % pc1)
        ax.set_ylabel("PC-2 (%.1f%%)" % pc2)

    return fig, ax, embeddings