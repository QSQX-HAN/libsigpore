import numpy as np
import scipy.stats as spstats
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns
import networkx as nx
import igraph as ig
import leidenalg as la
import umap

from sklearn.neighbors import KNeighborsTransformer
from sklearn.metrics import accuracy_score, rand_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from matplotlib import cm
from matplotlib.colors import rgb2hex
from sklearn.metrics import accuracy_score, rand_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency, fisher_exact
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from collections import Counter, deque
from fancyimpute import SoftImpute
from pyfaidx import Fasta

from .bmm_numba import BMMsNumba
from .hbbmm import HBBMMs
from .utils_io import load_data, filter_data_by_modrate
from .utils_suboptimal_structure import get_structs, parse_struct_into_contactmap, parse_structs_into_vector, find_centroid

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def compute_chisq(X):
    
    chisq = np.ones((X.shape[1], X.shape[1]))
    OE = np.zeros((X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        for j in range(i+1):

            cm = confusion_matrix(X.iloc[:,i], X.iloc[:,j])

            # calculate chi-squared approximation
            try:
                chisq, pval, dof, expected = chi2_contingency(cm)
            except ValueError:
                pval = 1.0
            
            chisq[i,j] = pval
            chisq[j,i] = pval

            # calculate obs/exp
            try:
                oe = np.log2((cm[1,1]+1e-100)/(expected[1,1]+1e-100))
            except IndexError:
                oe = 0.0
            except ValueError:
                oe = 0.0
            
            OE[i,j] = oe
            OE[j,i] = oe

    return chisq, OE


def run_dendrogram(new_X, results, params):

    if len(params) == 4:
        (X_orig, df_orig_WT, df_orig_MT, p_gene) = params
    elif len(params) == 6:
        (X_orig, df_orig_WT, df_orig_MT, p_gene, p_start, p_end) = params
    else:
        print("Wrong number of parameters")
        assert False

    plt.figure(figsize=(8, 4))

    # PLot Dendrogram
    clusters_prop = []
    for i, r in enumerate(results):
        # Project back to original matrix
        X_pred = X_orig[X_orig.index.isin(new_X.iloc[r,:].index)]
        clusters_prop.append(X_pred.sum(axis=0)/X_pred.shape[0])

    clusters_prop = np.array(clusters_prop)
    D = pdist(clusters_prop, metric="correlation")
    Z = linkage(D, "complete")
    DN = dendrogram(Z)

    return Z, DN

def run_BMM(new_X, p_clusters, params, p_seed=386, p_unknown=0.05, 
            p_no_of_runs=1, p_threshold=0.5, p_min_iters=20, f_likelihood=None, 
            f_fasta="/mnt/projects/chengyza/projects/proj_het/data/reference/reference_of_ribosxitch_v2.fa",
            p_visualize=True, p_run_struct_analysis=True, p_impute=False, p_sparse=True, p_individual=True, p_flip=False,
            p_max_iters=100):
    
    np.random.seed(p_seed)

    if p_impute:
        
        if p_sparse:
            V = new_X.sparse.to_dense().replace(0, np.nan)
        else:
            V = new_X.replace(0, np.nan)
        Vimp = SoftImpute(min_value=0, max_value=1, max_iters=p_max_iters).fit_transform(V.values)
        df_imp = pd.DataFrame(Vimp, index=V.index, columns=V.columns)
        df_imp_new = pd.DataFrame(np.zeros(df_imp.shape, dtype=int), index=df_imp.index, columns=df_imp.columns)
        df_imp_new[df_imp>np.percentile(Vimp, 70)] = 1
        new_X = df_imp_new

    bmm = BMMsNumba(n_clusters=p_clusters, f_likelihood=f_likelihood)
    model = bmm.fit(new_X.values, no_of_runs=p_no_of_runs,
                    thrshld_ll=p_threshold,
                    min_iters=p_min_iters)
    states = model.predict()

    results = []
    for i in range(p_clusters):
        results.append(np.where(model.labels_ == i)[0].tolist())
    results = np.array(results, dtype="object")

    if p_clusters > 1:

        if p_visualize:
            # Run Visualization
            ppvs_raw, ACC, rand_index, ari, others = run_visualization(new_X, results, params, p_unknown=p_unknown, p_run_struct_analysis=p_run_struct_analysis, f_fasta=f_fasta, p_individual=p_individual, p_flip=p_flip)
            # Run Dendrogram
            Z, DN = run_dendrogram(new_X, results, params)
        else:
            ppvs_raw, ACC, rand_index, ari = get_summary_statistics(new_X, results, params, p_unknown=p_unknown)
            Z, DN = None, None
            others = None
    else:
        ppvs_raw = None
        ACC, rand_index, ari = None, None, None
        Z = None
        DN = None
        others = None

    return model, results, Z, DN, ppvs_raw, ACC, rand_index, ari, others


def run_BMM_v2(new_X, p_clusters, params, p_seed=386, p_unknown=0.05, 
               p_no_of_runs=1, p_threshold=0.5, p_min_iters=20, f_likelihood=None, 
               p_visualize=False, p_impute=False, p_sparse=True, p_max_iters=100):

    np.random.seed(p_seed)

    if p_impute:
        if p_sparse:
            V = new_X.sparse.to_dense().replace(0, np.nan)
        else:
            V = new_X.replace(0, np.nan)
        Vimp = SoftImpute(min_value=0, max_value=1, max_iters=p_max_iters).fit_transform(V.values)
        df_imp = pd.DataFrame(Vimp, index=V.index, columns=V.columns)
        df_imp_new = pd.DataFrame(np.zeros(df_imp.shape, dtype=int), index=df_imp.index, columns=df_imp.columns)
        df_imp_new[df_imp>np.percentile(Vimp, 70)] = 1
        new_X = df_imp_new

    bmm = BMMsNumba(n_clusters=p_clusters, f_likelihood=f_likelihood)
    model = bmm.fit(new_X.values, no_of_runs=p_no_of_runs,
                    thrshld_ll=p_threshold,
                    min_iters=p_min_iters)
    states = model.predict()

    results = []
    for i in range(p_clusters):
        results.append(np.where(model.labels_ == i)[0].tolist())
    results = np.array(results, dtype="object")

    return model, results


def run_HBBMM(new_X, p_clusters, params, p_seed=386, p_unknown=0.05):

    np.random.seed(p_seed)
    #p_clusters = 2
    f_likelihood = None
    p_no_of_runs = 1
    p_threshold = 0.5
    p_min_iters = 20 #300

    bmm = HBBMMs(n_clusters=p_clusters, f_likelihood=f_likelihood)
    model = bmm.fit(new_X.values, no_of_runs=p_no_of_runs,
                    thrshld_ll=p_threshold,
                    min_iters=p_min_iters)
    states = model.predict()

    results = []
    for i in range(p_clusters):
        results.append(np.where(model.labels_ == i)[0].tolist())
    results = np.array(results, dtype="object")

    # Run Visualization
    run_visualization(new_X, results, params, p_unknown=p_unknown)
    
    # Run Dendrogram
    Z, DN = run_dendrogram(new_X, results, params)

    return model, results, Z, DN

def run_louvain(new_X, params, p_resolution=1, p_score_threshold=5, 
                p_knn=False, p_freq_threshold = 0.05, p_seed=386, 
                p_unknown=0.05, 
                p_visualize=True):
    
    import networkx as nx
    #import qstest as qs
    import community as com

    if p_knn:
        knt = KNeighborsTransformer(n_neighbors=90, mode="distance", metric="jaccard") #metric="cityblock")
        D = knt.fit_transform(new_X.values)
    else:
        # Calculate Adjacency Matrix
        D = np.matmul(new_X.values, new_X.values.T)
        np.fill_diagonal(D, 0)
        D[D<p_score_threshold] = 0
    
    # Initialize Graph from Adj. Matrix
    G = nx.convert_matrix.from_numpy_array(D)

    print("Number of Nodes: %s" % G.number_of_nodes())
    print("Number of Edges: %s" % G.number_of_edges())

    # Set seed for louvain clustering to maintain reproducibility
    np.random.seed(p_seed)

    # Louvain algorithm
    #results = qs.louvain(G)

    def louvain(network, resolution=1):
        coms = com.best_partition(network, resolution=resolution)

        nodes = network.nodes()
        C = max(coms.values()) + 1

        communities = []
        for i in range(C):
            communities.append([])

        for nid in nodes:
            communities[coms[nid]].append(nid)

        return communities
    
    results = louvain(G, resolution=p_resolution)


    # Number of modules
    
    total_reads = G.number_of_nodes() #new_X.shape[0]
    results = np.array([ r for r in results if (len(r)/total_reads >= p_freq_threshold) ], dtype="object")

    if p_visualize:
        # Run Visualization
        ppvs_raw, ACC, rand_index, ari, others = run_visualization(new_X, results, params, p_unknown=p_unknown, p_run_struct_analysis=False)
        # Run Dendrogram
        Z, DN = run_dendrogram(new_X, results, params)
    else:
        ppvs_raw, ACC, rand_index, ari = get_summary_statistics(new_X, results, params, p_unknown=p_unknown)
        Z, DN = None, None
        others = None

    return results, Z, DN, ppvs_raw, ACC, rand_index, ari, others

def run_leiden(new_X, params, p_resolution=1.0, p_score_threshold=5, p_knn=False, p_seed=386):

    if p_knn:
        knt = KNeighborsTransformer(n_neighbors=90, mode="distance", metric="cityblock")
        D = knt.fit_transform(new_X.values)
    else:
        # Calculate Adjacency Matrix
        D = np.matmul(new_X.values, new_X.values.T)
        np.fill_diagonal(D, 0)
        D[D<p_score_threshold] = 0

    # Initialize Graph from Adj. Matrix
    G = nx.convert_matrix.from_numpy_array(D)
    
    print("Number of Nodes: %s" % G.number_of_nodes())
    print("Number of Edges: %s" % G.number_of_edges())

    # Set seed for louvain clustering to maintain reproducibility
    np.random.seed(p_seed)

    # Leiden algorithm
    H = ig.Graph.from_networkx(G)
    partition = la.find_partition(H, la.ModularityVertexPartition)

    #partition = la.find_partition(H, la.CPMVertexPartition,
    #                              resolution_parameter=p_resolution)
    
    results = []
    for i in sorted(Counter(partition.membership).keys()):
        results.append(np.where(np.array(partition.membership) == i)[0].tolist())
    results = np.array(results, dtype="object")

    # Number of modules
    p_freq_threshold = 0.05
    total_reads = G.number_of_nodes() #new_X.shape[0]
    results = np.array([ r for r in results if (len(r)/total_reads >= p_freq_threshold) ], dtype="object")

    # Run Visualization
    run_visualization(new_X, results, params, p_run_struct_analysis=False)

    return results


def get_summary_statistics(new_X, results, params, p_unknown=0.05):

    (X_orig, df_orig_WT, df_orig_MT, p_gene) = params

    # Plot figure
    # -------------
    new_X1 = new_X[new_X.index.isin(df_orig_WT.index)]
    new_X2 = new_X[new_X.index.isin(df_orig_MT.index)]
    new_X1 = X_orig[X_orig.index.isin(new_X1.index)]
    new_X2 = X_orig[X_orig.index.isin(new_X2.index)]

    total_reads = new_X1.shape[0] + new_X2.shape[0]
    baseline = [new_X1.shape[0]/total_reads, new_X2.shape[0]/total_reads]

    no_of_clusters = len(results)

    # for each predicted module
    ppvs = [ [] for _ in range(2) ]
    ppvs_raw = [ [] for _ in range(2) ]
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
        ppv_raw = [c1, c2]

        p1 = "%.3f" % ppv[0]
        p2 = "%.3f" % ppv[1]
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

        # Calculate Correlation
        obs = [new_X1.sum(axis=0)/new_X1.shape[0], new_X2.sum(axis=0)/new_X2.shape[0]]
        pre = X_pred.sum(axis=0)/X_pred.shape[0]
        corr, pval = spstats.pearsonr(obs[best_ppv], pre)

        # Prepare Labels
        labels = ["WT", "MT"]
        label_ppv = "PPV to %s: %.3f" % (labels[best_ppv], ppv[best_ppv])
        label_corr = "Corr: %.3f" % (corr)
        label = "%s | %s" % (label_ppv, label_corr)

        X_pred_wt = X_pred[X_pred.index.isin(new_X1.index)]
        X_pred_mt = X_pred[X_pred.index.isin(new_X2.index)]

        ppvs[0].append(ppv[0]*100)
        ppvs[1].append(ppv[1]*100)
        ppvs_raw[0].append(ppv_raw[0])
        ppvs_raw[1].append(ppv_raw[1])
        

        label_ppv_wt = "PPV to %s: %.1f%% (n=%s)" % (labels[0], ppv[0]*100, int(ppv[0]*len(r)))
        label_ppv_mt = "PPV to %s: %.1f%% (n=%s)" % (labels[1], ppv[1]*100, int(ppv[1]*len(r)))

        freq = (len(r)/sum([len(rr) for rr in results])) * 100

    '''
    # Combine Figure to get overall accuracy
    # ---------------------------------------
    total = sum([ len(r) for r in results ])
    
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

    #total = X_pred_WT.shape[0] + X_pred_MT.shape[0]
    label_WT = "Cluster 0 (n=%s, %.1f%%) | Corr: %.3f" % (X_pred_WT.shape[0], (X_pred_WT.shape[0]/total)*100, corr_WT)
    label_MT = "Cluster 1 (n=%s, %.1f%%) | Corr: %.3f" % (X_pred_MT.shape[0], (X_pred_MT.shape[0]/total)*100, corr_MT)

    from collections import Counter
    counter_WT = Counter(X_pred_WT.index.isin(new_X1.index))
    counter_MT = Counter(X_pred_MT.index.isin(new_X2.index))
    
    print((counter_WT[True] + counter_MT[True])/total)
    ACC = (counter_WT[True] + counter_MT[True])/total
    

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
    '''
    ACC = 0.0
    rand_index = 0.0
    ari = 0.0
    
    return ppvs_raw, ACC, rand_index, ari

def run_visualization(new_X, results, params, p_unknown=0.05, p_run_struct_analysis=True, 
                      f_fasta="/mnt/projects/chengyza/projects/proj_het/data/reference/reference_of_ribosxitch_v2.fa", p_individual=True, p_flip=False):

    if len(params) == 4:
        (X_orig, df_orig_WT, df_orig_MT, p_gene) = params
    elif len(params) == 6:
        (X_orig, df_orig_WT, df_orig_MT, p_gene, p_start, p_end) = params
    else:
        print("Wrong number of parameters")
        assert False

    # Plot figure
    # -------------
    new_X1 = new_X[new_X.index.isin(df_orig_WT.index)]
    new_X2 = new_X[new_X.index.isin(df_orig_MT.index)]
    new_X1 = X_orig[X_orig.index.isin(new_X1.index)]
    new_X2 = X_orig[X_orig.index.isin(new_X2.index)]

    total_reads = new_X1.shape[0] + new_X2.shape[0]
    baseline = [new_X1.shape[0]/total_reads, new_X2.shape[0]/total_reads]

    no_of_clusters = len(results)
    fig, ax = plt.subplots(no_of_clusters, 2, figsize=(20, (no_of_clusters)*3.5), sharey=False, sharex=False)

    # for each predicted module
    ppvs = [ [] for _ in range(2) ]
    ppvs_raw = [ [] for _ in range(2) ]
    candidates = []
    dict_assignments = {"WT": [], "MT": [], "UN": []}

    flipping_order = []

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
        ppv_raw = [c1, c2]

        p1 = "%.3f" % ppv[0]
        p2 = "%.3f" % ppv[1]
        best_ppv = np.argmax(np.array(ppv) - np.array(baseline))
        flipping_order.append(best_ppv)
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

        # Calculate Correlation
        obs = [new_X1.sum(axis=0)/new_X1.shape[0], new_X2.sum(axis=0)/new_X2.shape[0]]
        pre = X_pred.sum(axis=0)/X_pred.shape[0]
        corr, pval = spstats.pearsonr(obs[best_ppv], pre)

        # Prepare Labels
        labels = ["WT", "MT"]
        label_ppv = "PPV to %s: %.3f" % (labels[best_ppv], ppv[best_ppv])
        label_corr = "Corr: %.3f" % (corr)
        label = "%s | %s" % (label_ppv, label_corr)

        """
        ax[i+0][1].plot(range(X_pred.shape[1]), 
                    X_pred.sum(axis=0)/X_pred.shape[0],
                    color=color,
                    label=label)
        """

        


        X_pred_wt = X_pred[X_pred.index.isin(new_X1.index)]
        X_pred_mt = X_pred[X_pred.index.isin(new_X2.index)]

        ppvs[0].append(ppv[0]*100)
        ppvs[1].append(ppv[1]*100)
        ppvs_raw[0].append(ppv_raw[0])
        ppvs_raw[1].append(ppv_raw[1])

        if p_individual:
            candidates.append([X_pred_wt.sum(axis=0)/X_pred_wt.shape[0], 
                               X_pred_mt.sum(axis=0)/X_pred_mt.shape[0]])
        else:
            candidates.append(X_pred.sum(axis=0)/X_pred.shape[0])

        label_ppv_wt = "PPV to %s: %.1f%% (n=%s)" % (labels[0], ppv[0]*100, int(ppv[0]*len(r)))
        label_ppv_mt = "PPV to %s: %.1f%% (n=%s)" % (labels[1], ppv[1]*100, int(ppv[1]*len(r)))

        ax[i+0][1].plot(range(X_pred.columns.start, 
                              X_pred.columns.start+X_pred.shape[1]), 
                    X_pred_wt.sum(axis=0)/X_pred_wt.shape[0],
                    color=colors[0],
                    label=label_ppv_wt)

        ax[i+0][1].plot(range(X_pred.columns.start, 
                              X_pred.columns.start+X_pred.shape[1]), 
                    X_pred_mt.sum(axis=0)/X_pred_mt.shape[0],
                    color=colors[1], linestyle="--",
                    label=label_ppv_mt)


        freq = (len(r)/sum([len(rr) for rr in results])) * 100
        ax[i+0][1].set_title("Cluster %s (n=%s, %.1f%%)" % (i, len(r), freq))
        ax[i+0][1].legend()

    ax[-1][1].set_xlabel("Position")
    for i in range(0+no_of_clusters):
        ax[i][1].set_ylabel("Mod Rate")


    # Plot Heatmap
    # -------------
    #fig, ax = plt.subplots(len(results), 1, figsize=(16, 4*len(results)))
    for i, r in enumerate(results):
        # Project back to original matrix
        X_pred = X_orig[X_orig.index.isin(new_X.iloc[r,:].index)]
        sns.heatmap(X_pred, ax=ax[i][0])

    # Plot Barplot
    # -------------
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    labels = ["Cluster %s\n(n=%s, %.1f%%)" % (i, len(results[i]), (len(results[i])/total_reads)*100) for i in range(no_of_clusters) ]
    ax.barh(labels[::-1], ppvs[0][::-1], height=0.9, label="WT")
    ax.barh(labels[::-1], ppvs[1][::-1], height=0.9, left=ppvs[0][::-1], label="MT")
    ax.set_xlabel("Percentage (%)")
    #ax.set_ylim(-0.5, 2.5)
    #ax.legend(loc="upper right")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05))
    ax.set_title("%s\nPercentage of WT/MT reads per Predicted Cluster" % p_gene)
    
    #print(ppvs)
    #print(labels)

    # Combine Figure to get overall accuracy
    # ---------------------------------------
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

    fig, ax = plt.subplots(2, 1, figsize=(12, 4+1), sharex=True, sharey=True)

    ax[0].plot(range(new_X1.shape[1]), new_X1.sum(axis=0)/new_X1.shape[0], 
            label="WT (n=%s, %.1f%%)" % (new_X1.shape[0], baseline[0]*100))
    ax[0].plot(range(new_X2.shape[1]), new_X2.sum(axis=0)/new_X2.shape[0], 
            label="MT (n=%s, %.1f%%)" % (new_X2.shape[0], baseline[1]*100))

    ax[0].legend()
    ax[0].set_title("Observed")

    #total = X_pred_WT.shape[0] + X_pred_MT.shape[0]

    if p_flip:
        label_WT = "Cluster 1 (n=%s, %.1f%%) | Corr: %.3f" % (X_pred_WT.shape[0], (X_pred_WT.shape[0]/total)*100, corr_WT)
        label_MT = "Cluster 0 (n=%s, %.1f%%) | Corr: %.3f" % (X_pred_MT.shape[0], (X_pred_MT.shape[0]/total)*100, corr_MT)
    else:
        label_WT = "Cluster 0 (n=%s, %.1f%%) | Corr: %.3f" % (X_pred_WT.shape[0], (X_pred_WT.shape[0]/total)*100, corr_WT)
        label_MT = "Cluster 1 (n=%s, %.1f%%) | Corr: %.3f" % (X_pred_MT.shape[0], (X_pred_MT.shape[0]/total)*100, corr_MT)

    
    counter_WT = Counter(X_pred_WT.index.isin(new_X1.index))
    counter_MT = Counter(X_pred_MT.index.isin(new_X2.index))
    print(counter_WT)
    print(counter_MT)
    
    print((counter_WT[True] + counter_MT[True])/total)
    ACC = (counter_WT[True] + counter_MT[True])/total

    if p_flip:
        ax[1].plot(range(X_pred_WT.shape[1]), X_pred_MT.sum(axis=0)/X_pred_MT.shape[0], label=label_MT)
        ax[1].plot(range(X_pred_WT.shape[1]), X_pred_WT.sum(axis=0)/X_pred_WT.shape[0], label=label_WT)
    else:
        ax[1].plot(range(X_pred_WT.shape[1]), X_pred_WT.sum(axis=0)/X_pred_WT.shape[0], label=label_WT)
        ax[1].plot(range(X_pred_WT.shape[1]), X_pred_MT.sum(axis=0)/X_pred_MT.shape[0], label=label_MT)
    ax[1].legend()

    ax[0].set_ylabel("Mod Rate")
    ax[1].set_xlabel("Position"), ax[1].set_ylabel("Mod Rate")
    

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
    
    ax[1].set_title("Predicted | ACC: %.3f | Rand Index: %.3f | ARI: %.3f" % (ACC, rand_index, ari))

    centroids = []
    struct_ens = []
    struct_mfe = []
    seq = None
    X_norms = []
    concordance = None
    
    if p_run_struct_analysis:
        # Convert modrate into reactivities
        tmp = []
        for candidate in candidates:
            if p_individual:
                candidate_wt, candidate_mt = candidate
                tmp.append(candidate_wt)
                tmp.append(candidate_mt)
            else:
                tmp.append(candidate)

        tmp1 = pd.concat(tmp)
        norm_factor = max(spstats.iqr(tmp1) * 1.5, np.percentile(tmp1, 90))
        print(norm_factor)

        fig, ax = plt.subplots(len(tmp), 1, figsize=(16, 2 * len(tmp)))

        
        for i, X_modrate in enumerate(tmp):

            X_norm = X_modrate/norm_factor
            print(X_norm.shape)
            ax[i].plot(range(X_norm.shape[0]), X_norm)
            X_norms.append(X_norm)

        # Get sequence
        o_fasta = Fasta(f_fasta)

        
        if p_start is not None and p_end is not None:
            seq = str(o_fasta[p_gene][p_start:p_end])
        else:
            seq = str(o_fasta[p_gene])
        
        """
        seq = str(o_fasta[p_gene])
        """
        print(len(seq))

        print(np.array(ppvs_raw).T.flatten())

        if p_individual:
            total_structures = np.array(ppvs_raw).T.flatten()
        else:
            total_structures = np.sum(np.array(ppvs_raw).T.flatten().reshape(-1, 2), axis=1)
            print(total_structures)

        
        total_structures_cumsum = np.concatenate([[0], total_structures.cumsum()])
        total_structures_by_wt_or_mt_cumsum = np.concatenate([[0], np.array(ppvs_raw).T.flatten().cumsum()])
        
        for no_of_structs, X_norm in zip(total_structures, X_norms):
            struct = get_structs(seq[:-1], constraints=None, SHAPE=X_norm.values[1:], p_shape_type="deigan", temp=25.0, subopt=True, no_of_folds=no_of_structs)
            #struct = get_structs(seq, constraints=None, SHAPE=X_norm.values, p_shape_type="deigan", temp=25.0, subopt=True, no_of_folds=250)
            struct_ens.append(struct)

        
        for X_norm in X_norms:
            struct = get_structs(seq[:-1], constraints=None, SHAPE=X_norm.values[1:], p_shape_type="deigan", temp=25.0, subopt=False, no_of_folds=1)
            #struct = get_structs(seq, constraints=None, SHAPE=X_norm.values, p_shape_type="deigan", temp=25.0, subopt=False, no_of_folds=250)
            struct_mfe.append(struct)

        
        p_viz_method = "pca"

        vectors_sg = []
        for struct in struct_ens:
            vector_sg = np.array(parse_structs_into_vector(struct, p_encode="single"))
            vectors_sg.append(vector_sg)
        vectors_sg_data = np.concatenate(vectors_sg)
        print(vectors_sg_data.shape)
        print(len(struct_ens))

        if p_viz_method == "umap":
            embeddings = umap.UMAP(n_components=2, n_neighbors=10, metric="kulsinski", random_state=386).fit_transform(np.concatenate(vectors_sg))
        elif p_viz_method == "mds":
            embeddings = MDS(n_components=2).fit_transform(np.concatenate(vectors_sg))
        elif p_viz_method == "pca":
            model = PCA(n_components=2)
            embeddings = model.fit_transform(np.concatenate(vectors_sg))

        # Find Centroid
        model_kmeans = KMeans(n_clusters=no_of_clusters, random_state=386)
        model_kmeans.fit(vectors_sg_data)

        struct_ens_data = []
        for s in struct_ens:
            struct_ens_data.extend(s)

        truth_assignments = np.concatenate([ np.zeros(p, dtype=int) if m%2 == 0 else np.ones(p, dtype=int) for m, p in enumerate(np.array(ppvs_raw).T.flatten()) ]).ravel()
        concordance = accuracy_score(model_kmeans.labels_, truth_assignments)

        print("kmeans")
        print(model_kmeans.labels_.shape)
        print(Counter(model_kmeans.labels_))


        centroids_index = []
        for n in range(no_of_clusters):
            index = np.where(model_kmeans.labels_==n)
            sqdist = ((vectors_sg_data[model_kmeans.labels_==n] - model_kmeans.cluster_centers_[n]) ** 2).sum(axis=1)
            print(index[0].shape)
            print(sqdist.shape)
            struct = struct_ens_data[index[0][np.argmin(sqdist)]]
            centroids.append(struct)
            centroids_index.append(index[0][np.argmin(sqdist)])

        #print(centroids)
        print(centroids_index)


        # Initialize parameters
        cmap_blue = cm.get_cmap("Blues", len(results)+2)
        cmap_blue_hex = [ rgb2hex(cmap_blue(i)) for i in range(2, cmap_blue.N) ]

        cmap_orange = cm.get_cmap("Oranges", len(results)+2)
        cmap_orange_hex = [ rgb2hex(cmap_orange(i)) for i in range(2, cmap_orange.N) ]

        new_colors = [cmap_blue_hex, cmap_orange_hex]
        wt_mt_colors = colors[3:]

        markers = ["v" , "," , "o" , "<" , "^" , ".", ">"]

        # Plot embedding
        if p_individual:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            for i in range(len(total_structures_cumsum)-1):
                if i % 2 == 0:
                    lab = "WT"
                else:
                    lab = "MT"

                ax.scatter(embeddings[total_structures_cumsum[i]:total_structures_cumsum[i+1],0], 
                        embeddings[total_structures_cumsum[i]:total_structures_cumsum[i+1],1], 
                        label="Cluster %s %s" % (i//2, lab),
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
        else:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

            for i in range(len(total_structures_cumsum)-1):

                ax[0].scatter(embeddings[total_structures_cumsum[i]:total_structures_cumsum[i+1],0], 
                            embeddings[total_structures_cumsum[i]:total_structures_cumsum[i+1],1], 
                            label="Cluster %s" % (i),
                            color=colors[i],
                            marker=markers[i],
                            s=3, 
                            alpha=0.8)
            
            for i in range(len(total_structures_by_wt_or_mt_cumsum)-1):

                if i % 2 == 0:
                    lab = "WT"
                else:
                    lab = "MT"

                count = total_structures_by_wt_or_mt_cumsum[i+1] - total_structures_by_wt_or_mt_cumsum[i]

                ax[1].scatter(embeddings[total_structures_by_wt_or_mt_cumsum[i]:total_structures_by_wt_or_mt_cumsum[i+1],0], 
                              embeddings[total_structures_by_wt_or_mt_cumsum[i]:total_structures_by_wt_or_mt_cumsum[i+1],1],
                              label="Cluster %s %s (n=%s)" % (i//2, lab, count),
                              color=wt_mt_colors[i%2],
                              s=3, 
                              alpha=0.8)
                
            for c, m, l in zip(centroids_index, ["o", "^"], ["Centroid 0", "Centroid 1"]):
                ax[0].scatter(embeddings[c,0], embeddings[c,1], color="black", s=10, marker=m, label=l)
                
            for i in range(2):
                ax[i].legend()

                if p_viz_method == "umap":
                    ax[i].set_xlabel("UMAP-1")
                    ax[i].set_ylabel("UMAP-2")
                elif p_viz_method == "mds":
                    ax[i].set_xlabel("MDS-1")
                    ax[i].set_ylabel("MDS-2")
                elif p_viz_method == "pca":
                    pc1 = model.explained_variance_ratio_[0] * 100
                    pc2 = model.explained_variance_ratio_[1] * 100
                    ax[i].set_xlabel("PC-1 (%.1f%%)" % pc1)
                    ax[i].set_ylabel("PC-2 (%.1f%%)" % pc2)

            

        


    """
    if p_expand:
        # Plot the direct results
        fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharey=True, sharex=True)

        ax[0].plot(range(new_X1.shape[1]), new_X1.sum(axis=0)/new_X1.shape[0], 
                label="WT (n=%s, %.1f%%)" % (new_X1.shape[0], baseline[0]*100))
        ax[0].plot(range(new_X2.shape[1]), new_X2.sum(axis=0)/new_X2.shape[0], 
                label="MT (n=%s, %.1f%%)" % (new_X2.shape[0], baseline[1]*100))
        
        predictions = np.zeros(sum([ len(r) for r in results ]), dtype=int)

        truths = np.zeros(new_X.shape[0], dtype=int)
        truths[new_X.index.isin(df_orig_WT.index)] = 0
        truths[new_X.index.isin(df_orig_MT.index)] = 1
        truths = truths[sorted(np.concatenate([ list(r) for r in results]))]

        TP, TOTAL = 0, 0
        for i, r in enumerate(results):
            r = np.array(r, dtype=int)

            predictions[r] = i

            # Project back to original matrix
            X_pred = X_orig[X_orig.index.isin(new_X.iloc[r,:].index)]

            # Selected Best PPV
            # Calculate Precision
            c1 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_WT.index))
            c2 = np.count_nonzero(new_X.iloc[r,:].index.isin(df_orig_MT.index))

            #truths[new_X.iloc[r,:].index.isin(df_orig_WT.index)] = 

            # Calculate PPV
            ppv = [(c1/(c1+c2)), (c2/(c1+c2))]
            p1 = "%.3f" % ppv[0]
            p2 = "%.3f" % ppv[1]
            best_ppv = np.argmax(np.array(ppv) - np.array(baseline))
            color = colors[best_ppv]

            # Calculate Correlation
            obs = [new_X1.sum(axis=0)/new_X1.shape[0], new_X2.sum(axis=0)/new_X2.shape[0]]
            pre = X_pred.sum(axis=0)/X_pred.shape[0]
            corr, pval = spstats.pearsonr(obs[best_ppv], pre)
            freq = (len(r)/sum([len(rr) for rr in results])) * 100

            # Prepare Labels
            labels = ["WT", "MT"]
            label_clu = "Cluster %s (n=%s, %.1f%%)" % (i, len(r), freq)
            label_ppv = "PPV to %s: %.3f" % (labels[best_ppv], ppv[best_ppv])
            label_corr = "Corr: %.3f" % (corr)
            label = "%s | %s | %s" % (label_clu, label_ppv, label_corr)

            ax[1].plot(range(X_pred.shape[1]), 
                    X_pred.sum(axis=0)/X_pred.shape[0],
                    color=color,
                    label=label)

            TP += (ppv[best_ppv] * (c1+c2))
            TOTAL += len(r)
            
        #acc = max(TP/TOTAL, 1-(TP/TOTAL))

        from sklearn.metrics import accuracy_score, rand_score, adjusted_rand_score
        acc = max(accuracy_score(truths, predictions), 1-accuracy_score(truths, predictions))
        rand_index = rand_score(truths, predictions)
        ari = adjusted_rand_score(truths, predictions)
        ax[0].set_title("ACC: %.3f | Rand Index: %.3f | ARI: %.3f" % (acc, rand_index, ari))


        ax[0].legend(), ax[1].legend()

        ax[-1].set_xlabel("Position")
        for i in range(0+no_of_clusters):
            ax[i].set_ylabel("Mod Rate")
    """
    #plt.clf()

    others = {"struct_ens": struct_ens, "struct_mfe": struct_mfe, 
              "centroids": centroids,
              "seq": seq, "reactivity": X_norms, "modrate": candidates, 
              "concordance": concordance}
    return ppvs_raw, ACC, rand_index, ari, others




# ---- followin code comes from NB_011_global_BMM_analysis.py -----
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

"""
def run_visualization(new_X, results, params, p_unknown=0.05):
    pass
"""

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
        struct = get_structs(seq, constraints=None, SHAPE=X_norm.values, p_shape_type="deigan", temp=p_temp, subopt=True, no_of_folds=no_of_structs, p_slope=p_slope, p_intercept=p_intercept)
        struct_ens.append(struct)

    struct_mfe = []
    for X_norm in X_norms:
        struct = get_structs(seq[:-1], constraints=None, SHAPE=X_norm.values[1:], p_shape_type="deigan", temp=p_temp, subopt=False, no_of_folds=1, p_slope=p_slope, p_intercept=p_intercept)
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
        
        ax.scatter(embeddings[total_structures_cumsum[i]:total_structures_cumsum[i+1],0], 
                   embeddings[total_structures_cumsum[i]:total_structures_cumsum[i+1],1], 
                   label="Cluster %s (n=%s, %.1f%%)" % (i, total_structures[i], (total_structures[i]/sum(total_structures))*100),
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