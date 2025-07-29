import scipy.stats.mstats as spmstats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def get_sizes(f_sizes):
    dict_sizes = {}
    with open(f_sizes, "r") as f:
        for line in f:
            row = line.strip("\r\n").split("\t")
            chrom, size = row[0], int(row[1])
            dict_sizes[chrom] = size
    return dict_sizes


def parse_annotations(f_annotations):
    dict_annotations = {}
    with open(f_annotations, "r") as f:
        for no, line in enumerate(f):
            if no == 0:
                continue
            row = line.strip("\r\n").split("\t")
            chrom, tx_start, utr_five, utr_three, tx_end = row[0], int(row[1]), int(row[2]), int(row[3]), int(row[4])
            #if utr_five < 0:
            #    utr_five = 0
            dict_annotations[chrom] = {"tx_start": tx_start, "5utr": utr_five, "3utr": utr_three, "tx_end": tx_end}

    return dict_annotations


def load_annotations_into_tree(p_chroms, dict_annotations):
    from intervaltree import IntervalTree

    dict_trees = {}
    for p_chrom in p_chroms:

        tree = IntervalTree()
        dict_coords = dict_annotations[p_chrom]

        tree[dict_coords["tx_start"]:dict_coords["5utr"]] = 0  # 5'UTR
        tree[dict_coords["5utr"]:dict_coords["3utr"]] = 1
        tree[dict_coords["3utr"]:dict_coords["tx_end"]] = 2

        dict_trees[p_chrom] = tree

    return dict_trees


def get_coverage(p_chrom, p_conditions, p_size,
                 p_prefix="/home/ubuntu/projects/proj_het/data/candida/nanopore/dtw/pline_porecupine/results/NAIN3/calbicans"):
    coverage = np.zeros(p_size)
    for p_condition in p_conditions:
        f_modrate = "%s/%s/first_pass_minimap2/nu_1e-3_gamma_2e-1/%s_modrate.txt" % (p_prefix, p_condition, p_chrom)
        with open(f_modrate, "r") as f:
            for line in f:
                row = line.strip("\r\n").split("\t")
                pos, cov = int(row[0]), int(row[2])
                coverage[pos] += cov
    return coverage


def get_modrate(p_chrom, o_bw):
    p_size = o_bw.chroms(p_chrom)
    modrate = np.ma.masked_invalid(np.array(o_bw.values(p_chrom, 0, p_size)).astype(float))
    return modrate


def identify_dsr(p_chrom,
                 cov_cond1, cov_cond2,
                 norm_cond1, norm_cond2,
                 p_step_corr=50,
                 p_threshold_corr=0.93, 
                 p_threshold_diff=(0.005*50),
                 p_min_coverage=500):
    
    p_step_half = p_step_corr//2
    pearson_r, pearson_coords = [], []
    diff, diff_coords = [], []
    for i in np.arange(p_step_half, norm_cond1.shape[0]-p_step_half):

        if ((norm_cond1[i-p_step_half:i+p_step_half].count() < p_step_corr) or 
            (norm_cond2[i-p_step_half:i+p_step_half].count() < p_step_corr)):
            continue
        
        R, pval = spmstats.pearsonr(norm_cond1[i-p_step_half:i+p_step_half],
                                    norm_cond2[i-p_step_half:i+p_step_half])
        pearson_r.append(R)
        pearson_coords.append(i)
    
    for i in np.arange(p_step_half, norm_cond1.shape[0]-p_step_half):

        if ((norm_cond1[i-p_step_half:i+p_step_half].count() < p_step_corr) or 
            (norm_cond2[i-p_step_half:i+p_step_half].count() < p_step_corr)):
            continue
        
        D = np.abs((norm_cond1[i-p_step_half:i+p_step_half]-norm_cond2[i-p_step_half:i+p_step_half])).sum()
        diff.append(D)
        diff_coords.append(i)


    # Express this into a Data Frame
    df_pearson = pd.DataFrame({"pos": pearson_coords, "pearson": pearson_r})
    df_pearson = df_pearson.set_index("pos")

    df_diff = pd.DataFrame({"pos": diff_coords, "diff": diff})
    df_diff = df_diff.set_index("pos")


    df_total = pd.merge(df_pearson, df_diff, on="pos")
    total_dsr = np.count_nonzero((df_total["pearson"] < p_threshold_corr) & (df_total["diff"] > p_threshold_diff))


    list_of_DSR =[]
    for p in df_total[(df_total["pearson"] < p_threshold_corr) & (df_total["diff"] > p_threshold_diff)].index:

        if cov_cond1[p] < p_min_coverage or cov_cond2[p] < p_min_coverage:
            continue

        list_of_DSR.append((p_chrom, p, p+1))

    return df_total, list_of_DSR


def tabulate_DSR_by_regions(list_of_DSR, dict_trees):
    total_regions = [0, 0, 0]
    for p_chrom, p, q in list_of_DSR:
        result = list(dict_trees[p_chrom][p:q])
        if len(result) > 0:
            total_regions[result[0].data] += 1

    return total_regions
