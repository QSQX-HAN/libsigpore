import sys
import pandas as pd
import ruptures as rpt
import argparse
import h5py
import numpy as np
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as spstats
import pickle as pkl

from collections import OrderedDict
from tslearn import metrics
from pyfaidx import Fasta
from itertools import product
from multiprocessing import Pool

def load_signal_reference(f_reference):
    signal_references = {}
    
    o_h5 = h5py.File(f_reference)
    p_genes = list(o_h5.keys())

    for p_gene in p_genes:
        try:
            dset = o_h5[p_gene]
            signal_references[p_gene] = dset[:]
        except KeyError:
            pass
        
    return signal_references

def get_list_of_reads(f_h5):
    with h5py.File(f_h5, "r") as o_fast5:
        list_of_reads = [ iid.split("_")[1] for iid in o_fast5["/"] ]
    return list_of_reads

def align_with_subseq_DTW(signal_ref, signal_query, p_use_norm):
    if p_use_norm:
        signal_query = spstats.zscore(signal_query)
        signal_ref = spstats.zscore(signal_ref)
        #signal_query[signal_query<-2] = 0

    path, simi = metrics.dtw_subsequence_path(signal_query, signal_ref)
    final_length = path[-1][1]-path[0][1]
    return path, simi, final_length

def align_to_reference(datum):
    
    read_id, ref_genes, dict_results = datum
    
    dict_results["success_align"] = False
    dict_results["ref_gene"] = []
    dict_results["path"] = []
    dict_results["score_edist"] = []
    dict_results["query_align_len"] = []
    
    
    changepoint_signal = dict_results["changepoint_signal"]
    index_of_adapter = dict_results["index_of_adapter"]

    if dict_results["success"]:
        
        for ref_gene in ref_genes:
        
            signal_reference = signal_references[ref_gene]

            try:
                path, score, align_len = align_with_subseq_DTW(signal_reference,
                                                               changepoint_signal[:index_of_adapter],
                                                               p_use_norm=True)
                norm_score = score/len(changepoint_signal[:index_of_adapter])
                dict_results["success_align"] = True
            except IndexError:
                path, score, align_len, norm_score = None, None, None, None
            
            dict_results["ref_gene"].append(ref_gene)
            dict_results["path"].append(path)
            dict_results["score_edist"].append(score)
            dict_results["query_align_len"].append(align_len)
        
    
    return (read_id, dict_results)

def run_alignment(f_pkl, f_reference, ref_genes, p_depth=-1, p_cores=4):

    with gzip.open(f_pkl, "rb") as f:
        dict_results = pkl.load(f)

    read_ids = list(dict_results.keys())
    if (p_depth is not None) or (p_depth != -1):
        read_ids = read_ids[:p_depth]

    # Load Signal Reference
    # ----------------------
    def init_vars(f_reference):
        global signal_references
        signal_references = load_signal_reference(f_reference)

    data = []
    for read_id in read_ids:
        dict_preprocess = dict_results[read_id]
        data.append((read_id, ref_genes, dict_preprocess))


    # Run Analysis in Chunks
    # ------------------------
    with Pool(processes=p_cores, initializer=init_vars, initargs=(f_reference,)) as p:
        results = p.map(align_to_reference, data)
    return results