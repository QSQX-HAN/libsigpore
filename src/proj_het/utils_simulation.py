import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import numpy as np
import scipy.stats as spstats
import itertools
import argparse
import pickle
import os
import glob
import RNA
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pyfaidx import Fasta
from collections import OrderedDict

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids

from sklearn.metrics import accuracy_score, homogeneity_score, completeness_score
from sklearn.metrics import v_measure_score, fowlkes_mallows_score
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from forgi.graph.bulge_graph import BulgeGraph

from .dreem import DREEM
from .bmm_vi import BMM_VI
from .bmm import BMMs
from .utils_rnafold import get_structs, get_matrix

class Simulation(object):
    """
    The Simulation object builds the simulated nanopore reads 
    and has methods for immediate evaluation via a variety of clustering 
    methods. It does this by receiving an identifier (p_sample) 
    to generate reference sequence and secondary structure,
    followed by the scenario type, frequency and libraries.

    Attributes
    ----------
    p_sample : str
        the identifier that determines how to proceed with the simulation
    p_maxlen : int
        the maximum length of the reference sequence - useful for non-real sequence
    p_seed : int
        the random number seed for maintaining reproducibility
    p_rep : int
        the replicate index/number
    p_depth: int
        expected sequencing depth
    p_mod_rate : float
        expected modification rate for each transcript
    p_length: int
        length of transcript
    p_no_of_mods : int
        the number of modifications per molecule
    p_freq_code : str
        frequency code to determine what level of conformation is wanted
    p_freq : float
        relative frequency for each 
    p_no_of_seq : int
        number of sequence (reference + alternatives) 
    p_lib_seed : int
        the random number seed for libraries generation
    p_similarity: float
        similarity (jaccard) of the different conformation
    f_dreem_prefix : str
        folder to write out dreem results to
    p_FPR : float
        false positive rate levels of false modifications
    p_molecule : str
        use dstrb of modifications using inferred params for NAI or 1AI
    seq : str
        the sequence of the reference
    reference : str
        pass
    alternative : list of str
        pass
    scenario : list of numpy int
        pass
    truths : 
        pass
    library : list of numpy int
        library of nanopore reads as defined by the modification rate
    predictions : list of int
        list of predictions from the evaluator
    predicted_counts : list
        pass
    index : list of int
        current list of indexes to truth assignments
    """

    def __init__(self, p_sample, p_rep, p_depth, p_mod_rate, p_length,
                 p_freq_code, p_similarity, f_dreem_prefix=None,
                 p_FPR=0.0, p_molecule=None):

        self.p_sample = p_sample
        self.p_rep = p_rep
        self.p_depth = p_depth
        self.p_mod_rate = p_mod_rate
        self.p_FPR = p_FPR
        self.p_eff_mod_rate = p_mod_rate - p_FPR
        self.p_length = p_length
        self.p_no_of_mods = -1 if p_mod_rate < 0 else int(self.p_eff_mod_rate * p_length)
        self.p_freq_code = p_freq_code
        self.p_freq = self.get_freq(p_freq_code)
        self.p_no_of_seq = len(self.p_freq)
        self.p_similarity = p_similarity
        self.f_dreem_prefix = f_dreem_prefix
        
        self.p_molecule = p_molecule
        
        self.p_sid = None

        # others
        #self.predictions = None
        #self.predicted_counts = None  # shld be a list []
        #self.index = None

    def generate_samples(self):
        # initialize parameters
        self.p_maxlen, self.p_seed = self.get_sample_info()
        if self.p_length == -1:
            self.p_length = self.p_maxlen
        np.random.seed(self.p_seed)

        # build sequences
        self.seq = self.get_seq()
        self.structures = self.create_structures()
        self.scenario = self.create_scenario()
        self.truths = self.generate_truths()

    def prepare_library(self):
        self.p_lib_seed = self.p_rep * self.p_seed
        self.library = self.generate_library()
        
    def get_freq(self, fid):
        dict_freq = {"A1": [1.0],
                     "A2": [0.5, 0.5],
                     "B2": [0.6, 0.4],
                     "C2": [0.75, 0.25],
                     "D2": [0.80, 0.20],
                     "E2": [0.90, 0.10],
                     "F2": [0.95, 0.05],
                     "G2": [0.99, 0.01],
                     "A3": [0.5, 0.25, 0.25],
                     "A4": [0.5, 0.25, 0.125, 0.125],
                     "A5": [0.5, 0.25, 0.125, 0.075, 0.05],
                     "A6": [0.5, 0.25, 0.125, 0.075, 0.025, 0.025],
                     "A7": [0.5, 0.25, 0.125, 0.075, 0.025, 0.015, 0.01],
                     "A8": [0.5, 0.25, 0.125, 0.075, 0.025, 0.010, 0.01, 0.005],
                     "U3": [0.34, 0.33, 0.33],
                     "U4": [0.25, 0.25, 0.25, 0.25],
                     "U5": [0.2, 0.2, 0.2, 0.2, 0.2],
                     "U6": [0.167, 0.167, 0.167, 0.167, 0.167, 0.165],
                     "U7": [0.148, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142],
                     "U8": [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]}
        return dict_freq[fid]

    def get_sample_info(self):
        dict_sample_info = {"SIM001": (2000, 386)}
        return dict_sample_info[self.p_sample]

    def create_structures(self):
        '''
        Data structure holding the RNA structures 
        dict_structs: 
        key | int: cluster
        value | list: list of ints: structure
        '''
        dict_structs = OrderedDict()
        struct = np.random.randint(0, 1+1, self.p_maxlen)
        dict_structs[0] = np.array([struct])

        for no in range(1, len(self.p_freq)):
            alt = []
            prob = np.random.random(self.p_maxlen)
            for c, o in zip(struct, prob):
                if o > self.p_similarity:
                    if c == 1:
                        c = 0
                    else:
                        c = 1
                alt.append(c)
            alt = np.array(alt)
            dict_structs[no] = np.array([alt])
        
        return dict_structs


    def get_sid(self, p_method, p_cluster):
        p_sid = "_".join(map(str, [self.p_sample, p_method, self.p_length,
                                   self.p_depth, self.p_mod_rate,
                                   self.p_freq_code, self.p_similarity, p_cluster, self.p_rep]))
        return p_sid

    def get_seq(self):
        alphabets = ["A", "T", "G", "C"]
        np.random.seed(self.p_seed)
        seq = "".join([ alphabets[np.random.randint(0, 4)] for _ in range(self.p_maxlen) ])
        return seq

    def gen_joint_interdist_vs_modrate(self):
        params_dist = (0.9473481125912115, 97.96492942510092,
                       0.001574803149606299, 9.548446279341746)
        if self.p_molecule == "NAI":
            params_mods = (2.743941781321144, 18.316278199484522,
                           -0.001104242858945835, 0.054028979717955)
        elif self.p_molecule == "1AI":
            params_mods = (2.6334556377049547, 12.2767347158523,
                           0.0020349613579214616, 0.049000229105008154)
        
        no_of_mods = int(spstats.beta.rvs(params_mods[0], params_mods[1],
                                          params_mods[2], params_mods[3], size=1) * self.p_length)
        #print(no_of_mods)
        read = np.zeros(len(self.seq), dtype=int)
        if no_of_mods == 0:
            return read

        # Perform Pseudo Rejection Sampling
        index_of_mods = []
        is_rejected = True
        counter = 0
        while is_rejected:

            dist_of_mod = int(spstats.beta.rvs(params_dist[0], params_dist[1],
                                               params_dist[2], params_dist[3], size=1) * self.p_length)
            if dist_of_mod >= self.p_length:
                continue

            if len(index_of_mods) == 0:
                if self.seq[dist_of_mod] == 1:
                    index_of_mods.append(dist_of_mod)
                    counter = 0
                else:
                    counter += 1
            else:
                mm = index_of_mods[-1]+dist_of_mod
                if mm < self.p_length and self.seq[mm] == 1:
                    index_of_mods.append(mm)
                    counter = 0
                else:
                    counter += 1

            if len(index_of_mods) == no_of_mods or counter == 10:
                #print(no_of_mods, index_of_mods)
                is_rejected = False
                counter = 0
            
        read[index_of_mods] = 1

    
    def simulate_read(self, seq):
        '''
        Given an RNA structure (list of integers), simulate a nanopore read
        using 3 modes inferred from our chemical probes: NAI | 1AI | stated mod-rate
        '''
        seq = np.array(seq)
        index_of_truths = np.where(seq==1)[0]
        if self.p_molecule == "NAI":
            params_mods = (2.743941781321144, 18.316278199484522,
                           -0.001104242858945835, 0.054028979717955)
            no_of_mods = int(spstats.beta.rvs(params_mods[0], params_mods[1],
                                              params_mods[2], params_mods[3], size=1) * self.p_length)
            index_of_mods = np.random.choice(index_of_truths, size=no_of_mods)
        elif self.p_molecule == "1AI":
            params_mods = (2.6334556377049547, 12.2767347158523,
                           0.0020349613579214616, 0.049000229105008154)
            no_of_mods = int(spstats.beta.rvs(params_mods[0], params_mods[1],
                                              params_mods[2], params_mods[3], size=1) * self.p_length)
            index_of_mods = np.random.choice(index_of_truths, size=no_of_mods)
        elif self.p_molecule == "poisson":
            no_of_mods = spstats.poisson.rvs(self.p_mod_rate*self.p_length)
            index_of_mods = np.random.choice(index_of_truths, size=no_of_mods)
        else:
            index_of_mods = np.random.choice(index_of_truths, size=self.p_no_of_mods)
        
        read = np.zeros(len(seq), dtype=int)
        read[index_of_mods] = 1
        
        return read

    def simulate_error(self, read):
        '''
        Introduction of simulated false positive errors
        0: Negative
        1: Positive
        '''
        if (self.p_FPR is not None) and (self.p_FPR != 0.0):
            for i in np.where(read==0)[0]:
                if np.random.rand() < self.p_FPR:
                    read[i] = 1
            return read
        else:
            return read

    def create_scenario(self):
        '''
        Truncate all to the stated length. If p_length > full length, use full length
        New structure accomodates clusters of structures found in RNA ensembles
        '''
        scenario = []
        for cluster, structures in self.structures.items():
            scenario.append(structures[:,:self.p_length])
        return scenario
    
    def generate_truths(self):
        datum_truth = np.random.choice(self.p_no_of_seq, self.p_depth,
                                       p=self.p_freq)  # generate truths
        return datum_truth

    def generate_library(self):
        np.random.seed(self.p_lib_seed)  # set seed for replicates
        data = []
        for i in self.truths:  # simulate reads for lib
            index = np.random.randint(0, self.scenario[i].shape[0])
            read = self.simulate_read(self.scenario[i][index])
            read = self.simulate_error(read)
            data.append(read)
        return np.array(data)
        
    # TODO - remove
    def generate_library_2D(self):
        data = np.array(data)
        print(data.shape)
        D = []
        for bv in data:
            datum = np.zeros((data.shape[1], data.shape[1]))
            mod_ind = np.where(bv == 1)[0]
            num_ind = len(mod_ind)
            for i in range(num_ind):
                for j in range(i):
                    datum[mod_ind[i],mod_ind[j]] += 1
                    datum[mod_ind[j],mod_ind[i]] += 1
            D.append(datum[np.triu_indices(data.shape[1])])
        D = np.array(D)
        print(D.shape)
        return(D)
        
    def convert_to_dreem(self, f_output):
        with open(f_output, "w") as f:
            f.write("\t".join(["@ref", "%s;%s" % (self.sid, self.sid), self.seq]) + "\n")
            f.write("\t".join(["@coordinates:length", "1,%s:%s" % (self.p_length, self.p_length)]) + "\n")
            f.write("\t".join(["Query_name", "Bit_vector", "N_Mutations"]) + "\n")
            for no, d in enumerate(self.library):
                no_of_mutations = d.sum()
                f.write("\t".join(["%s.1" % no,
                                "".join(map(str, d)),
                                str(no_of_mutations)]) + "\n")
        return True

    def convert_to_draco(self, f_output):
        with open(f_output, "w") as f:
            f.write("\t".join(map(str, range(self.p_length))) + "\n")
            for no, d in enumerate(self.library):
                f.write("\t".join(map(str, d)) + "\n")
        return True

    def write_to_fasta(self, f_output):
        with open(f_output, "w") as f:
            f.write(">%s\n" % self.sid)
            f.write("%s\n" % self.seq)
        return True

    def parse_dreem_results(self, f_result, p_cluster, p_threshold=0.5):
        datum_pred = []
        dict_pred = {}
        with open(f_result, "r") as f:
            for no, line in enumerate(f):
                if no % 2 == 1:
                    row = line.strip("\r\n").split("\t")
                    rid, posteriors, bitvector = row[0], list(map(float, row[1:1+p_cluster])), row[-1]
                    cluster_no = np.argmax(posteriors)  # max posterior
                    datum_pred.append(cluster_no)
                    dict_pred[bitvector] = cluster_no

        new_datum_truth = []
        new_datum_pred = []
        new_data = []
        new_index = []
        for no, (d, t) in enumerate(zip(self.library, self.truths)):
            try:
                p = dict_pred["".join(map(str, d.tolist()))]
                new_datum_truth.append(t)
                new_datum_pred.append(p)
                new_data.append(d)
                new_index.append(no)
            except KeyError:
                pass
        new_datum_truth = np.array(new_datum_truth)
        new_datum_pred = np.array(new_datum_pred)
        new_data = np.array(new_data)
        new_index = np.array(new_index)

        return new_datum_pred, new_datum_truth, new_data, new_index

    def get_predicted_counts(self):
        return [ self.library[self.predictions==j].sum(axis=0) for j in range(self.p_cluster) ]

    def evaluate(self, p_method, p_cluster, f_input_prefix=None):

        self.sid = self.get_sid(p_method, p_cluster)
        self.p_method = p_method
        self.p_cluster = p_cluster

        if p_method == "kmeans":
            model = KMeans(n_clusters=p_cluster, random_state=self.p_seed)
            model.fit(self.library)
            self.predictions = model.labels_
            self.predicted_counts = self.get_predicted_counts()
            self.index = range(len(self.library))
        elif p_method == "ward":
            model = AgglomerativeClustering(n_clusters=p_cluster)
            model.fit(self.library)
            self.predictions = model.labels_
            self.predicted_counts = self.get_predicted_counts()
            self.index = range(len(self.library))
        elif p_method == "kmedoids":
            model = KMedoids(n_clusters=p_cluster, random_state=self.p_seed)
            model.fit(self.library)
            self.predictions = model.labels_
            self.predicted_counts = self.get_predicted_counts()
            self.index = range(len(self.library))
        elif p_method == "spectral":
            model = SpectralClustering(n_clusters=p_cluster, random_state=self.p_seed,
                                       affinity="nearest_neighbors")
            model.fit(self.library)
            self.predictions = model.labels_
            self.predicted_counts = self.get_predicted_counts()
            self.index = range(len(self.library))
        elif p_method == "BMM_VI":
            model = BMM_VI(n_clusters=p_cluster)
            model.fit(self.library)
            self.predictions = model.labels_
            self.predicted_counts = self.get_predicted_counts()
            self.index = range(len(self.library))
        elif p_method == "BMM":
            models = BMMs(n_clusters=p_cluster, f_likelihood="stderr")
            model = models.fit(self.library, no_of_runs=1, thrshld_ll=1e-2)  #10
            model.predict()
            self.predictions = model.labels_
            self.predicted_counts = self.get_predicted_counts()
            self.index = range(len(self.library))
        elif p_method == "dreem":

            #f_input_prefix = "/mnt/projects/chengyza/projects/proj_het/scratch/dreem/"

            dir_input = self.f_dreem_prefix + "input/"
            os.system("mkdir -p %s" % dir_input)
            dir_output = "./"
            dir_outplot = self.f_dreem_prefix + ("output/%s/" % self.sid)
            f_input = dir_input + self.sid + "." + p_method
            f_fasta = dir_input + self.sid + ".fasta"
            os.system("mkdir -p %s" % dir_outplot)

            self.convert_to_dreem(f_input)
            self.write_to_fasta(f_fasta)

            model = DREEM(self.sid, f_input, dir_input, dir_output, dir_outplot,
                          MIN_ITS=3, INFO_THRESH=0.05, CONV_CUTOFF=0.5, NUM_RUNS=1,  # 1e-5
                          MAX_K=self.p_cluster, SIG_THRESH=0.001, BV_THRESH=0, 
                          NORM_PERC_BASES=10, inc_TG=True, p_seed=self.p_seed)

            # Fit model and Parse DREEM results into DREEM object
            model.fit()
            model.parse_dreem_results(self.library, self.truths)
            
            old_truths = self.truths  # will be truncated
            self.truths = model.truths
            self.library = model.data
            self.predictions = model.labels_
            self.predicted_counts = self.get_predicted_counts()
            self.index = model.index

        elif p_method == "draco":

            f_input_prefix = "/mnt/projects/chengyza/projects/proj_het/data/draco/"
            f_input = f_input_prefix.rstrip("/") + "/input/" + self.p_sid + "." + self.p_method
            f_fasta = f_input_prefix.rstrip("/") + "/input/" + self.p_sid + ".fasta"
            f_mm = f_input_prefix.rstrip("/") + "/input/" + self.p_sid + "." + self.p_method + ".mm"
            f_draco_output = f_input_prefix.rstrip("/") + "/output/" + self.p_sid + ".json"

            seq = self.get_seq(p_length, self.p_seed)
            self.convert_to_draco(f_input)
            self.write_to_fasta(f_fasta)

            os.system("perl /mnt/projects/chengyza/projects/proj_het/src/wanlab_tsv2mm.pl %s %s" % (f_input, f_fasta))
            os.system("draco --mm %s --output %s --processors 4 --shape --winLenFraction 1.0 --minMutationFreq 0 --minPermutations 20" % (f_mm, f_draco_output))

        return self

    def load_results(self, p_method, f_result):
        self.p_method = p_method
        if self.p_method == "dream":
            pass
        elif self.p_method == "draco":
            pass

    def output_results(self, f_output=None, f_results=None):
        """
        A method to output all files from
        this Simulation
        """
        # performance measures
        if self.p_cluster == 2:
            acc = accuracy_score(self.truths, self.predictions)
            acc = max(acc, 1-acc)
        else:
            acc = "NaN"
        fms = fowlkes_mallows_score(self.truths, self.predictions)
        ari = adjusted_rand_score(self.truths, self.predictions)
        ami = adjusted_mutual_info_score(self.truths, self.predictions)

        # output
        results = [self.p_sample, self.p_method, self.p_length,
                   self.p_depth, self.p_mod_rate,
                   self.p_freq_code, self.p_similarity, self.p_cluster,
                   self.p_rep, self.p_FPR,
                   acc, fms, ari, ami]

        if f_results:
            with open(f_results, "w") as f:
                f.write(",".join(map(str, results)) + "\n")

        # output truth and predictions
        if f_output:
            with open(f_output, "w") as f:
                for i, m, n in zip(self.index, self.truths, self.predictions):
                    f.write("%s,%s,%s" % (i,m,n) + "\n")

        return results

    def output_correlation(self):
        correlation = []
        for i, j in itertools.product(range(self.p_no_of_seq), range(self.p_cluster)):
            R, pval = spstats.pearsonr(self.library[self.truths==i].sum(axis=0),
                                       self.library[self.predictions==j].sum(axis=0))
            correlation.append([i, j, R, pval])

        correlation_matrix = np.zeros((2,2))
        for i, j, R, pval in correlation:
            correlation_matrix[i,j] = R

        ax = sns.heatmap(correlation_matrix, cmap="bwr", annot=True, fmt=".3f") #, vmin=-0.1, vmax=0.1)
        plt.title("Correlation Matrix")
        #plt.xticks([0, 1], [0, 1])
        #plt.yticks([0, 1], [0, 1])
        plt.xlabel("Predicted Label")
        plt.ylabel("Truth Label")
        return self

    def plot_embeddings(self, ax = None):

        data = self.library
        dist = pdist(data, "cityblock")
        dist = squareform(dist)

        mds = MDS(n_components=2, random_state=386, dissimilarity="precomputed")
        embedding = mds.fit_transform(dist)

        if not ax:
            fig, ax = plt.subplots(figsize=(5, 5))
        for i in range(self.p_cluster+1):
            ax.scatter(embedding[self.truths==i,0],
                       embedding[self.truths==i,1], s=10, label="observed")
        ax.set_title("S=%s M=%s" % (self.p_similarity, self.p_mod_rate))
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

        return self

    def plot_embeddings_overlaying_predictions(self, ax = None):

        data = self.library
        dist = pdist(data, "cityblock")
        dist = squareform(dist)

        mds = MDS(n_components=2, random_state=386, dissimilarity="precomputed")
        embedding = mds.fit_transform(dist)

        if not ax:
            fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_prop_cycle(color=sns.color_palette("Dark2", self.p_cluster))
        for i in range(self.p_cluster+1):
            ax.scatter(embedding[self.predictions==i,0],
                       embedding[self.predictions==i,1], s=10, label="observed")
        ax.set_title("S=%s M=%s" % (self.p_similarity, self.p_mod_rate))
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

        return self

    def plot_embedding_side_by_side(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
        self.plot_embeddings(ax[0])
        self.plot_embeddings_overlaying_predictions(ax[1])

        return self

class SimulationRealistic(Simulation):
    
    def __init__(self, p_sample, p_rep, p_depth, p_mod_rate, p_length,
                 p_freq_code, f_dreem_prefix=None, p_FPR=0.0, p_molecule=None):
        Simulation.__init__(self, p_sample, p_rep, p_depth, p_mod_rate, p_length,
                            p_freq_code, p_similarity=None,
                            f_dreem_prefix=f_dreem_prefix, p_FPR=p_FPR, p_molecule=p_molecule)

    def generate_samples(self, f_pkl, f_fasta, p_seed):

        o_fasta = Fasta(f_fasta)
        seq = str(o_fasta[self.p_sample])

        with open(f_pkl, "rb") as f:
            dict_structs = pkl.load(f)
        cc = dict_structs[self.p_sample]["cluster_centers_bv"]

        self.seq = seq[:self.p_length]
        self.dotbracket = dict([ (n, [i])
        for n, i in enumerate(dict_structs[self.p_sample]["cluster_centers_st"]) ])
        self.structures = dict([ (n, np.array([i])) 
        for n, i in enumerate(dict_structs[self.p_sample]["cluster_centers_bv"]) ])
        self.scenario = self.create_scenario()
        self.truths = self.generate_truths()

        s = (1.0 - squareform(pdist(cc, metric="cityblock"))/cc.shape[1]).flatten()
        self.p_similarity = float("%.3f" % np.sort(s[s!=1.0])[-1])  # highest similarity

        # initialize parameters
        self.p_maxlen, self.p_seed = len(seq), p_seed
        if self.p_length == -1:
            self.p_maxlen = self.p_length 
        np.random.seed(self.p_seed)

    def simulate_splash_read(self, st, p_armwidth=30, p_truth=False):
        '''
        Simulate SPLASH reads using RNAcofold
        '''

        bv = np.array(get_matrix([st], p_single=True)[0])
        bg = BulgeGraph.from_dotbracket(st)
        pairs = bg.to_pair_tuples()
        
        # remove not base-paired and only keep left bracket
        pairs = [ pair for pair in pairs 
                if (0 not in pair) and (pair[0] < pair[1]) ]
        
        # generate the intervals
        no_of_pairs = len(pairs)
        lig_start, lig_end = pairs[np.random.randint(no_of_pairs)]
        s1, e1 = lig_start-p_armwidth+1, lig_start
        s2, e2 = lig_end, lig_end+p_armwidth-1

        # refine interval positions if boundaries crossed
        s1 = 0 if s1 < 0 else s1
        e2 = len(st) if e2 > len(st) else e2
        #DEBUG print(s1, e1, s2, e2)

        # run RNAcofold and parse results
        read1, read2 = self.seq[s1:e1], self.seq[s2:e2]
        result = RNA.cofold("%s&%s" % (read1, read2))
        structs = [result[0][:len(read1)], result[0][len(read1):]]  # structs of arms
        matrix = [ np.array(get_matrix([s], p_single=True)[0], dtype=int) for s in structs ]
        
        # truth
        if p_truth:
            new_bv = np.zeros(len(st), dtype=int)
            new_bv[s1:e1] = bv[s1:e1]
            new_bv[s2:e2] = bv[s2:e2]
            #print("".join(map(str, new_bv.tolist())))
            #print(st)
        else:
            # new results
            new_bv = np.zeros(len(st), dtype=int)
            new_bv[s1:e1] = matrix[0] 
            new_bv[s2:e2] = matrix[1]
            #DEBUG print("".join(map(str, new_bv.tolist())))
        
        return new_bv

    def prepare_library(self):
        self.p_lib_seed = self.p_rep * self.p_seed
        self.library = self.generate_library(p_splash=0.0)

    def generate_library(self, p_splash=0.0):
        np.random.seed(self.p_lib_seed)  # set seed for replicates
        data = []
        for i in self.truths:  # simulate reads for lib
            
            index = np.random.randint(0, self.scenario[i].shape[0])
            #print("".join(map(str, self.scenario[i][index])))
            if np.random.rand() < p_splash:
                read = self.simulate_splash_read(self.dotbracket[i][index], p_armwidth=30, p_truth=True)
            else:
                read = self.simulate_read(self.scenario[i][index])
                read = self.simulate_error(read)
            
            #read = self.simulate_splash_read(self.dotbracket[i][index], p_armwidth=30, p_truth=True)
            
            #assert False
            data.append(read)
        return np.array(data)

class SimulationRealisticEnsemble(SimulationRealistic):
    
    def __init__(self, p_sample, p_rep, p_depth, p_mod_rate, p_length,
                 p_freq_code, f_dreem_prefix=None, p_FPR=0.0, p_molecule=None):
        SimulationRealistic.__init__(self, p_sample, p_rep, p_depth, p_mod_rate, p_length,
                                     p_freq_code, f_dreem_prefix=f_dreem_prefix, p_FPR=p_FPR, 
                                     p_molecule=p_molecule)

    """
    def pseudocluster(self):
        p_min_cluster, p_max_cluster = 2, 10
        p_model_selection = "calinski_harabasz"
        models = []
        for i in range(p_min_cluster, p_max_cluster+1):
            model = KMeans(n_clusters=i, random_state=386)
            model.fit(data)
            labels = np.unique(model.labels_)

            if p_model_selection == "silhouette":
                score = silhouette_score(data, labels=model.labels_)  # highest
            elif p_model_selection == "davies_bouldin":
                score = davies_bouldin_score(data, labels=model.labels_) # lowest
            elif p_model_selection == "calinski_harabasz":
                score = calinski_harabasz_score(data, labels=model.labels_) # highest
            print(i, score)
            models.append((model, score))
        if p_model_selection in ("silhouette", "calinski_harabasz"):
            models = sorted(models, key=lambda q: q[1], reverse=True)
            print("highest: %.3f" % models[0][1])
        else:
            models = sorted(models, key=lambda q: q[1], reverse=False)
            print("lowest: %.3f" % models[0][1])

        model = models[0][0]
    """
    
    def get_cluster_of_structures(self, dict_structs):
        structures = dict_structs[self.p_sample]["structures"]
        cc = dict_structs[self.p_sample]["cluster_centers_bv"]
        data = np.array(get_matrix(structures, p_single=True))

        model = KMeans(n_clusters=len(cc), random_state=386)
        model.fit(data)
        labels = np.unique(model.labels_)
        
        dotbracket = {}
        for i, st in zip(model.labels_, structures):
            try:
                dotbracket[i].append(st)
            except KeyError:
                dotbracket[i] = [st]
        return dict([ (i, data[model.labels_==i]) for i in labels ]), dotbracket

    def generate_samples(self, f_pkl, f_fasta, p_seed):

        o_fasta = Fasta(f_fasta)
        seq = str(o_fasta[self.p_sample])

        with open(f_pkl, "rb") as f:
            dict_structs = pkl.load(f)
        cc = dict_structs[self.p_sample]["cluster_centers_bv"]

        self.seq = seq[:self.p_length]
        self.structures, self.dotbracket = self.get_cluster_of_structures(dict_structs)
        self.scenario = self.create_scenario()
        self.truths = self.generate_truths()

        s = (1.0 - squareform(pdist(cc, metric="cityblock"))/cc.shape[1]).flatten()
        self.p_similarity = float("%.3f" % np.sort(s[s!=1.0])[-1])  # highest similarity

        # initialize parameters
        self.p_maxlen, self.p_seed = len(seq), p_seed
        if self.p_length == -1:
            self.p_length = self.p_maxlen
        np.random.seed(self.p_seed)

"""
def plot_violinplot(f_results):
    # PLOT
    #print(contingency_matrix(datum_truth, datum_pred))
    df = pd.DataFrame(results, columns=["sample", "method", "length", "mod_rate",
                                        "similarity", "rep", "acc", "fms"])
    print(df)
    print(len(results))

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(data=df, x="mod_rate", y="acc", hue="method")
    plt.xlabel("Modification Rate")
    plt.ylabel("Accuracy")
    plt.title("Similarity = %s | Depth = %s | Length = %s" % (p_similarity, p_depth, p_length))
    plt.legend(loc="lower right")
    plt.axhline(0.8, linewidth=0.5, linestyle="--", color="black", zorder=-10)
    plt.ylim(0.2, 1.2)
    plt.savefig("violinplot_%s_%s_%s.png" % (p_similarity, p_depth, p_length))
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", default=None, help="sample name")
    parser.add_argument("-l", default=100, type=int, help="length")
    parser.add_argument("-m", default=None, type=float, help="mod rate")
    parser.add_argument("-d", default=100, type=int, help="depth")
    parser.add_argument("-f", default="A2", help="freq code")
    parser.add_argument("-s", default=None, type=float, help="similarity")
    parser.add_argument("-num", default=None, type=int, help="cluster")
    parser.add_argument("-r", default=None, type=int, help="rep")
    parser.add_argument("-m", default=None, 
                        help="method: kmeans | ward | spectral | dreem | BMM")
    parser.add_argument("-o", default=None, help="output pred")
    parser.add_argument("-t", default=None, help="output file")
    parser.add_argument("-prefix", default=None, help="input prefix for dreem and BMM")
    parser.add_argument("-fpr", default=0.0, type=float,
                        help="Simulated Error: False Positive Rate")
    parser.add_argument("-mod", default=None, help="Molecular Probe: NAI | 1AI")
    args = parser.parse_args()
    
    p_sample = args.n
    p_method = args.t
    p_length = args.l
    p_depth = args.d
    p_mod_rate = args.m
    p_freq_code = args.f
    p_similarity = args.s
    p_cluster = args.num
    p_rep = args.r
    f_output = args.o
    f_results = args.t
    f_dreem_prefix = args.prefix
    p_FPR = args.fpr
    p_molecule = args.mod
    
    sim = Simulation(p_sample, p_rep, p_depth, p_mod_rate, p_length,
                     p_freq_code, p_similarity,
                     f_dreem_prefix=f_dreem_prefix,
                     p_FPR=p_FPR, p_molecule=p_molecule)
    sim.generate_samples()
    sim.prepare_library()

    if p_cluster is None:
        p_cluster = sim.p_no_of_seq
    
    sim.evaluate(p_method, p_cluster, f_input_prefix=None)
    sim.output_results(f_output, f_results)

