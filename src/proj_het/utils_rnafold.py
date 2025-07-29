import matplotlib.pyplot as plt
import numpy as np
import RNA
import pickle as pkl
import umap

from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from pyfaidx import Fasta
from collections import OrderedDict, Counter

from .bmm_numba import BMMsNumba


def get_structs(seq, no_of_folds=250, temp=37.0, p_rescale=False):
    md = RNA.md()
    md.uniq_ML = 1
    md.temperature = temp
    fc = RNA.fold_compound(seq, md)

    if p_rescale:
        # compute MFE and MFE structure
        ss, mfe = fc.mfe()
        # rescale Boltzmann factors for partition function computation
        fc.exp_params_rescale(mfe)
    
    fc.pf()
    structs = [ fc.pbacktrack() for i in range(no_of_folds) ]
    return structs

def get_mfe(seq, temp=37.0):
    md = RNA.md()
    md.uniq_ML = 1
    md.temperature = temp
    fc = RNA.fold_compound(seq, md)
    ss, mfe = fc.mfe()
    return [ss]

# Maximum Expected Acccuracy Structure
def get_centroid(seq, temp=37.0):
    md = RNA.md()
    md.uniq_ML = 1
    md.temperature = temp

    # create fold_compound data structure (required for all subsequently applied  algorithms)
    fc = RNA.fold_compound(seq, md)

    # compute MFE and MFE structure
    ss, mfe = fc.mfe()

    # rescale Boltzmann factors for partition function computation
    fc.exp_params_rescale(mfe)
    (pp, pf) = fc.pf()

    # compute centroid structure
    (centroid_struct, dist) = fc.centroid()
    
    # compute free energy of centroid structure
    #centroid_en = fc.eval_structure(centroid_struct)
    
    # compute MEA structure
    #(MEA_struct, MEA) = fc.MEA()
    
    # compute free energy of MEA structure
    #MEA_en = fc.eval_structure(MEA_struct)

    return [centroid_struct]

def get_matrix(structs, p_single=False):
    data = []
    for struct in structs:
        new_struct = []
        for s in list(struct):
            if s == ".":
                if p_single:
                    new_struct.append(1)
                else:
                    new_struct.append(0)
            else:
                if p_single:
                    new_struct.append(0)
                else:
                    new_struct.append(1)
        data.append(new_struct)
    return data

def plot_RNAsubopt(seq=None, p_chrom=None, 
                   f_fasta=None, no_of_folds=250, temp=37.0, p_incl=[], ax=None):
    
    if p_chrom is not None and f_fasta is not None:
        o_fasta = Fasta(f_fasta)
        seq = str(o_fasta[p_chrom]).upper()
    
    structs = get_structs(seq, no_of_folds, temp)

    if "centroid" in p_incl:
        print("centroid")
        struct = get_centroid(seq, temp)
        print(len(struct))
        structs += struct

    if "mfe" in p_incl:
        print("mfe")
        struct = get_mfe(seq, temp)
        print(len(struct))
        structs += struct

    data = np.array(get_matrix(structs))
    print(data.shape)
    
    dist = pdist(data, "cityblock")
    dist = squareform(dist)

    # ===============
    # Plot Embedding
    # ===============
    mds = MDS(n_components=2, random_state=386, dissimilarity="precomputed")
    embedding = mds.fit_transform(dist)
    
    """
    ax[0].scatter(embedding[model.labels_==0,0],
                  embedding[model.labels_==0,1], s=2, label="cluster 1")
    ax[0].scatter(embedding[model.labels_==1,0],
                  embedding[model.labels_==1,1], s=2, label="cluster 2")
    ax[0].scatter(embedding[centroids[0],0],
                  embedding[centroids[0],1], s=2, label="cluster 1")
    ax[0].scatter(embedding[centroids[1],0],
                  embedding[centroids[1],1], s=2, label="cluster 2")
    """
    labels = ["MFE", "Centroid"]
    markers = ["X", "D"]
    
    ax.scatter(embedding[:-2,0], embedding[:-2,1], s=2, label="RNAsubopt")
    for i, label, marker in zip(range(2, 0, -1), labels, markers):
        ax.scatter(embedding[-i,0], embedding[-i,1], label=label, marker=marker, s=50)

    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.legend()

    return ax

class RNAStructureEnsemble(object):

    def __init__(self):
        self.iid = None
        self.cluster_centers_bv = None
        self.cluster_centers_st = None
        self.cluster_centers_similarity = None
        self.similarity = None
        self.structures = None

        self.data = None
        self.p_cluster_centers = None
        self.p_success = None
        self.model = None
    
    @staticmethod
    def load_pickle(f_pickle):
        with open(f_pickle, "rb") as f:
            o_pkl = pkl.load(f)
        rse = RNAStructureEnsemble()
        rse.iid = list(o_pkl.keys())[0]
        rse.cluster_centers_bv = o_pkl[rse.iid]["cluster_centers_bv"]
        rse.cluster_centers_st = o_pkl[rse.iid]["cluster_centers_st"]
        try:
            rse.cluster_centers_similarity = o_pkl[rse.iid]["similarity"]
        except KeyError:
            pass
        rse.structures = o_pkl[rse.iid]["structures"]
        rse.convert_dot_to_data()
        return rse
    
    def dump_pickle_into_dict(self, f_pickle):
        dict_output = []
        with open(f_pickle, "rb") as f:
            pass
    
    def generate_structures(self, seq, p_no_of_folds, p_temp=37.0):
        self.structures = get_structs(seq, p_no_of_folds, p_temp)

    def convert_dot_to_data(self):
        self.data = np.array(get_matrix(self.structures, p_single=True))
    
    def cluster(self, K=None, p_method="kmeans"):
        K = K if K is not None else len(self.cluster_centers_bv)
        if p_method == "kmeans":
            self.model = KMeans(n_clusters=K, random_state=386)
            self.model.fit(self.data)
        elif p_method == "BMM":
            np.random.seed(472)
            bmm = BMMsNumba(n_clusters=K, f_likelihood=None)
            model = bmm.fit(self.data, no_of_runs=1, thrshld_ll=0.05, min_iters=100)
            states = model.predict()
            self.model = model
        return self
    
    def find_cluster_centroids(self, p_min_freq=0.05, p_centers=True):
        counter = Counter(self.model.labels_)
        p_no_of_folds = len(self.structures)
        self.p_success = OrderedDict()
        self.p_cluster_centers = []

        for k, v in sorted(counter.items(), key=lambda q: q[1], reverse=True):
            if v/p_no_of_folds > p_min_freq:
                self.p_success[k] = (v, v/p_no_of_folds)

                if p_centers:  # cluster centers
                    curr_index = np.argmin(squareform(pdist(self.data[self.model.labels_==k,:], 
                                                    "cityblock")).sum(axis=0))
                else:  # random
                    curr_index = np.random.choice(np.where(self.model.labels_==k)[0])

                index = np.where(self.model.labels_==k)[0][curr_index]
                self.p_cluster_centers.append(index)

                print("Cluster: %d | No. of Structs: %.d | Freq: %.3f | Index: %s" % (k, v, v/p_no_of_folds, index))
                print(self.structures[index])  # dot structure notation
        
        return self

    def calc_similarity(self, p_similarity="manhattan"):
        if p_similarity == "manhattan":
            self.similarity = 1.0 - squareform(pdist(self.data, "cityblock"))/self.data.shape[1]
        elif p_similarity == "jaccard":
            self.similarity = 1.0 - squareform(pdist(self.data, "jaccard"))
        return self

    def calc_similarity_summary(self, p_similarity="manhattan"):
        s = self.similarity.flatten()
        s = s[s!=1.0]
        return (s.min(), s.mean(), np.percentile(s, 50), s.max())
    
    def plot_embedding(self, n_clusters=None, figsize=(6, 6), p_method="MDS"):
        n_clusters = n_clusters if n_clusters is not None else len(np.unique(self.model.labels_))
        if p_method == "MDS":
            dist = pdist(self.data, "cityblock")
            dist = squareform(dist)
            mds = MDS(n_components=2, random_state=386, dissimilarity="precomputed")
            embedding = mds.fit_transform(dist)
        elif p_method == "umap":
            embedding = umap.UMAP(n_components=2, n_neighbors=10, metric="kulsinski", random_state=386).fit_transform(self.data)

        fig, ax = plt.subplots(figsize=figsize)
        for i in range(n_clusters):
            if i in self.p_success:
                n, freq = self.p_success[i]
                ax.plot(embedding[self.model.labels_==i, 0], 
                        embedding[self.model.labels_==i, 1], "o", 
                        label="Cluster %d (n=%d, %.2f)" % (i, n, freq))
            else:
                ax.plot(embedding[self.model.labels_==i, 0], 
                        embedding[self.model.labels_==i, 1], "o", color="grey")

        for n, i in zip(self.p_success.keys(), self.p_cluster_centers):
            ax.plot(embedding[i, 0], embedding[i, 1], "o", ms=5.0,
                    color="black")
        ax.legend()
        return fig, ax


if __name__ == "__main__":

    pass
    
