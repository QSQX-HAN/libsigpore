import numpy as np
import RNA
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

def get_bppm(seq, p_res=1, icSHAPE=None, temp=37.0, 
             p_method="api", p_display="full"):
    """ Get the base-pairing probability matrix from RNAfold

    Args:
        seq (str): Sequence to fold
        p_res (int): Resolution
        icSHAPE (numpy ndarray): SHAPE reactivities used as restraints
        temp (float): Temperature of the experiment
        p_method (str): {api, wrapper}
        p_display (str): Which quadrant to plot the s.s. on the contact map {full, upper, lower}

    Return:
        data (numpy ndarray): Contact map of the s.s.
    """
    
    if p_method == "api":
        md = RNA.md()
        md.uniq_ML = 1
        md.temperature = temp
        fc = RNA.fold_compound(seq, md)
        if icSHAPE:
            m, b = 1.8, -0.6
            fc.sc_add_SHAPE_deigan(icSHAPE, m, b)
        #DEBUG 
        struct, fe = fc.mfe()
        #DEBUG 
        fc.exp_params_rescale(fe)
        fc.pf()
        structs = np.array(fc.bpp()) # upper triangle
    elif p_method == "wrapper":
        """
        structs = get_structures_bppm(seq, icSHAPE=icSHAPE)
        """
        pass
    
    n_bins = len(seq)//p_res
    data = np.zeros((n_bins+1, n_bins+1))
    for i in range(len(structs)):
        for j in range(len(structs)):
            
            if p_display in ("full", "lower"):
                data[i//p_res,j//p_res] += structs[i,j]
            if p_display in ("full", "upper"):
                data[j//p_res,i//p_res] += structs[i,j]
    
    return data

def get_matrix(seq, p_res=1, constraints=None, icSHAPE=None, 
               no_of_folds=250, subopt=True, 
               p_method="api", p_display="full"):
    """ Get the structure predictions from RNAfold in dot-bracket notation
    
    Args:
        seq (str): Sequence to fold
        p_res (int): Resolution
        constraints (str): RNAfold constraints
        icSHAPE (numpy ndarray): SHAPE reactivities used as restraints
        no_of_folds (int): No. of folds to make if subopt is True
        subopt (boolean): Draw samples from the partition function to get an ensemble
        p_method (str): {api, wrapper}
        p_display (str): Which quadrant to plot the s.s. on the contact map {full, upper, lower}

    Return:
        data (numpy ndarray): Contact map of the s.s.
    """
    
    if p_method == "api":
        md = RNA.md()
        md.uniq_ML = 1
        md.temperature = 37.0
        fc = RNA.fold_compound(seq, md)

        if icSHAPE:
            m, b = 1.8, -0.6
            fc.sc_add_SHAPE_deigan(icSHAPE, m, b)

        if subopt:
            fc.pf()
            structs = [ fc.pbacktrack() for i in range(no_of_folds) ]
        else:
            # MFE
            struct, fe = fc.mfe()
            structs = [struct]
    elif p_method == "wrapper":
        """
        if subopt:
            structs = get_structures(seq, icSHAPE=icSHAPE, 
                                     constraints=constraints,
                                     num_of_folds=no_of_folds, subopt=True)
        else:
            structs = get_structures(seq, icSHAPE=icSHAPE, 
                                     constraints=constraints,
                                     subopt=False)
        """
        pass
    
    n_bins = len(seq)//p_res
    data = np.zeros((n_bins+1, n_bins+1))
    for struct in structs:
        data += parse_struct_into_contactmap(struct, p_res, p_display)
    return data

def get_structs(seq, constraints=None, SHAPE=None, 
                p_shape_type="deigan",
                no_of_folds=250, temp=37.0, subopt=True, 
                p_slope=2.6, p_intercept=-0.8):
    """ Get the structure predictions from RNAfold in dot-bracket notation
    
    Args:
        seq (str): Sequence to fold
        constraints (str): RNAfold constraints
        SHAPE (numpy ndarray): SHAPE reactivities used as restraints
        p_shape_type (str): {deigan, zarringhalam}
        no_of_folds (int): No. of folds to make if subopt is True
        temp (float): Temperature of the experiment
        subopt (boolean): Draw samples from the partition function to get an ensemble

    Return:
        structs (list of str): List of s.s. in dot-notation
    """
    
    md = RNA.md()
    md.uniq_ML = 1
    md.temperature = temp
    fc = RNA.fold_compound(seq, md)
    if SHAPE is not None:
        if p_shape_type == "deigan":
            #m, b = 1.8, -0.6
            #m, b = 2.6, -0.8
            #m, b = 2.4, -0.2
            m, b = p_slope, p_intercept
            fc.sc_add_SHAPE_deigan(SHAPE, m, b)
        elif p_shape_type == "zarringhalam":
            beta = 0.89  # beta: 0.5 to 1.5 check paper fig 5
            fc.sc_add_SHAPE_zarringhalam(SHAPE, beta, -1, "Z")
    if subopt:
        struct, fe = fc.mfe()
        fc.exp_params_rescale(fe)
        fc.pf()
        structs = [ fc.pbacktrack() for i in range(no_of_folds) ]
    else:  # MFE
        struct, fe = fc.mfe()
        structs = [struct]
    return structs

def parse_structs_into_vector(structs, p_encode="double"):
    """ Convert list of s.s. in dot-bracket notation into a binary vector
    """
    data = [ parse_struct_into_vector(struct, p_encode) for struct in structs ]
    return data

def parse_struct_into_vector(struct, p_encode="double"):
    """ Convert s.s. in dot-bracket notation in a binary vector
    """
    if p_encode == "double":
        dict_symbol = {".": 0, ",": 0, "(": 1, ")": 1, "{": 1, "}": 1, "[": 1, "]": 1, "|": 1}
    elif p_encode == "single":
        dict_symbol = {".": 1, ",": 1, "(": 0, ")": 0, "{": 0, "}": 0, "[": 0, "]": 0, "|": 0}
    new_struct = [ dict_symbol[s] for s in list(struct) ]
    return new_struct

def parse_struct_into_contactmap(struct, p_res, p_display):
    """ Parse s.s. in dot-bracket notation into a contact map
    
    Args:
        struct (list of str): List of s.s. in dot-bracket notation
        p_res (int): Resolution to visualize contact map in
        p_display (str): Which quadrant to plot the s.s. on the contact map {full, upper, lower}

    Returns:
        data (numpy ndarray): Contact map of the s.s.
    """
    
    n_bins = len(struct)//p_res
    data = np.zeros((n_bins+1, n_bins+1))
    ds = []
    for j, fold in enumerate(struct):
        if fold == "(":
            ds.append(j)
        elif fold == ")":
            i = ds.pop()
            bin_i = i//p_res
            bin_j = j//p_res
            if p_display in ("full", "lower"):
                data[bin_i][bin_j] += 1
            if p_display in ("full", "upper"):
                data[bin_j][bin_i] += 1
    return data

def find_centroid(data, structs, p_no_of_clusters, p_res, p_seed=386):
    """ Find center of cluster (centroid) in an RNAfold ensemble

    Args:
        data (numpy ndarray): Contact map of the s.s.
        structs (list of str): List of s.s. in dot-bracket notation
        p_no_of_clusters (int): Number of Clusters to look for
        p_res (int): Resolution to visualize contact map in
        p_seed (int): Seed

    Returns:
        model (sklearn): Model describing attributes of a fitted KMeans instance
        centroids (list of str): List of each cluster center in dot-bracket notation
    """
    
    model = KMeans(n_clusters=p_no_of_clusters, random_state=p_seed)
    model.fit(data)

    centroids = []
    for n in range(p_no_of_clusters):
        index = np.where(model.labels_==n)
        sqdist = ((data[model.labels_==n] - model.cluster_centers_[n]) ** 2).sum(axis=0)
        struct = structs[index[0][np.argmin(sqdist)]]
        centroids.append(struct)        
    return model, centroids



if __name__ == "__main__":

    data = get_matrix("CCCTTAGACGAGCAAGGG", 1)
    plt.imshow(data, "Reds")
    plt.colorbar()
    plt.savefig("tmp31.png")
    
