import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io as sio


def get_sizes(f_sizes):
    sizes = {}
    with open(f_sizes, "r") as f:
        for line in f:
            row = line.strip("\r\n").split("\t")
            sizes[row[0]] = int(row[1])
    return sizes


def convert_mm_to_df(f_mtx, f_id, p_gene, 
                     f_sizes, p_sep=",", p_threshold_len_prop=0.8):
    """
    Utility function for processing sparse matrix .mtx files
    """

    # load sizes file
    sizes = get_sizes(f_sizes)
    
    # load iid file
    if pd.read_csv(f_id, index_col=0, sep=p_sep).shape[1] == 1:
        df_iid = pd.read_csv(f_id, index_col=0, sep=p_sep, names=["iid"])
    else:
        df_iid = pd.read_csv(f_id, index_col=0, sep=p_sep, names=["iid", "length", "start"])
    
    # load sparse matrix
    sm = sio.mmread(f_mtx)

    # determine whether sparse matrix is undersized and remedy if so
    new_row_size = df_iid.shape[0]
    new_column_size = sizes[p_gene]
    if sm.shape[1] < new_column_size:  # resize
        sm.resize((new_row_size, new_column_size))
    elif sm.shape[1] > new_column_size:
        print(sm.shape[1], new_column_size)
        print("shapemap column greater than genome sizes file")
        assert False

    # load sparse matrix as a pandas dataframe
    df_mm = pd.DataFrame.sparse.from_spmatrix(sm)
    df_mm.index = df_iid.iid

    if df_iid.shape[1] >= 2:
        # Threshold by Length
        df_mm = df_mm.loc[df_iid[df_iid.length >= (p_threshold_len_prop * sizes[p_gene])].iid,:]
    
    return df_mm


def load_data(f_mtxs, f_ids, p_genes, f_sizes=None, p_depth=-1, p_length=-1, 
              p_start=None, p_end=None, p_threshold=1,
              p_threshold_len_prop=0.8, p_verbose=True):
    """
    Loads SHAPE-MaP or PORE-cupine data based on the input specifications
    Must be able to accept both dense matrix or sparse matrix
    """

    dfs = []
    for f_mtx, f_id, p_gene in zip(f_mtxs, f_ids, p_genes):
        df = convert_mm_to_df(f_mtx, f_id, p_gene,
                            f_sizes=f_sizes,
                            p_threshold_len_prop=p_threshold_len_prop)
        dfs.append(df)
    
    if p_length == -1:
        p_start = 0
        p_end = dfs[0].shape[1]

    if p_depth == -1:
        p_depth = min([ df.shape[0] for df in dfs ])

    if p_verbose:
        print("%s %s %s" % (p_start, p_end, " ".join(map(str, [ df.shape[0] for df in dfs]))))
        print(p_depth)

    if p_depth is None:
        ndfs =  [ df.iloc[:,p_start:p_end] for df in dfs ]
    else:
        ndfs = [ df.iloc[:p_depth,p_start:p_end] for df in dfs ]
    
    ndf = pd.concat(ndfs, axis=0)

    X = ndf
    X = X.loc[X.sum(axis=1)>=p_threshold,:]
    X = X.fillna(0)

    return X, ndfs


'''
def filter_data_by_modrate(X, p_modrate_low=0.0075, p_modrate_high=0.025):
    """
    Filters reads by the min/max modrate per read
    """
    modrate = X.sum(axis=1)/X.shape[1]
    X = X[(modrate > p_modrate_low)&(modrate < p_modrate_high)]
    print(X.sum(axis=1).mean(), p_modrate_low * X.shape[1], p_modrate_high * X.shape[1], X.shape[1])
    return X
'''


def filter_data_by_modrate(X, p_modrate_low=0.0075, p_modrate_high=0.025, p_axis=0):
    """
    Filters reads/positions by the specified min/max modrate
    """
    modrate = X.sum(axis=p_axis)/X.shape[p_axis]
    if p_axis == 0:  # by cols/positions
        #X = X.loc[:,(modrate > p_modrate_low) & (modrate < p_modrate_high)]
        X.loc[:,(modrate <= p_modrate_low)] = 0
        X.loc[:,(modrate >= p_modrate_high)] = 0
    elif p_axis == 1:  # by rows/reads
        X = X.loc[(modrate > p_modrate_low) & (modrate < p_modrate_high),:]
        #print(X.sum(axis=1).mean(), p_modrate_low * X.shape[1], p_modrate_high * X.shape[1], X.shape[1])
    return X


def filter_data_by_dist(X, p_distmuts_threshold=4):

    index_retained, index_removed = [], []
    for index, x in zip(X.index, X.values):

        to_remove = False
        latest_mutbit_index = 0
        for i in range(len(x)):
            if x[i] == 1:
                if (i - latest_mutbit_index) < p_distmuts_threshold:
                    to_remove = True
                latest_mutbit_index = i

        if not to_remove:
            index_retained.append(index)
        else:
            index_removed.append(index)

    print(X.shape)
    new_X = X.loc[index_retained,:]
    print(new_X.shape)

    return new_X


def impute_data_by_dist(X, p_distmuts_threshold=4):
    new_X = np.array([ x for x in X.values ])
    print(int(new_X.sum().sum()))
    for no, x in enumerate(new_X):
        latest_mutbit_index = 0
        for i in range(len(x)):
            if x[i] == 1:
                if (i - latest_mutbit_index) < p_distmuts_threshold:
                    #x[i] = 0
                    new_X[no,i] = 0
                else:
                    latest_mutbit_index = i
    print(int(new_X.sum().sum()))
    new_X = pd.DataFrame(new_X, index=X.index, columns=X.columns)
    return new_X


def collapse_data(X, p_res=5):
    dim = X.shape
    new_dim = (dim[0], dim[1]//p_res)
    new_X = np.zeros(new_dim)
    np_X = X.values
    for i in range(new_dim[0]):
        for j in range(new_dim[1]-p_res):
            segments = np_X[i,(j*p_res):(j+1)*p_res]
            if segments.sum() > 0:
                new_X[i,j] = 1
    new_X = pd.DataFrame(new_X)
    new_X = new_X.assign(iid=X.index)
    new_X = new_X.set_index("iid")
    new_X.columns = [ i * p_res for i in range(new_dim[1]) ]
    #print(new_X.head())
    return new_X