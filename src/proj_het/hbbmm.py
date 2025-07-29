import sys
import numpy as np
import scipy as sp
import scipy.stats as spstats
import argparse

from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import rand_score, adjusted_rand_score

#--Cython-- cimport cython
#--Cython-- from cython.parallel import prange, parallel
#--Cython-- from libc.math cimport log

class HBBMMs(object):

    def __init__(self, n_clusters, f_likelihood=None):
        self.K = n_clusters
        self.models = []
        if f_likelihood is not None:
            if f_likelihood == "stderr":
                self.f_ll = sys.stderr
            else:
                self.f_ll = open(f_likelihood, "w")
        else:
            self.f_ll = None
        
    def fit(self, data, thrshld_ll=1e-5, min_iters=-1, no_of_runs=1):

        list_of_BICs = []
        for run_idx in range(no_of_runs):

            print("Current Run: %s" % run_idx)
            
            model = HBBMM(n_clusters=self.K, run_idx=run_idx)
            model.run_EM(data,
                         thrshld_ll=thrshld_ll,
                         min_iters=min_iters,
                         o_likelihood=self.f_ll)
            
            self.models.append(model)
            list_of_BICs.append(model.BIC)

        # close loglikelihood log file
        if self.f_ll is not None:
            self.f_ll.close()
        
        # find best model
        best_model = np.argmin(list_of_BICs)
        print("Best Model - Run %s" % best_model)
        return self.models[best_model]
        

class HBBMM(object):

    def __init__(self, n_clusters, run_idx=0):
        self.K = n_clusters
        self.beta_A = 1.5
        self.beta_B = 20
        self.pseudo = 1e-308
        self.run_idx = run_idx
        # others
        # self.N, self.D
        # self.mu, self.pi
        # self.data
        # self.alpha_hat
        # self.beta_hat

    def initialize_beta_prior(self):
        N = self.data.shape[0] * self.data.shape[1]
        C = np.count_nonzero(self.data == 1)
        
        beta_hat_0 = (N + np.sqrt(N**2 - (4*C*(N-C))))/(2*C)
        alpha_hat_0 = ((beta_hat_0 - C + N) + np.sqrt(((beta_hat_0*C) + N)**2 + (4*C*(C-N)*beta_hat_0)))/(2*(N-C))

        beta = ((((N-C)*alpha_hat_0)+C) + np.sqrt(((((N-C)*alpha_hat_0)+C)**2) - (4*C*(N-C)*alpha_hat_0)))/(2*C)
        alpha = (((C*beta) + N) + np.sqrt(((C*beta)+N)**2 + (4*C*beta*(C-N))))/(2*(N-C))

        self.alpha_hat = alpha
        self.beta_hat = beta
        
    def initialize_mu(self):
        self.mu = np.asarray([spstats.beta.rvs(self.beta_A, self.beta_B, size=self.D)
                              for k in range(self.K)])
        #self.mu = np.asarray([(np.random.random(size=self.D)/2.0)+0.25
        #                      for k in range(self.K)])
        
    def initialize_pi(self):
        self.pi = np.asarray([1.0 / self.K for _ in range(self.K)])

    # -- @cython.boundscheck(False)
    def log_p_x_mu(self, k_ind, n_ind):  # log probability of x given mu
        ''' Definitions: shape of inputs
        u = U_k: (D,)
        x = X_n: (D,)
        '''
        
        #--CYTHON-- cdef double [:] u = self.mu[k_ind,:]
        #--CYTHON-- cdef int [:] x = self.data[n_ind,:]
        
        u = self.mu[k_ind,:]
        x = self.data[n_ind,:]
        return x * np.log(u+self.pseudo) + (1-x) * np.log(1-u+self.pseudo)

    def log_p_x_mu_pi(self, k_ind, n_ind):  # log probability of x given joint dstrb of mu and pi
        x = np.log(self.pi[k_ind]) + self.log_p_x_mu(k_ind, n_ind).sum()
        return(x)
    
    #--CYTHON-- @cython.boundscheck(False)
    def calc_posterior(self):  # posterior probability of latent variable z_n_k
        loglikelihood = []  # N x K
        posterior = []  # N x K
        
        for n_ind in range(self.N):
            
            ll = np.array([ self.log_p_x_mu_pi(k_ind, n_ind) for k_ind in range(self.K) ])
            denom = sp.special.logsumexp(ll)
            loglikelihood.append(ll)
            posterior.append(np.exp(ll-denom))
        
        self.loglikelihood = np.asarray(loglikelihood)
        self.posterior = np.asarray(posterior)
        
    def calc_complete_loglikelihood(self):
        self.complete_loglikelihood = (self.posterior * self.loglikelihood).sum()
        
    def update_mu(self):  # analytical step for getting new mu
        mu = []
        for k_ind in range(self.K):
            mu_k = []
            for d_ind in range(self.D):
                
                #num = ((self.posterior[:,k_ind] * self.data[:,d_ind]) + self.alpha_hat - 1).sum()
                #denom = (self.posterior[:,k_ind] + self.alpha_hat + self.beta_hat - 2).sum()
                
                num = ((self.posterior[:,k_ind] * self.data[:,d_ind]).sum() + self.alpha_hat - 1)
                denom = (self.posterior[:,k_ind].sum() + self.alpha_hat + self.beta_hat - 2)
                
                mu_k.append(num/denom)
            mu.append(mu_k)
        self.mu = np.asarray(mu)

    def update_pi(self):  # analytical step for getting new pi
        pi = []
        for k_ind in range(self.K):
            pi.append(self.posterior[:,k_ind].sum()/self.N)
        self.pi = np.asarray(pi)
        
    def expectation_step(self):
        self.calc_posterior()
        self.calc_complete_loglikelihood()
        
    def maximization_step(self):
        self.update_mu()
        self.update_pi()

    def compute_performance(self, truth):
        pred = np.argmax(self.posterior, axis=1)
        ari = adjusted_rand_score(truth, pred)
        ami = adjusted_mutual_info_score(truth, pred)
        print(self.complete_loglikelihood, ari, ami)

    def calc_BIC(self):
        self.BIC = (np.log(self.N) * (self.D + 1) * self.K -
                    (2 * self.complete_loglikelihood))
        return(self.BIC)

    def predict(self):
        self.labels_ = np.argmax(self.posterior, axis=1)
        return(self.labels_)

    def run_EM(self, data,
               thrshld_ll=1e-5, min_iters=np.inf,
               o_likelihood=None):
        
        self.N, self.D = data.shape
        self.data = data

        self.initialize_beta_prior()
        self.initialize_mu()
        self.initialize_pi()
        
        converged = False
        iteration = 1
        self.complete_loglikelihood = -np.inf
        
        while not converged:

            curr_ll = self.complete_loglikelihood
            
            # Expectation Step
            self.expectation_step()
            
            # Maximization Step
            self.maximization_step()
            
            new_ll = self.complete_loglikelihood
            delta_ll = new_ll - curr_ll

            #print("%s\t%s\t%s\t%s\t%s" % (self.run_idx, iteration,
            #                              new_ll, delta_ll,
            #                              self.calc_BIC()))
            
            if o_likelihood is not None:
                o_likelihood.write("%s\t%s\t%s\t%s\t%s" % (self.run_idx, iteration,
                                                           new_ll, delta_ll,
                                                           self.calc_BIC()) + "\n")
            else:
                self.calc_BIC()
            
            if (abs(new_ll - curr_ll) < thrshld_ll) and (iteration >= min_iters):
                converged = True
            else:
                iteration += 1
            

    def output_posteriors(self, index, f_output, p_output_type="dreem"):
        datum_pred = self.labels_
        datum_posteriors = self.posterior
    
        # ----- output -----
        with open(f_output, "w") as f:  # output truth and predictions

            if p_output_type == "dreem":
                f.write("\t".join(["Number"]
                                  + ["Cluster_%s" % (m+1) for m in range(self.posterior.shape[1])]
                                  + ["N", "Bit_vector"]) + "\n")
            
            for n, (i, m, p) in enumerate(zip(index, datum_pred, datum_posteriors)):

                if p_output_type == "csv":
                    str_p = ",".join(["%.5f" % q for q in p ])
                    f.write("%s,%s,%s" % (i, m, str_p) + "\n")
                elif p_output_type == "dreem":
                    str_p = "\t".join(["%.5f" % q for q in p ])
                    bv = "".join(list(map(str, self.data[n])))
                    f.write("%s\t%s\t%s\t%s" % (i, str_p, 1, bv) + "\n\n")
                    

    def output_mu(self, f_output, p_output_type="dreem", header=None):
        mu = self.mu.T
        with open(f_output, "w") as f:

            if p_output_type == "dreem" and header:
                for h in header:
                    f.write(h + "\n")
                f.write("\t".join(["Position"]
                                  + ["Cluster_%s" % (m+1) for m in range(mu.shape[1])]) + "\n")
            
            for i, m in enumerate(mu):

                if p_output_type == "csv":
                    str_m = ",".join(["%.5f" % q for q in m ])
                    f.write("%s,%s" % (i, str_m) + "\n")
                elif p_output_type == "dreem":
                    str_m = "\t".join(["%.5f" % q for q in m ])
                    f.write("%s\t%s" % (i+1, str_m) + "\n")

def import_dreem(f_dreem, p_bv):
    header, index, data = [], [], []
    with open(f_dreem, "r") as f:
        for no, line in enumerate(f):
            if no <= 1:
                header.append(line.strip("\r\n"))
                continue
            elif no == 2:
                continue
            
            row = line.strip("\r\n").split("\t")
            iid, bs, nmuts = row[0], row[1], int(row[2])

            if is_distmuts_valid(bs, distmuts_threshold=p_bv):
                bv = list(map(int, list(bs)))
                data.append(bv)
                index.append(iid)
    
    print(len(data), len(index))        
    return header, index, np.array(data)

def is_distmuts_valid(bs, distmuts_threshold=4):
    """
    Additional constraint from DREEM
    Function copied from EM_Functions.py
    @author DREEM
    """
    for i in range(len(bs)):
        if bs[i] == '1':
            try:
                if i - latest_mutbit_index < distmuts_threshold:
                    return False
            except NameError:  # This happens the first time we see a '1'
                None
            latest_mutbit_index = i
    return True

# TODO: to implement or copy from
def import_csv(f_csv):
    pass

def main(f_input, f_output, f_output_mu, f_likelihood,
         p_clusters, p_input_type = "dreem",
         p_no_of_runs=10, p_threshold=0.5, p_min_iters=300, p_bv=0):

    # ----- load data -----
    if p_input_type == "dreem":
        header, index, data = import_dreem(f_input, p_bv)
    elif p_input_type == "csv":
        header, index, data = import_csv(f_input)
    
    # ----- run model -----
    hbbmm = HBBMMs(n_clusters=p_clusters, f_likelihood=f_likelihood)
    model = hbbmm.fit(data, no_of_runs=p_no_of_runs, 
                    thrshld_ll=p_threshold, 
                    min_iters=p_min_iters)
    model.predict()
    
    if f_output is not None:
        model.output_posteriors(index, f_output, p_output_type="dreem")
    
    if f_output_mu is not None:
        model.output_mu(f_output_mu, p_output_type="dreem", header=header)
    
    return(hbbmm)


def example1(p_clusters):

    n_clusters = p_clusters #3
    
    p0 = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    p1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
    p2 = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    p = np.array([p0, p1, p2])
    K = 3
    N = 1000
    
    np.random.seed(386)
    
    #z = np.random.choice(range(K), p=[1/K for _ in range(K)], size=100)
    
    from bayespy.utils import random
    z = random.categorical([1/K for _ in range(K)], size=1000)
    x = random.bernoulli(p[z])
    x = np.asarray(x, dtype=int)

    # --------------------------------
    # Initialize HBBMM class and Run EM
    # --------------------------------
    model = HBBMM(n_clusters=n_clusters)
    model.run_EM(x)

    print(model.mu)
    pred = np.argmax(model.posterior, axis=1)
    ari = adjusted_rand_score(z, pred)
    ami = adjusted_mutual_info_score(z, pred)
    print(model.complete_loglikelihood, model.calc_BIC(), ari, ami)
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", default=None, help="Input file")
    parser.add_argument("-o", default=None, help="Output file")
    parser.add_argument("-m", default=None, help="Cluster Mu file")
    parser.add_argument("-l", default=None, help="Log likelihood file")
    parser.add_argument("-r", default=10, type=int, help="Number of Runs")
    parser.add_argument("-t", default=0.5, type=float, help="Convergence threshold")
    parser.add_argument("-min_iters", default=300, type=int, help="Min number of iterations")
    parser.add_argument("-c", default=2, type=int, help="Number of Clusters")
    parser.add_argument("-q", default=None, help="Input Type: dreem | csv")
    parser.add_argument("-seed", default=386, type=int, help="seed")
    parser.add_argument("-bv", default=0, type=int, help="Bit Vector threshold")
    args = parser.parse_args()
    
    f_input = args.i
    f_output = args.o
    f_output_mu = args.m
    f_likelihood = args.l
    p_clusters = args.c
    p_no_of_runs = args.r
    p_threshold = args.t
    p_min_iters = args.min_iters
    p_input_type = args.q
    p_seed = args.seed
    p_bv = args.bv
    
    np.random.seed(p_seed)
    
    main(f_input, f_output, f_output_mu, f_likelihood,
         p_clusters, p_input_type=p_input_type,
         p_no_of_runs=p_no_of_runs, p_threshold=p_threshold,
         p_min_iters=p_min_iters, p_bv=p_bv)
    
    
        
        

    
