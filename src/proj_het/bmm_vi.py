import numpy as np
from bayespy.utils import random
from bayespy.nodes import Categorical, Dirichlet
from bayespy.nodes import Beta
from bayespy.nodes import Mixture, Bernoulli
from bayespy.inference import VB

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from .bmm import import_dreem

class BMM_VI(object):
    
    def __init__(self, n_clusters):
        self.K = n_clusters
        
    def fit(self, data):

        data = np.array(data) > 0
        self.N, self.D = data.shape
        
        self.R = Dirichlet(self.K*[1e-5], name="R")
        self.Z = Categorical(self.R, plates=(self.N,1), name='Z')
        self.P = Beta([0.5, 0.5], plates=(self.D, self.K), name="P")
        self.X = Mixture(self.Z, Bernoulli, self.P)
        
        self.Q = VB(self.Z, self.R, self.X, self.P)
        self.P.initialize_from_random()
        self.X.observe(data)
        self.Q.update(repeat=1000)

        self.prob = self.Z.u[0].reshape(self.N, self.K)
        self.labels_ = np.argmax(self.prob, axis=1)
    
          
def main(f_input, f_output, 
         p_clusters, p_input_type = "dreem"):

    # ----- load data -----
    if p_input_type == "dreem":
        index, data = import_dreem(f_input)
    elif p_input_type == "csv":
        index, data = import_csv(f_input)

    # ----- run model -----
    model = BMM_VI(n_clusters=p_clusters)
    model.fit(data)
    
    datum_pred = model.labels_
    datum_posteriors = model.Z.u[0].reshape(N, K)
    
    # ----- output -----
    if f_output is not None:
        with open(f_output, "w") as f:  # output truth and predictions
            for i, m, n, p in zip(index, datum_truth, datum_pred, datum_posteriors):
                str_p = ",".join(["%.5f" % q for q in p ])
                f.write("%s,%s,%s,%s" % (index[i], m, n, str_p) + "\n")

    results = [acc, fms, ari, ami]  # output stats
    print(results)

    return(bmm)

def example1():
    
    p0 = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    p1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
    p2 = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    p = np.array([p0, p1, p2])
    print(p.shape)
    
    N = 100
    D = 300
    K = 3
    
    np.random.seed(386)
    
    z = random.categorical([1/3, 1/3, 1/3], size=N)
    x = random.bernoulli(p[z])

    x = np.array(x, dtype=int)
    model = BMM_VI(n_clusters=K)
    model.fit(x)
    
    #preds = []
    #for truth, prob in zip(z, Z.u[0].reshape(N, K)):
    #for truth, label in zip(z, model.labels_):
    #    print(truth, label)
    #    preds.append(label)

    preds = model.labels_
    print(adjusted_rand_score(z, preds))
    
    #print(model.prob[:5])
    #print(Z.get_parameters()[0].reshape(K, N))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", default=None, help="Input file")
    parser.add_argument("-o", default=None, help="Output file")
    parser.add_argument("-c", default=3, type=int, help="Number of Clusters")
    parser.add_argument("-q", default=None, help="Input Type: dreem | csv")
    args = parse_args()

    f_input = args.i
    f_output = args.o
    p_clusters = args.c
    p_no_of_runs = args.r
    p_threshold = args.t
    p_min_iters = args.min_iters

    main()