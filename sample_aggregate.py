import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
sns.set_style("whitegrid")
sns.set_context("poster")

# plot histogram of conduction on samples
def hist_plot(sample_conds, actual_cond, eps=0.5):
    plt.hist(sample_conds, alpha=0.7, edgecolor='grey', linewidth=1.5)
    plt.axvline(actual_cond, color='r', label='Conductance=' +
                str(actual_cond)[:6], alpha=0.8)
    med = np.median(sample_conds)
    dp_med = dp_median(sample_conds, eps)
    plt.axvline(med, color='g', label='Median=' + str(med)[:6], alpha=0.8)
    plt.axvline(dp_med, color='b', label='DP Med.=' + str(dp_med)[:6], alpha=0.5, ls='dashed')
    plt.legend()
    plt.show()

# partition graph g into k parts
def partition(g, k=None):
    n = g.number_of_nodes()
    # if number of partitions not specified, use log of total number of nodes
    if k == None:
        k = int(np.log(n))
    # get random permutation of vertices
    perm = np.random.permutation(n)
    sz = int(1.0 * n / k)
    subgraphs = []
    for i in np.arange(0, k * sz, sz, dtype=int):
        block = perm[i:min(i + sz, n)]
        # make a new copy of the subgraph
        subgraph = nx.Graph(g.subgraph(block))
        # convert node labels to integer values from 0 to sz
        subgraph = nx.convert_node_labels_to_integers(subgraph)
        subgraphs.append(subgraph)
    return subgraphs

# release the median of X under eps-DP
def dp_median(X, eps):
    n = len(X)
    # compute ranks of all elements
    ranks = rankdata(X, method='min')
    # generate distribution to sample from (using quality of -|rank - n/2|)
    dist = np.array([np.exp(-1*eps*np.abs(r - (n/2))) for r in ranks])
    dist = dist/sum(dist)
    # randomly sample median
    return np.random.choice(X, size=1, p=dist)[0]

# sample and aggregate g, with eps-DP and using function f, aggregator agg
def sample_agg(g, eps, f, agg, k=None):
    subgraphs = partition(g)
    sample_conds = [f(sg) for sg in subgraphs]
    return agg(sample_conds, eps)