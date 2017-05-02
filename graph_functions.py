import os
import itertools
import csv

import networkx as nx
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn import random_projection


#######################################
##          GRAPH GENERATORS         ##
#######################################


# generate a graph according to the planted partition model
def generatePlantedPartition(num, size, p_in, p_out, fmt="graph"):
    g = nx.planted_partition_graph(num, size, p_in, p_out)
    if fmt == "matrix":
        return nx.to_numpy_matrix(g)
    if fmt == "sparse_matrix":
        return nx.to_scipy_sparse_matrix(g)
    if fmt == "graph":
        return g

#######################################
## EDGE/TRIANGLE SPECTRAL CLUSTERING ##
#######################################

# reweight the edges of a graph by the number of triangles each edge
# participates in


def reweightEdgesByTri(g):
    adjList = nx.to_dict_of_lists(g).values()
    adjListStr = "\n".join([",".join([str(e) for e in v]) for v in adjList])

    # write adjacency list to input file to triangle counter
    with open("input.grh", "w") as inFile:
        inFile.write(adjListStr)
    # count triangles using MACE
    os.system("../mace22/mace Cq -l 3 -u 3 input.grh tri.out")

    # initialize adjacency matrix
    n = g.number_of_nodes()
    M = np.zeros((n, n), dtype=int)

    with open("tri.out", "r") as triFile:
        triangles = csv.reader(triFile, delimiter=' ',
                               quoting=csv.QUOTE_NONNUMERIC)
        for t in triangles:
            for u, v in itertools.permutations(t, 2):
                M[int(u), int(v)] += 1

    return M

# cluster adjacency matrix using spectral clustering


def cluster(M):
    return spectral_clustering(M, n_clusters=2)

# take in a clustering into two clusters and return the conductance


def conductance(C, M):
    # get the clusters
    c1_ind = np.where(C)[0]
    c2_ind = np.where(C - 1)[0]
    c1 = M[c1_ind]
    c2 = M[c2_ind]
    # total weight crossing the cut
    cross = c1.transpose()[c2_ind].sum()
    # total weight within the cut
    vol = min(c1.transpose()[c1_ind].sum(), c2.transpose()[c2_ind].sum())
    return 1.0 * cross / vol


def approxTriConductance(g, orig=None, ret_M=False):
    M = reweightEdgesByTri(g)
    C_tri = cluster(M)
    O = orig if np.any(orig) else M
    cond = conductance(C_tri, O)
    if ret_M:
        return M, cond
    return cond


def approxEdgeConductance(g, orig=None, ret_M=False):
    A = np.array(nx.to_numpy_matrix(g))
    C_edge = cluster(A)
    O = orig if np.any(orig) else A
    cond = conductance(C_edge, O)
    if ret_M:
        return A, cond
    return cond

################################
## GRAPH PERTURBATION METHODS ##
################################


def PIG(g, eps):
    A = nx.to_numpy_matrix(g)
    # flip probability
    s = 2.0 / (np.exp(eps) + 1)
    for i in range(len(A)):
        for j in range(len(A)):
            # randomize with prob s
            if np.random.rand() < s:
                A[i, j] = A[j, i] = int(np.random.rand() < 0.5)
    return nx.Graph(A)


def EdgeFlipShrink(g, eps):
    n = g.number_of_nodes()
    # initialize new adjacency matrix
    ghat = np.zeros((n, n))
    eps2 = 0.1
    # noisy count of number of edges
    m_dp = g.number_of_edges() + np.random.laplace(1.0 / eps2)
    # flip probability
    s = 2 / (np.exp(eps - eps2) + 1)
    m0 = ((1 - s) * m_dp) + (n * (n - 1) * s / 4)
    p = m_dp / m0
    # process 1-edges
    n1 = 0
    for (i, j) in g.edges():
        e = int(np.random.rand() < p * (1 - s) / 2)
        ghat[i, j] = ghat[j, i] = e
        n1 += e
    n0 = m_dp - n1
    # process 0-edges
    while n0 > 0:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if not g.has_edge(i, j) and ghat[i, j] == 0:
            ghat[i, j] = ghat[j, i] = 1
            n0 -= 1
    return nx.Graph(ghat)

################################
## GAUSSIAN PROJECTION METHOD ##
################################
# Paper's approach does not make sense


def spectralPrivate(g, eps, delta, m=200):
    n = g.number_of_nodes()
    A = nx.to_numpy_matrix(g)
    sigma = (1.0 / eps) * np.sqrt(10 *
                                  (eps + np.log(1 / (2 * delta))) * np.log(n / delta))
    # project matrix into lower dimension using random gaussian
    P = random_projection.GaussianRandomProjection(eps=0.5)
    A_p = P.fit_transform(A)
    print A_p.shape
    # add noise
    Q = np.random.normal(0, sigma, A_p.shape)
    A_p += Q
    return cluster(Q)
