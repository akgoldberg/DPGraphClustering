import os, itertools, csv

import networkx as nx
import numpy as np
from sklearn.cluster import spectral_clustering

# reweight the edges of a graph by the number of triangles each edge participates in
def reweightEdgesByTri(g):
    adjList = nx.to_dict_of_lists(g).values()
    adjListStr = "\n".join([",".join([str(e) for e in v]) for v in adjList])

    # write adjacency list to input file to triangle counter
    with open("input.grh", "w") as inFile:
        inFile.write(adjListStr)
    # count triangles using MACE
    os.system("../mace22/mace C -l 3 -u 3 input.grh tri.out")

    # initialize adjacency matrix
    n = g.number_of_nodes()
    M = np.zeros((n, n), dtype=int)

    with open("tri.out", "r") as triFile:
        triangles = csv.reader(triFile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        for t in triangles:
            for u, v in itertools.permutations(t, 2):
                M[int(u), int(v)] += 1

    return M

# generate a graph according to the planted partition model
def generatePlantedPartition(num, size, p_in, p_out, format="graph"):
    g = nx.planted_partition_graph(num, size, p_in, p_out)
    if format == "matrix":
        return nx.to_numpy_matrix(g)
    if format == "sparse_matrix":
        return nx.to_scipy_sparse_matrix(g)
    if format == "graph":
        return g

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
    return 1.0*cross/vol

def approxTriConductance(g, time=False):
    M = reweightEdgesByTri(g)
    C_tri = cluster(M)
    return conductance(C_tri, M)

def approxEdgeConductance(g, time=False):
    A = np.array(nx.to_numpy_matrix(g))
    C_edge = cluster(A)
    return conductance(C_edge, A)

