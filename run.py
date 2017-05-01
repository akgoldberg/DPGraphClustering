from graph_functions import * 
from sample_aggregate import *

'''
Run edge conductance on planted partition graph and on
k samples of the graph and plot the results.
'''
def edge_sample_pp(num_vertices, p_in, p_out, k=None):
    # generate planted partition graph
    g = generatePlantedPartition(2, num_vertices, p_in, p_out)

    # compute conductance on entire graph
    edgeCond = approxEdgeConductance(g)

    # partition the graph
    samples = partition(g, k)
    # compute conductance on samples of graph
    cond_samples = [approxEdgeConductance(sg) for sg in samples]
    # plot the hist of the conductance of the samples
    hist_plot(cond_samples, edgeCond)

    return edgeCond, cond_samples

'''
Run triangle conductance on planted partition graph and on
k samples of the graph and plot the results.
'''
def tri_sample_pp(num_vertices, p_in, p_out, k=None):
    # generate planted partition graph
    g = generatePlantedPartition(2, num_vertices, p_in, p_out)

    # compute conductance on entire graph
    triCond = approxTriConductance(g)

    # partition the graph
    samples = partition(g, k)
    # compute conductance on samples of graph
    cond_samples = [approxTriConductance(sg) for sg in samples]
    # plot the hist of the conductance of the samples
    hist_plot(cond_samples, triCond)

    return triCond, cond_samples

""" Planted Partition Models to Try:
    - 2, 5000, 0.05, 0.01
    - 2, 10000, 0.02, 0.005
"""
