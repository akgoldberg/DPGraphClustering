from graph_functions import *
from sample_aggregate import *
from util import *

# functions to time
generatePlantedPartition = timeit(generatePlantedPartition)
PIG = timeit(PIG)
EdgeFlipShrink = timeit(EdgeFlipShrink)
approxEdgeConductance_ut = approxEdgeConductance
approxEdgeConductance = timeit(approxEdgeConductance)
approxTriConductance_ut = approxTriConductance
approxTriConductance = timeit(approxTriConductance)
sample_agg = timeit(sample_agg)

'''
Test all methods on planted partition graph (with 2 clusters) and
params as specified.
'''


def test_all_pp(num_vertices, p_in, p_out, eps=0.5, k_sample=None):
    # generate planted partition graph
    g = generatePlantedPartition(2, num_vertices, p_in, p_out)
    # test all methods on this graph
    RES = test_all(g, eps, k_sample)
    return RES

'''
Test all methods on graph g with DP eps and number of sample k_sample for 
sample and aggregate
'''


def test_all(g, eps=0.5, k_sample=None, iters=1):
    # return dict
    RES = {
        'g': g,
    }
     # compute edge conductance on entire graph
    M, RES['edgeCond'] = approxEdgeConductance(g, ret_M=True)
    M_tri, RES['triCond'] = approxTriConductance(g, ret_M=True)

    for i in range(iters):
        RES[i] = {}
        print "\n*********ITER %d************" % i

        # generate synthetic graphs
        # g_PIG = PIG(g, eps)
        # g_EFS = EdgeFlipShrink(g, eps)
        # RES[i]['g_EFS'] = g_EFS

        # edge #
        # print "\n######### EDGE CONDUTANCE ##########"

        # compute edge conductance using sample and aggregate
        RES[i]['edgeCond_samp'] = sample_agg(
            g, eps, approxEdgeConductance_ut, dp_median, k=k_sample)
        # compute edge conductance on synthetic graph
        # print "EFS: "
        # RES[i]['edgeCond_EFS'] = approxEdgeConductance(g_EFS, orig=M)

        print ""

        # tri #
        # print "######### TRI CONDUTANCE ##########"

        # compute tri conductance using sample and aggregate
        RES[i]['triCond_samp'] = sample_agg(
            g, eps, approxTriConductance_ut, dp_median, k=k_sample)
        # compute tri conductance on synthetic graphs
        # print "EFS"
        # RES[i]['triCond_EFS'] = approxTriConductance(g_EFS, orig=M_tri)

    return RES

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

# run edge-flip-shrink on graph g with adj matrix m and triangle reweighted matrix m-tri
def run_EFS(g, M, M_tri, eps, iters=5):
    edgeConds = []
    triConds = []
    for i in range(iters):
        print "\n************ITER %d***************" % i
        g_EFS = EdgeFlipShrink(g, eps)
        edgeConds.append(approxEdgeConductance(g_EFS, orig=M))
        triConds.append(approxTriConductance(g_EFS, orig=M_tri))
    return edgeConds, triConds

