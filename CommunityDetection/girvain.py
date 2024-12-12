import numpy as np
import helper
     
def algorithm_level_one(adjlist, community_list = []):
    #One level of the Girvan Newman
    if community_list == []:
        community_list = np.zeros(len(adjlist))
    num_nodes = len(adjlist)
    run = True
    while run:
        edge = helper.max_betweeness_edge(adjlist)
        s, t = edge
        for i in range(len(adjlist)):
            adjlist[i] = list(adjlist[i])
        adjlist[s].remove(t)
        adjlist[t].remove(s)

        new_community = helper.new_community_list(adjlist)
        for i in range(num_nodes):
            if community_list[i] != new_community[i]:
                run = False
                break
    
    return new_community


def algorithm(adjlist, levels):
    #Algorithm for girvan newman
    num_nodes = len(adjlist)
    community_matrix = []
    current_community_list = [0]*num_nodes
    community_matrix.append(current_community_list)

    for i in range(levels - 1):
        new_communities = algorithm_level_one(adjlist, current_community_list)
        community_matrix.append(new_communities)
        current_community_list = new_communities
    
    return community_matrix[::-1]