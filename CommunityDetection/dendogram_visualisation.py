import numpy as np
import matplotlib.pyplot as plt
import helper
from scipy.cluster.hierarchy import dendrogram

def visualise_dendogram(community_matrix, reordering_dict, kwargs = {}):
    n = len(community_matrix[0])
    min_number = n
    node_cluster_map = {i: i for i in range(n)}
    cluster_size_map = {i: 1 for i in range(n)}
    depth = 1
    Z = []
    for i in range(len(community_matrix)):
        for j in range(len(community_matrix[i])):
            value = community_matrix[i][j]
            key = helper.get_key_from_value(reordering_dict, value)
            community_matrix[i][j] = key

    # If at the start, all nodes aren't there in their own cluster, then make them
    for i in range(n):
        cluster1 = node_cluster_map[i]
        if node_cluster_map[community_matrix[0][i]] != cluster1:
            Z.append([cluster1, node_cluster_map[community_matrix[0][i]], depth, cluster_size_map[cluster1] + 1])
            node_cluster_map[community_matrix[0][i]] = min_number
            node_cluster_map[i] = min_number
            cluster_size_map[min_number] = cluster_size_map[cluster1] + 1
            del cluster_size_map[cluster1]
            min_number += 1

    depth += 1

    # Splitting the clusters as in the matrix
    for i in range(1, len(community_matrix)):
        for j in range(n):
            cluster1 = node_cluster_map[community_matrix[i - 1][j]]
            cluster2 = node_cluster_map[community_matrix[i][j]]
            if cluster1 != cluster2:
                Z.append([cluster1, cluster2, depth, cluster_size_map[cluster1] + cluster_size_map[cluster2]])
                node_cluster_map[community_matrix[i][j]] = min_number
                node_cluster_map[community_matrix[i-1][j]] = min_number
                cluster_size_map[min_number] = cluster_size_map[cluster1] + cluster_size_map[cluster2]
                del cluster_size_map[cluster1]
                del cluster_size_map[cluster2]
                min_number += 1
                break
        depth += 1 

    # If everything isn't combined into one cluster at the end, combine them
    unused = set(community_matrix[-1])
    while len(unused) > 1:
        node1 = unused.pop()
        node2 = unused.pop()
        cluster1 = node_cluster_map[node1]
        cluster2 = node_cluster_map[node2]
        Z.append([cluster1, cluster2, depth, cluster_size_map[cluster1] + cluster_size_map[cluster2]])
        node_cluster_map[node1] = min_unused_number
        node_cluster_map[node2] = min_unused_number
        cluster_size_map[min_unused_number] = cluster_size_map[cluster1] + cluster_size_map[cluster2]
        del cluster_size_map[cluster1]
        del cluster_size_map[cluster2]
        unused.add(node1)
        min_unused_number += 1


    dendrogram(np.array(Z, dtype=np.float64), **kwargs)
    plt.savefig()

'''
def visualise_dendogram(community_matrix, nodes, reordering_dict = None):
    #Visualises the dendogram for the communities obtained in Girvan_Newman
    if reordering_dict is not None:
        for i in range(len(community_matrix)):
            for j in range(len(community_matrix[i])):
                value = community_matrix[i][j]
                key = get_key_from_value(reordering_dict, value)
                community_matrix[i][j] = key
    else:
        # print(f'community_matrix = {community_matrix}', f'nodes = {nodes}')
        community_matrix, nodes = helper.ordering(community_matrix, nodes)
        # print(f'community_matrix = {community_matrix}, nodes = {nodes}')
        community_matrix = helper.community_prepare(community_matrix, nodes)
    # print(community_matrix)
    n = len(nodes) # Number of nodes
    # Conversion to linkage matrix
    linkage_matrix = []
    space = 0.1
    info = np.array([community_matrix[0], np.ones(n)])
    merged_cluster_ID = n
    merged = set() # Those nodes whose communities (which they were representatives of) which have been merged to another community with a lower Id
    for level, community_IDS in enumerate(community_matrix[1:], 1):
        for node, new_community_ID in enumerate(community_IDS):
            if (node not in merged) and new_community_ID != community_matrix[level - 1][node]:
                cluster_ID = info[0, node]
                new_cluster_ID = info[0, new_community_ID]
                nodes_merged = info[1, new_community_ID] + info[1, node]
                linkage_matrix.append([cluster_ID, new_cluster_ID, space, nodes_merged])
                merged.add(node)
                info[0, new_community_ID] = merged_cluster_ID
                merged_cluster_ID += 1 # Update the cluster next cluster Id to be assigned
                info[1, new_community_ID] = nodes_merged
        space += 0.1
    
    # if reordering_dict is not None:
    #     for node in community_matrix[0]:
    #         node = list(reordering_dict.keys())[list(reordering_dict.values()).index(node)]
    # print(np.shape(linkage_matrix),np.shape(community_matrix[0]))

    print(f'linkage_matrix = {linkage_matrix}')
    # print(np.shape(linkage_matrix),np.shape(community_matrix[0]))
    plt.figure(figsize =(30,15))
    plt.title('Dendogram')
    plt.xlabel('Community Id')
    plt.ylabel('Space')
    sch.dendrogram(linkage_matrix, leaf_rotation = 90, leaf_font_size = 8, labels=nodes, truncate_mode='level', p=5)
    plt.show()
'''

# # nodes_connectivity_list = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [0, 4], [2, 4], [5, 6], [6, 7], [7, 8], [5, 8], [5, 7], [6, 8], [9, 10], [10, 11], [9, 11], [8, 11], [1, 6]])
# # nodes_connectivity_list = np.array([[0,1],[0,2],[1,2],[2,3],[3,4],[4,5],[5,6],[4,6]])
# # nodes_connectivity_list = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [0, 4], [2, 4], [5, 6], [6, 7], [7, 8], [5, 8], [5, 7], [6, 8], [9, 10], [10, 11], [9, 11], [8, 11], [1, 6]])
# nodes_connectivity_list = np.array([[0, 1], [1, 2], [2,0], [4,2], [0,4], [2,3]])
# # nodes_connectivity_list = np.array([[0,2],[0,3],[0,5],[1,2],[1,4],[1,7],[2,5],[2,6],[2,4],[3,7],[4,0],[4,10],[5,7],[5,11],[6,7],[6,11],[8,9],[8,10],[8,11],[8,14],[8,15],[9,12],[9,14],[10,11],[10,12],[10,13],[10,14],[11,13]])
# # print('sjdfhsajodfh', helper.list_nodeIDS(helper.dict_nodeIDS(nodes_connectivity_list)))
# # community_matrix = louvain.algorithm(nodes_connectivity_list)
# # visualise_dendogram(np.array(community_matrix), helper.list_nodeIDS(helper.dict_nodeIDS(nodes_connectivity_list)))
# community2 = girvain.girvan_newman(helper.girvain_adjacency_list(nodes_connectivity_list, helper.reordering(helper.list_nodeIDS(helper.dict_nodeIDS(nodes_connectivity_list)))), 2)
# # print(f'community2 = {community2}')
# reorder = helper.reordering(helper.list_nodeIDS(helper.dict_nodeIDS(nodes_connectivity_list)))
# # print(f'reorder = {reorder}')
# visualise_dendogram(np.array(community2), reordering_dict=reorder)