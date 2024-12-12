import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import helper
import louvain
# import question1
import girvain

#Graph partition is a nx1 numpy array where each row is a node in the graph and the element is the community id of the node.
#As convention, we take the community id to be the smallest node id in that community
#Let us maintain a dictionary of all the nodeIds and their corresponding communityIds
#For this, we need a function that will obtain all the nodeIds from the list of unique edges that we have.
#The helper file has a function that does this.

# nodes_connectivity_list = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [0, 4], [2, 4], [5, 6], [6, 7], [7, 8], [5, 8], [5, 7], [6, 8], [9, 10], [10, 11], [9, 11], [8, 11], [1, 6]])
nodes_connectivity_list = np.array([[0,1],[0,2],[1,2],[2,3],[3,4],[4,5],[5,6],[4,6]])
# nodes_connectivity_list = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [0, 4], [2, 4], [5, 6], [6, 7], [7, 8], [5, 8], [5, 7], [6, 8], [9, 10], [10, 11], [9, 11], [8, 11], [1, 6]])
# nodes_connectivity_list = np.array([[0, 1], [1, 2], [2,0], [4,2], [0,4], [2,3]])
# nodes_connectivity_list = np.array([[0,2],[0,3],[0,5],[1,2],[1,4],[1,7],[2,5],[2,6],[2,4],[3,7],[4,0],[4,10],[5,7],[5,11],[6,7],[6,11],[8,9],[8,10],[8,11],[8,14],[8,15],[9,12],[9,14],[10,11],[10,12],[10,13],[10,14],[11,13]])

# nodes_connectivity_list = question1.import_wiki_vote_data(url = 'wiki-Vote.txt')


# print(nodes_connectivity_list)
# dict_nodeIDS = helper.dict_nodeIDS(nodes_connectivity_list)
# list_nodeIDS = helper.list_nodeIDS(dict_nodeIDS)
# community_dataframe = pd.DataFrame.from_dict(dict_nodeIDS, orient='index', columns=['Community'])
# print(helper.adjacency_list(nodes_connectivity_list))
print(louvain.algorithm(nodes_connectivity_list))
# print(helper.list_nodeIDS(helper.dict_nodeIDS(nodes_connectivity_list)))
# print(girvain.girvan_newman_level_one(helper.girvain_adjacency_list(nodes_connectivity_list, helper.reordering(helper.list_nodeIDS(helper.dict_nodeIDS(nodes_connectivity_list))))))
# print(girvain.algorithm(helper.girvain_adjacency_list(nodes_connectivity_list, helper.reordering(helper.list_nodeIDS(helper.dict_nodeIDS(nodes_connectivity_list)))), 3))