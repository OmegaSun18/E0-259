import numpy as np
import pandas as pd
from collections import deque

# Description: This file contains helpful functions that are used in the main file.

def import_data(url):
    #Returns a numpy array of unique edges in the wiki-vote data
    if url == '../data/Wiki-Vote.txt':
        sep = '\t'
    else:
        sep = ','
    df = pd.read_csv(url, sep=sep, header=None, names=['FromNodeId', 'ToNodeId'])

    #Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_wiki = df.to_numpy()

    unique_nodes_connectivity_list_wiki = {}
    #Use a dictionary
    for element in nodes_connectivity_list_wiki:
        if element[0].isnumeric() == False or element[1].isnumeric() == False:
            continue
        #for all edges in list
        try:
            temp = unique_nodes_connectivity_list_wiki[element[1]]
            #See if the ToNodeId has been added to the dictionary
            if temp is None:
                #If not, add the edge FromNodeId to ToNodeId
                unique_nodes_connectivity_list_wiki[element[0]] = [element[1]]
            else:
                #If it has, check if the vertex FromNodeId is in the list of ToNodeId vertices
                if element[0] not in temp:
                    #If not, add edge FromNodeId to ToNodeId
                    temp = unique_nodes_connectivity_list_wiki[element[0]]
                    temp.append(element[1])
                    unique_nodes_connectivity_list_wiki[element[0]] = temp
                else:
                    pass
        except:
            #Add edge FromNodeId to ToNodeId
            try:
                temp = unique_nodes_connectivity_list_wiki[element[0]]
                temp.append(element[1])
                unique_nodes_connectivity_list_wiki[element[0]] = temp
            except:
                unique_nodes_connectivity_list_wiki[element[0]] = [element[1]]
        
    final_nodes_connectivity_list_wiki = []
    #Convertion of dictionary to list
    for element in unique_nodes_connectivity_list_wiki.keys():
        for value in unique_nodes_connectivity_list_wiki[element]:
            final_nodes_connectivity_list_wiki.append([element, value])

    #Final numpy list.
    connectivity_list = np.array(final_nodes_connectivity_list_wiki)

    return connectivity_list

def conversion_to_dict(array):
    #Converts a numpy array to a dictionary
    dict = {}
    for element in array:
        try:
            temp = dict[element[0]]
            temp.append(element[1])
            dict[element[0]] = temp
        except:
            dict[element[0]] = [element[1]]

    return dict

def conversion_to_list(dict):
    #Converts a dictionary to a list
    list = []
    for key in dict.keys():
        list.append(dict[key])

    return np.array(list)
            

def dict_nodeIDS(array):
    #Creates a dictionary of all the NodeIDS and their corresponding communityIDs
    # >>> dict_nodeIDS(np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]))
    # {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}
    dict = {}
    for element in array:
        if element[0] in dict.keys():
            pass
        else:
            dict[element[0]] = element[0]
        if element[1] in dict.keys():
            pass
        else:
            dict[element[1]] = element[1]

    return dict

def list_nodeIDS(dict):
    #Creates a list of all the NodeIDS
    # >>> list_nodeIDS({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11})
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    list = []
    for key in dict.keys():
        list.append(key)

    return list

def adjacency_list(array):
    #Creates an adjacency list from a numpy array
    # >>> adjacency_list(np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]))
    adjacency_dict = {}
    for element in array:
        try:
            temp = adjacency_dict[element[0]]
            temp.append(element[1])
            adjacency_dict[element[0]] = temp
        except:
            adjacency_dict[element[0]] = [element[1]]
        try:
            temp = adjacency_dict[element[1]]
            temp.append(element[0])
            adjacency_dict[element[1]] = temp
        except:
            adjacency_dict[element[1]] = [element[0]]
    adjacency_list = []
    for key in adjacency_dict.keys():
        temp = [key]
        temp.extend(adjacency_dict[key])
        adjacency_list.append(temp)

    return adjacency_list
    

def degree_of_node(i, nodes_connectivity_list):
    #Calculates the degree of a node
    # >>> degree_of_node(1, np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]))
    # 1
    degree = 0
    for element in nodes_connectivity_list:
        if element[0] == i:
            degree += 1
        if element[1] == i:
            degree += 1

    return degree

#NOT USED
def modularity_node(i, nodes_connectivity_list):
    #We calculate the modularity of one node
    #A_ii term is 0
    # >>> modularity_node(1, np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]))
    # -0.05
    m = len(nodes_connectivity_list)
    k = degree_of_node(i, nodes_connectivity_list)

    return - (k**2) / 2*m

def summation_i(edge_dict, community_nodes):
    #Calculates the A_i term
    if len(community_nodes) == 1:
        #Returns 0 if there is only one node in the community
        return 0
    A_i = 0
    for node in edge_dict.keys():
        if node in community_nodes:
            for element in edge_dict[node]:
                if element in community_nodes:
                    A_i += 1
                else:
                    pass
        else:
            pass
    A_i *= 2

    return A_i

def degree_of_community(community_nodes, nodes_connectivity_list):
    #Calculates the degree of a community
    k_i = 0
    for node in community_nodes:
        k_i += degree_of_node(node, nodes_connectivity_list)

    return k_i

def node_to_community(i, community_nodes, edge_dict):
    #Calculates the number of edges from i to nodes in the community
    k_i_in = 0
    if i in edge_dict.keys():
        for element in edge_dict[i]:
            if element in community_nodes:
                k_i_in += 1
            else:
                pass
    else:
        for node in community_nodes:
            if node in edge_dict.keys():
                if i in edge_dict[node]:
                    k_i_in += 1
                else:
                    pass
            else:
                pass

    return k_i_in

def modularity_community_add(i, j, nodes_connectivity_list, community_dataframe):
    #Calculates the modularity of a community when a node is added to it
    #i is added to j
    m = len(nodes_connectivity_list)
    community_nodes = community_dataframe[community_dataframe['Community'] == j].index.tolist()
    edge_dict = conversion_to_dict(nodes_connectivity_list)
    community_edges = summation_i(edge_dict, community_nodes)
    community_degree = degree_of_community(community_nodes, nodes_connectivity_list)
    k_i = degree_of_node(i, nodes_connectivity_list)
    k_i_in = node_to_community(i, community_nodes, edge_dict)

    return ((2*k_i_in) / (2*m)) - ((community_degree + k_i) / (2*m))**2 + (community_degree / (2*m))**2 + (k_i / (2*m))**2 
    

def community_edges_subtract(i, edge_dict, community_nodes):
    #Calculates the number of edges in a community when a node is removed from it
    A_i = 0
    for node in edge_dict.keys():
        if node != i and node in community_nodes:
            for element in edge_dict[node]:
                if element != i and element in community_nodes:
                    A_i += 1
                else:
                    pass
        else:
            pass
    A_i *= 2

    return A_i

#NOT NEEDED
def degree_of_node_subract(i, node, nodes_connectivity_list):
    #Calculates the degree of a node while ignoring a specific node i
    degree = 0
    for element in nodes_connectivity_list:
        if element[0] == node and element[1] != i:
            degree += 1
        if element[1] == node and element[0] != i:
            degree += 1
    return degree

def community_degree_subtract(i, community_nodes, nodes_connectivity_list):
    #Calculates the degree of a community when a node is removed from it
    k_i = 0
    for node in community_nodes:
        if node != i:
            k_i += degree_of_node(node, nodes_connectivity_list)
        else:
            pass

    return k_i

def modularity_community_subtract(i, j, nodes_connectivity_list, community_dataframe):
    #Calculates the modularity of a community when a node is removed from it
    #i is removed from j
    m = len(nodes_connectivity_list)
    community_nodes = community_dataframe[community_dataframe['Community'] == j].index.tolist()
    edge_dict = conversion_to_dict(nodes_connectivity_list)
    community_edges = summation_i(edge_dict, community_nodes)
    community_degree = degree_of_community(community_nodes, nodes_connectivity_list)
    new_community_edges = community_edges_subtract(i, edge_dict, community_nodes)
    new_community_degree = community_degree_subtract(i, community_nodes, nodes_connectivity_list)
    k_i = degree_of_node(i, nodes_connectivity_list)

    return ((new_community_edges)/ (2*m)) - ((new_community_degree) / (2*m))**2 - (k_i / (2*m))**2 - ((community_edges) / (2*m)) + ((community_degree) / (2*m))**2   

def community_changer(max_mod_pair, dict_nodeIDS, community_dataframe):
    #Changes the community of a node, Update the community ids of all other things as well
    #Communities follow the convention that the id is the same id of the samllest node in the community

    old_community_id = dict_nodeIDS[max_mod_pair[0]]
    if max_mod_pair[0] >= max_mod_pair[1]:
        #No need to change community ids of Community Y
        dict_nodeIDS[max_mod_pair[0]] = max_mod_pair[1] 
        if old_community_id == max_mod_pair[0]:
            #Need to change community nodes of Community X
            old_community_nodes = community_dataframe[community_dataframe['Community'] == old_community_id].index.tolist()
            old_community_nodes.remove(max_mod_pair[0])
            if old_community_nodes == []:
                #This should not throw an error as there are no nodes in the community
                pass
            else:
                new_id = min(old_community_nodes)
                for key in dict_nodeIDS.keys():
                    if key != max_mod_pair[0] and dict_nodeIDS[key] == old_community_id:
                        dict_nodeIDS[key] = new_id
                    else:
                        pass
        else:
            #No nodes in Community X need to be changed
            pass
    else:
        #Need to change community ids of Community Y
        dict_nodeIDS[max_mod_pair[0]] = max_mod_pair[0]
        if old_community_id == max_mod_pair[0]:
            #Need to change community nodes of Community X
            old_community_nodes = community_dataframe[community_dataframe['Community'] == old_community_id].index.tolist()
            old_community_nodes.remove(max_mod_pair[0])
            if old_community_nodes == []:
                for key in dict_nodeIDS.keys():
                    if dict_nodeIDS[key] == max_mod_pair[1]:
                        dict_nodeIDS[key] = max_mod_pair[0]
                    else:
                        pass                
            else:
                new_id = min(old_community_nodes)                
                for key in dict_nodeIDS.keys():
                    if key != max_mod_pair[0] and dict_nodeIDS[key] == old_community_id:
                        #If there are no more nodes in the old community, then this block is not executed.
                        dict_nodeIDS[key] = new_id
                    else:
                        pass
                    if dict_nodeIDS[key] == max_mod_pair[1]:
                        dict_nodeIDS[key] = max_mod_pair[0]
                    else:
                        pass
        else:
            #No need to change community nodes of Community X
            for key in dict_nodeIDS.keys():
                if dict_nodeIDS[key] == max_mod_pair[1]:
                    dict_nodeIDS[key] = max_mod_pair[0]
                else:
                    pass

    return dict_nodeIDS

def community_prepare(community_matrix, nodes):
    #Preparing the community matrix for the dendogram function
    community_matrix = community_matrix.reshape(-1,1)
    community_matrix = np.hstack((np.zeros((community_matrix.shape[0], 1)), community_matrix))
    node_2d = np.array(nodes).reshape(-1, 1)
    community_matrix = np.fliplr(np.hstack((community_matrix, node_2d))).T
    community_matrix = community_matrix.astype(int)

    return community_matrix

def ordering(community_matrix, nodes):
    #Order the community matrix, the community_matrix is a 1D numpy array
    c = []
    sorted_nodes = sorted(nodes)
    for i in sorted_nodes:
        c.append(community_matrix[nodes.index(i)])

    return np.array(c), sorted_nodes    

def reordering(list_nodeIDS):
    #Renames the list of nodes as 0 to len(list_nodeIDS) - 1
    #And keeps track of the changes in a dictionary
    dict = {}
    for i in range(len(list_nodeIDS)):
        dict[list_nodeIDS[i]] = i

    return dict

def girvain_adjacency_list(nodes_connectivity_list, reordering_dict):
    #Creates an adjacency list from a numpy array
    # >>> adjacency_list(np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]))
    # >>> [[2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9], [8, 10], [9, 11], [10]]
    adjacency_dict = {}
    for element in nodes_connectivity_list:
        try:
            temp = adjacency_dict[element[0]]
            temp.append(element[1])
            adjacency_dict[element[0]] = temp
        except:
            adjacency_dict[element[0]] = [element[1]]
        try:
            temp = adjacency_dict[element[1]]
            temp.append(element[0])
            adjacency_dict[element[1]] = temp
        except:
            adjacency_dict[element[1]] = [element[0]]
    adjacency_list = [0] * len(adjacency_dict)
    for key in adjacency_dict.keys():
        new_key = reordering_dict[key]
        nodes = adjacency_dict[key]
        for i in range(len(nodes)):
            nodes[i] = reordering_dict[nodes[i]]
        adjacency_list[new_key] = nodes

    return adjacency_list

def number_of_shortest_paths(root, adjlist):
    #BFS to get the number of shortest paths from root to all other nodes
    num_nodes = len(adjlist)
    depth = [-1] * num_nodes
    queue = deque()
    queue.append(root)
    shortest_paths = [0] * num_nodes
    shortest_paths[root] = 1
    depth[root] = 0
    next_level = set()

    while len(queue):
        node = queue.popleft()
        for child in adjlist[node]:
            if depth[child] == -1:
                next_level.add(child)
                shortest_paths[child] += shortest_paths[node]

        if not len(queue):
            for x in next_level:
                queue.append(x)
                depth[x] = depth[node] + 1

            next_level.clear()
    
    return shortest_paths, depth

def credit_contribution_dfs(root, adjlist, depth, node_credits, edge_credits, shortest_paths):
    #DFS to compute the credit contribution of each edge
    for child in adjlist[root]:
        if depth[child] > depth[root]:
            credit_contribution_dfs(child, adjlist, depth, node_credits, edge_credits, shortest_paths)
            edge_credits[(root, child)] = node_credits[child] * (shortest_paths[root] / shortest_paths[child])
            node_credits[root] += edge_credits[(root, child)]

def credit_contribution(root, adjlist):
    #Computes the credit contribution of each edge
    num_nodes = len(adjlist) 
    shortest_paths, depth = number_of_shortest_paths(root, adjlist)
    node_credits = {i:1 for i in range(num_nodes)}
    edge_credits = {}

    credit_contribution_dfs(root, adjlist, depth, node_credits, edge_credits, shortest_paths)

    return edge_credits

def max_betweeness_edge(adjlist):
    #Computes the max betweeness edge
    num_nodes = len(adjlist)
    cumulative_edge_credits = {}

    for i in range(num_nodes):
        edge_credit_contribution = credit_contribution(i, adjlist)
        for edge, credit in edge_credit_contribution.items():
            if edge in cumulative_edge_credits:
                cumulative_edge_credits[edge] += credit
            elif (edge[1], edge[0]) in cumulative_edge_credits:
                cumulative_edge_credits[(edge[1], edge[0])] += credit
            else:
                cumulative_edge_credits[edge] = credit

    max_edge_credit = -1
    max_credit_edge = None

    for edge, credit in cumulative_edge_credits.items():
        if credit > max_edge_credit:
            max_edge_credit = credit
            max_credit_edge = edge

    return max_credit_edge

def new_community_list(adjlist):
    #Creates a new community list from an adjecency list
    num_nodes = len(adjlist)
    queue = deque()
    current_community = [-1] * num_nodes

    for i in range(num_nodes): 
        if current_community[i] == -1:
            current_community[i] = i
            queue.append(i)
            while len(queue):
                node = queue.popleft()
                for child in adjlist[node]:
                    if current_community[child] == -1:
                        queue.append(child)
                        current_community[child] = i

    return current_community

def get_key_from_value(dictionary, value):
    # Getting the mapped key value from a dictionary
    for key, val in dictionary.items():
        if val == value:
            return key
        
    return None