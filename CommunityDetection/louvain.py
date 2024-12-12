import pandas as pd
import helper


def algorithm(nodes_connectivity_list):
    #Runs one iteration of the Louvain algorithm
    dict_nodeIDS = helper.dict_nodeIDS(nodes_connectivity_list)
    # list_nodeIDS = helper.list_nodeIDS(dict_nodeIDS)
    adjacency_list = helper.adjacency_list(nodes_connectivity_list)

    #We want to implement louvain algorithm
    #We have a dictionary of all the nodes and their corresponding community ids and they are already assigned to their own communities
    #We have a list of all the nodes in the order they appear in the dictionary
    #We have the nodes_connectivity list which has all unique edges in a list
    #We need a function to calculate modularity which is taken care of in helper
    #The communities follow a convention of following the smallest node id in that community, since we use a dictionary, I can start by referencing the dictionary directly, then moving on to a iterative search of the next nodes in the community right? No, use pandas to create a dataframe of the dictionary and then use the pandas functions to get the community ids of the nodes in the community
    #We can calculate the modularity of both the current community and the community after the node is moved to another community
    total_modularity = 1
    while total_modularity > 0:
        modularity_global = []
        for i in adjacency_list:
            node = i[0]
            modularity_list = []
            for j in i[1:]:
                if dict_nodeIDS[node] == dict_nodeIDS[j]:
                    pass
                else:
                    after = helper.modularity_community_add(node, dict_nodeIDS[j], nodes_connectivity_list, pd.DataFrame.from_dict(dict_nodeIDS, orient='index', columns=['Community']))
                    before = helper.modularity_community_subtract(node, dict_nodeIDS[node], nodes_connectivity_list, pd.DataFrame.from_dict(dict_nodeIDS, orient='index', columns=['Community']))
                    modularity = after + before
                    modularity_list.append([modularity, dict_nodeIDS[j]])
            if len(modularity_list) == 0:
                pass
            else:
                max_mod = max(modularity_list)[0]
                if max_mod > 0:
                    community = max(modularity_list)[1]
                    dict_nodeIDS = helper.community_changer((node, community), dict_nodeIDS, pd.DataFrame.from_dict(dict_nodeIDS, orient='index', columns=['Community']))
                else:
                    pass
                modularity_global.append(max_mod)
        if len(modularity_global) == 0:
            print('No more modularity to be gained')
            return helper.conversion_to_list(dict_nodeIDS)
        else:
            total_modularity = sum(modularity_global)
    return helper.conversion_to_list(dict_nodeIDS)


