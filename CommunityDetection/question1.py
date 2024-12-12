#import wiki_vote_data
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import helper

def import_data(url):
    #Returns a numpy array of unique edges in the wiki-vote data
    if url == 'wiki-Vote.txt':
        sep = '\t'
    else:
        sep = ','
    df = pd.read_csv(url, sep=sep, header=None, names=['FromNodeId', 'ToNodeId'])
    # df = df.iloc[:][4:]
    # print(df.head(40))

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
    # print(connectivity_list)
    # print(len(connectivity_list))
    return connectivity_list

# c = import_wiki_vote_data(url = 'wiki-Vote.txt')
c = import_wiki_vote_data(url = 'lastfm_asia_edges.csv')
# # print(helper.conversion_to_dict(c))
# print(c)
# dict_nodeIDS = helper.dict_nodeIDS(c)
# # print(dict_nodeIDS)
# community_dataframe = pd.DataFrame.from_dict(dict_nodeIDS, orient='index', columns=['Community'])
# print(community_dataframe.head(40))
# print(community_dataframe[community_dataframe['Community'] == '30'])