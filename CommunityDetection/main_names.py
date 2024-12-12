import helper
import girvain
import louvain

def import_wiki_vote_data(url):
    #Imports the wiki-vote data
    connectivity_list = helper.import_data(url)
    return connectivity_list

def import_lastfm_asia_data(url):
    #Imports the lastfm_asia data
    connectivity_list = helper.import_data(url)
    return connectivity_list

def Girvan_Newman_one_level(connectivity_list):
    #Executes one level of the Girvan-Newman algorithm
    community = girvain.algorithm_level_one(helper.girvain_adjacency_list(connectivity_list, helper.reordering(helper.list_nodeIDS(helper.dict_nodeIDS(connectivity_list)))))
    return community

def Girvan_Newman(connectivity_list):
    #Executes the Girvan-Newman algorithm
    community_matrix = girvain.algorithm(helper.girvain_adjacency_list(connectivity_list, helper.reordering(helper.list_nodeIDS(helper.dict_nodeIDS(connectivity_list)))), levels = 50)
    return community_matrix

def visualise_dendogram(community_matrix):
    #Visualises the dendogram for the communities obtained in Girvan_Newman
    '''Could not implement properly in time'''
    return None

def louvain_one_iter(connectivity_list):
    #Executes one iteration of the Louvain algorithm
    community = louvain.algorithm(connectivity_list)
    #Community List may not be in the increasing order of the node IDS
    return community