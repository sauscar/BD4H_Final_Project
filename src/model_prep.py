import pandas as pd
from scipy import sparse

def prepare(seqs,labels,num_features):
    # listMatrix = []
    # noOfVisits= len(seqs)
    # print(noOfVisits)
    # for sublist in seqs:
        # noOfVisits= len(sublist)
        # total_features = []
        # row_list = []
        # for j in range(len(sublist)):
        #     row_list += ([j] * len(sublist[j]))
        #     total_features += sublist[j]

        # col_Index =  total_features
        # row_Index = row_list
        # data_set = [1]*len(total_features)
        # coo_matrix = sparse.coo_matrix((data_set, (row_Index,col_Index)),shape=(noOfVisits, num_features)).toarray()
        # listMatrix.append(coo_matrix)
    
    listMatrix = []
    # noOfVisits= len(seqs)
    noOfVisits = 1
    print(len(seqs))
    print(len(labels))
    for sublist in seqs:
        
        col_Index =  sublist
        data_set = [1]*len(col_Index)
        row_Index = [0]*len(sublist)
    
        coo_matrix = sparse.coo_matrix((data_set, (row_Index,col_Index)),shape=(noOfVisits, num_features)).toarray()
        listMatrix.append(coo_matrix)
    print(listMatrix[0])
