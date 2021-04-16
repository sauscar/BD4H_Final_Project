import pandas as pd
import numpy as np
from scipy import sparse
import pdb

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
    
    # listMatrix = []
    # emptyDF = pd.DataFrame()
    # # noOfVisits= len(seqs)
    # noOfVisits = 1
    # print(len(seqs))
    # print(len(labels))
    # for sublist in seqs:
    #     # coo_matrix = [0]*(num_features+1)
    #     # for i in sublist:
    #     #     coo_matrix[i] = 1
    #     # emptyDF.append(coo_matrix)

    #     # pdb.set_trace()  
    #     col_Index =  sublist
    #     data_set = [1]*len(col_Index)
    #     row_Index = [0]*len(sublist)
    
    #     # pdb.set_trace()
    #     coo_matrix = sparse.coo_matrix((data_set, (row_Index,col_Index)),shape=(noOfVisits, num_features+1)).toarray()
    #     listMatrix.append(coo_matrix)
    # # listMatrix = listMatrix.reshape(1,-1)
    # train_input = pd.DataFrame(listMatrix)



    # create tuple indices
    tuple_idx = [(i, j) for i in range(len(seqs)) for j in seqs[i]]
    # convert tuple indices, values to be all 1s
    row_idxs, col_idxs = zip(*tuple_idx)
    values = [1] * len(tuple_idx)
    # create sparse matrix, with shape to be (number of visits, number of features)
    patient_sparse = sparse.coo_matrix(
        (values, (row_idxs, col_idxs)), shape=(len(seqs), num_features),
    )

    # print(patient_sparse.todense()[:10, :10])
    print(patient_sparse.shape)


    # print(emptyDF.shape)
    # print(emptyDF.head())

    return patient_sparse
    

   
