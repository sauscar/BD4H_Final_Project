import pdb

import numpy as np
import pandas as pd
from scipy import sparse


def prepare(seqs, num_features):

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

