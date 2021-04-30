import itertools

import numpy as np
import pandas as pd
import torch
from scipy import sparse


def read_table(inp_folder, filename):
    path = inp_folder + "/" + filename
    df = pd.read_csv(path)
    print("******", filename)
    print("TOTAL RECORDS in ", df.shape)
    return df


def read_table_spark(spark_session, inp_folder, filename, cols=None):
    path = inp_folder + "/" + filename
    spark_df = spark_session.read.csv(path, header=True)
    if cols:
        spark_df = spark_df.select(*cols)

    print("******", filename)
    print(f"TOTAL RECORDS ({spark_df.count()}, {len(spark_df.columns)})")

    return spark_df


def build_codemap(feature_ids):
    """
	:return: Dict of code map {Feature ID: unique feature ID}
	"""
    # apply the transform to get the desired codes
    feature_ids_unique = feature_ids.dropna().unique()
    # create code mapping
    codemap = {code: idx for idx, code in enumerate(feature_ids_unique)}
    return codemap


def create_sequence_data(seqs, num_features):
    # create tuple indices
    tuple_idx = [(i, j) for i in range(len(seqs)) for j in seqs[i]]

    # convert tuple indices, values to be all 1s
    row_idxs, col_idxs = zip(*tuple_idx)
    values = [1] * len(tuple_idx)

    # create sparse matrix, with shape to be (number of visits, number of features)
    patient_sparse = sparse.coo_matrix(
        (values, (row_idxs, col_idxs)), shape=(len(seqs), num_features),
    )

    return patient_sparse


def calculate_num_features(seqs):
    """
	:param seqs:
	:return: the calculated number of features
	"""
    # flatten the list twice to get the max index + 1
    num_features = max(list(itertools.chain(*itertools.chain(*seqs)))) + 1

    return num_features


def pad_with(vector, pad_width, iaxis, kwargs):
    """ 
        From np.pad function, created to pad 1 dimension
        Referenced from Stack Overflow 
        https://stackoverflow.com/questions/59093533/how-to-pad-an-array-non-symmetrically-e-g-only-from-one-side
    """
    pad_value = kwargs.get("padder", 0)
    vector[: pad_width[0]] = pad_value
    if pad_width[1] != 0:  # 0 indicates no padding
        vector[-pad_width[1] :] = pad_value


def event_collate_fn(batch):
    """
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

    # sort the bactch list by lengths of visit for each patient, descending
    batch_sorted = sorted(batch, key=lambda visit_tuple: visit_tuple[0].shape[0], reverse=True)

    # collect seqs, lengths and labels in 3 different tuples
    seqs, lengths, labels = zip(
        *[(seq.toarray(), seq.shape[0], label) for (seq, label) in batch_sorted]
    )
    # pad 0s to the desired shape (max rows, number of features)
    max_length = lengths[0]
    padded_seqs = [np.pad(seq, ((0, max_length - seq.shape[0]), (0, 0)), pad_with) for seq in seqs]

    # convert to tenssors
    seqs_tensor = torch.FloatTensor(padded_seqs)
    lengths_tensor = torch.LongTensor(lengths)
    labels_tensor = torch.LongTensor(labels)

    return (seqs_tensor, lengths_tensor), labels_tensor
