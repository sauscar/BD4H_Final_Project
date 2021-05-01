from data_transform.make_datasets import CreateDataset
from utils.utils import calculate_num_features

inp_folder = "../data/unzipped_files"


def build_data_loaders(batch_size):
    """ 
    Use CreateDataset object to build pytorch train, validation and test loader
    Input: None,
    Output: train_loader, val_loader, test_loader, num_features
    """
    # instantiate data
    dataset = CreateDataset(inp_folder, batch_size)

    # import all tables
    (df_icustays, _, df_microbiology, df_diagnosis, _, df_labevents) = dataset.import_tables()

    # roll up all events
    df_all_events_by_admission = dataset.generate_all_events_by_admission(
        df_microbiology, df_labevents, df_icustays
    )

    ### add sepis events
    df_all_events_by_admission_w_labels = dataset.generate_sepsis_event(
        df_all_events_by_admission, df_diagnosis
    )

    # train, validation and test split
    train_set, validation_set, test_set = dataset.train_validation_test_split(
        df_all_events_by_admission_w_labels
    )

    # generate training, validation and test sequence
    train_seqs = dataset.generate_sequence_data(train_set[0])
    val_seqs = dataset.generate_sequence_data(validation_set[0])
    test_seqs = dataset.generate_sequence_data(test_set[0])

    # generate train, validation and test labels
    train_labels = list(train_set[1].astype(int))
    val_labels = list(validation_set[1].astype(int))
    test_labels = list(test_set[1].astype(int))

    # number of features
    num_features = calculate_num_features(list(df_all_events_by_admission["FEATURE_ID"]))

    # generate torch dataset
    train_loader = dataset.generate_torch_dataset_loaders(train_seqs, train_labels, num_features)
    val_loader = dataset.generate_torch_dataset_loaders(val_seqs, val_labels, num_features)
    test_loader = dataset.generate_torch_dataset_loaders(test_seqs, test_labels, num_features)
    # import pdb
    # pdb.set_trace()
    return train_loader, val_loader, test_loader, num_features
