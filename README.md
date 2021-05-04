# Sepsis Predictions for Hospital Admissions

*By Mohammed Almanassra, Bowen Chen, Erima Goyal, Oscar Parrilla*

### Overview

This project builds a machine learning pipeline that trains a recurrent neural network architecture with the MIMIC dataset provided in this [link](https://mimic.physionet.org/gettingstarted/access/). 

### Evironment Setup

The project is built in Anaconda Python 3.8.8, the dependencies are all outlined in the `environment.yml` file. To recreate the environment, run `conda env create -f environment.yml`

### Data Download

The dataset could be downloaded using the shell script named `extract_data.sh`. To get started, 
1. replace the `<CHANGE TO YOUR USER NAME>` to your MIMIC user name in line `wget -r -N -c -np --user <CHANGE TO YOUR USER NAME> --ask-password https://physionet.org/files/mimiciii/1.4/`
2. run `sh extract_data.sh` in your terminal

Then the data would be downloaded and extracted to `data/unzipped_files`

### Model Training

To train the model, in your terminal, 

1. Run `cd src`
2. Run `python main.py`

The whole pipeline will be completed in less than 5 minutes.

### Results

The model performed the best when setting the batch size to be 1 and traning for only 2 epochs. The best model achieved 54% in recall and 46% in precision. The confusion matrix is shown below

<img src="confusion_matrix_test_temp.png">


### Files Overview

The folder structure is the following

* `main.py` - main script that calls the etl pipeline, model training and model evaluations steps
* `train_model.py` - script that calls the dataloaders and training steps using the model defined in `model_definition/variable_rnn.py`
* `evaluate_model.py` - script that evaluates the model on the test set, plot the metrics
* `etl.py` - script that builds and loads the raw data set into PyTorch data loaders by calling `data_transformation/make_dataset.py`

* data_transform - data transformation scripts
* model_definition - variable RNN definition
* utils - utility functions
