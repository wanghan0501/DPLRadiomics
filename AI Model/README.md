# Deep learning radiomics model related with genomics phenotypes for lymph node metastasis prediction in colorectal cancer

## AI Model
This folder incldes the python codes of extracting DPL features by using `AutoEncoder`.

## How to use

### Preprocessing
During the training process, we use the input CT cube with the format of `npy`. We provide [an example](jupyter_notebooks/ct2npy.ipynb) of how to convert the input CT from `dicom` to `npy`.

### Train

Our code entry file is `crc_main.py`.

### Extracting DPL features
Please use the notebook to extracing the DPL features. We provide [an example](jupyter_notebooks/test_3d_sample.ipynb) of how to extracing DPL features.


### Configuration
We use the yaml files to configure some parameters. The yaml files can be found in [cfgs](cfgs/) folder.