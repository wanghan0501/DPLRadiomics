# Deep learning radiomics model related with genomics phenotypes for lymph node metastasis prediction in colorectal cancer

## AI Model
This folder includes the python codes of extracting DPL features by using `AutoEncoder`.

## How to use

### Preprocessing
During the training process, we use the input CT cube with the format of `npy`. We provide [an example](jupyter_notebooks/ct2npy.ipynb) of how to convert the input CT from `dicom` to `npy`.

### Train

Our code entry file is `crc_main.py`.

```
usage: crc_main.py [-h] [--seed SEED] [--use_cuda USE_CUDA] [--use_parallel USE_PARALLEL] [--gpu GPU] [--model {ae}] [--logdir LOGDIR] [--train_sample_csv TRAIN_SAMPLE_CSV]
                   [--eval_sample_csv EVAL_SAMPLE_CSV] [--config CONFIG]

PyTorch DCH_AI

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           random seed for training. default=42
  --use_cuda USE_CUDA   whether use cuda. default: true
  --use_parallel USE_PARALLEL
                        whether use parallel. default: false
  --gpu GPU             use gpu device. default: all
  --model {ae}          which model used. default: seg
  --logdir LOGDIR       which logdir used. default: None
  --train_sample_csv TRAIN_SAMPLE_CSV
                        train sample csv file used. default: None
  --eval_sample_csv EVAL_SAMPLE_CSV
                        eval sample csv file used. default: None
  --config CONFIG       configuration file. default: cfgs/crc_ae.yaml
```

### Extracting DPL features
Please use the notebook to extrace the DPL features. We provide [an example](jupyter_notebooks/test_3d_sample.ipynb) of how to extracing DPL features.


### Configuration
We use the yaml files to configure some parameters. The yaml files can be found in [cfgs](cfgs/) folder.
