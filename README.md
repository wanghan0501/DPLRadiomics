# Deep learning radiomics model related with genomics phenotypes for lymph node metastasis prediction in colorectal cancer

## Description
This repository is the implementation of the paper " Deep learning radiomics model related with genomics phenotypes for lymph node metastasis prediction in colorectal cancer".

##  Contents
This repository contains the code about developing the model, assessing the performance of model, transcript analysis (GO and KEGG analysis), and the correlation analysis.

- model developed and validated.R: developed the model through `LASSO` and `5 fold cross validation`.

- Assessment of model.R: including the calibrate cure and decision curve analysis.

- Correlation analysis.R: including the correlation analysis of DPL features and genes, the correlation of DPL score and clinical characteristic. 

- Transcript analysis.R: including the differential expressed genes, GO analysis, and KEGG analysis.

- Univarible and multivariable analysis.R: including the univarible and multivariable analysis of DPL features and top differential expressed genes for predicting the LNM.

- AI Model: including the python codes of extracting DPL features by using AutoEncoder.


## Dataset
Due to privacy restrictions, we don’t upload our data used in the paper. If you are interested in our paper, please contact us.

## Cite
If you use our code, please cite:

```

```