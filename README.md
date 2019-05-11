# <p align=center>`SciCite`</p> 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/structural-scaffolds-for-citation-intent/citation-intent-classification-acl-arc)](https://paperswithcode.com/sota/citation-intent-classification-acl-arc?p=structural-scaffolds-for-citation-intent) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/structural-scaffolds-for-citation-intent/sentence-classification-acl-arc)](https://paperswithcode.com/sota/sentence-classification-acl-arc?p=structural-scaffolds-for-citation-intent) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/structural-scaffolds-for-citation-intent/citation-intent-classification-scicite)](https://paperswithcode.com/sota/citation-intent-classification-scicite?p=structural-scaffolds-for-citation-intent)

This repository contains datasets and code for classifying citation intents in academic papers.  
For details on the model and data refer to our NAACL 2019 paper:
["Structural Scaffolds for Citation Intent Classification in Scientific Publications"](https://arxiv.org/pdf/1904.01608.pdf).

## Data

We introduce `SciCite` a new large dataset of citation intents. Download from the following link:

[`scicite.tar.gz (22.1 MB)`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz)  


The data is in the Jsonlines format (each line is a json object).   
The main citation intent label for each Json object is spacified with the `label` key while the citation context is specified in with a `context` key.
Example entry:

```
{
 'string': 'In chacma baboons, male-infant relationships can be linked to both
    formation of friendships and paternity success [30,31].'
 'sectionName': 'Introduction',
 'label': 'background',
 'citingPaperId': '7a6b2d4b405439',
 'citedPaperId': '9d1abadc55b5e0',
 ...
 }
```

You may obtain the full information about the paper using the provided paper ids with the [Semantic Scholar API](https://api.semanticscholar.org/).

We also run experiments on a pre-existing dataset of citation intents in the computational linguistics domain (ACL-ARC) introduced by [Jurgens et al., (2018)](https://transacl.org/ojs/index.php/tacl/article/view/1266).
The preprocessed dataset is available at [`ACL-ARC data`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/acl-arc.tar.gz).

## Setup

The project needs Python 3.6 and is based on the [AllenNLP](https://github.com/allenai/allennlp) library.

#### Setup an environment manually

Use pip to install dependencies in your desired python environment

`pip install -r requirements.in -c constraints.txt`


## Running a pre-trained model on your own data

Download one of the pre-trained models and run the following command:

```bash
allennlp predict [path-to-model.tar.gz] [path-to-data.jsonl] \
--predictor [predictor-type] \
--include-package scicite \
--overrides "{'model':{'data_format':''}}"
```

Where 
* `[path-to-data.jsonl]` contains the data in the same format as the training data.
* `[path-to-model.tar.gz]` is the path to the pretrained model
* `[predictor-type]` is one of `predictor_scicite` (for the SciCite dataset format) or `predictor_aclarc` (for the ACL-ARC dataset format).
* `--output-file [out-path.jsonl]` is an optional argument showing the path to the output. If you don't pass this, the output will be printed in the stdout.

If you are using your own data, you need to first convert your data to be according to the SciCite data format.

#### Pretrained models

We also release our pretrained models; download from the following path:

* __[`SciCite`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/models/scicite.tar.gz)__
* __[`ACL-ARC`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/models/aclarc.tar.gz)__

## Training your own models

First you need a `config` file for your training configuration.
Check the `experiment_configs/` directory for example configurations.
Important options (you can specify them with environment variables) are:

```
  "train_data_path":  # path to training data,
  "validation_data_path":  #path to development data,
  "test_data_path":  # path to test data,
  "train_data_path_aux": # path to the data for section title scaffold,
  "train_data_path_aux2": # path to the data for citation worthiness scaffold,
  "mixing_ratio": # parameter \lambda_2 in the paper (sensitivity of loss to the first scaffold)
  "mixing_ratio2": # parameter \lambda_3 in the paper (sensitivity of loss to the second scaffold)
``` 

After downloading the data, edit the configuration file with the correct paths.
You also need to pass in an environment variable specifying whether to use [ELMo](https://allennlp.org/elmo) contextualized embeddings.

`export elmo=true`

Note that with elmo training speed will be significantly slower.

After making sure you have the correct configuration file, start training the model.

```
python scripts/train_local.py train_multitask_2 [path-to-config-file.json] \
-s [path-to-serialization-dir/] 
--include-package scicite
```

Where the model output and logs will be stored in `[path-to-serialization-dir/]`

## Citing

If you found our dataset, or code useful, please cite [Structural Scaffolds for Citation Intent Classification in Scientific Publications](https://arxiv.org/pdf/1904.01608.pdf).

```
@InProceedings{Cohan2019Structural,
  author={Arman Cohan and Waleed Ammar and Madeleine Van Zuylen and Field Cady},
  title={Structural Scaffolds for Citation Intent Classification in Scientific Publications},
  booktitle="NAACL",
  year="2019"
}
```
