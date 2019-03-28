# `SciCite`

This repository contains datasets and code for classifying citation intents in academic papers.  
For details on the model and data refer to our NAACL 2019 paper: ["Structural Scaffolds for Citation Intent Classification in Scientific Publications"](https://arxiv.org/).

### Data

We introduce `SciCite` a new large dataset of citation intents.
You can download the data from:  
  
* __[`SciCite`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz)__

We also run experiments on the ACL-ARC Citations dataset introduced by [Jurgens et al., (2018)](https://transacl.org/ojs/index.php/tacl/article/view/1266).
The pre-processed dataset is available at:

* __[`ACL-ARC`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/acl-arc/acl-arc.tar.gz)__

The format of the datasets are identical and in Jsonlines format (each line is a json object).

### Setup

The project needs Python 3.6 and is based on the [AllenNLP](https://github.com/allenai/allennlp) library.

#### Setup an environment manually

Use pip to install dependencies in your desired python environment

`pip install -r requirements.in -c constraints.txt`

#### Setup an environment using the provided conda scrip

`./setup_env.sh`

python citation_intent/run_model.py predict saved-model/elmo_model.tar.gz data/s2/v2/v03-nested/crowd_test.jsonl --output-file /tmp/predict-out-crowd_test-elmo2.txt --include-package citation_intent --predictor citation_classifier --cuda-device 0 --overrides "{model: {predict_mode:true}, dataset_reader: {reader_format: 'nested'}}"



### Running a pre-trained model on your own data

Download one of the pre-trained models and run the following command:

`allennlp predict [path-to-data.jsonl] [path-to-model.tar.gz] --predictor predictor_aclarc --include-package scicite  --output-file [out-path.jsonl]`

Where 
* `[path-to-data.jsonl]` contains the data in the same format as the training data.
* `[path-to-model.tar.gz]` is the path to the pretrained model
* `--output-file [out-path.jsonl]` is an optional argument showing the path to the output. If you don't pass this, the output will be printed in the stdout.

#### Pretrained models

Download from the following path:

* __[`SciCite`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/models/scicite.tar.gz)__
* __[`ACL-ARC`](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/models/aclarc.tar.gz)__

### Training your own models

First you need a `config` file for your training configuration.
Check the `experiment_configs directory/` for example configurations.
Important options are:

```
  "train_data_path":  # path to training data,
  "validation_data_path":  #path to development data,
  "test_data_path":  #path to test data,
  "train_data_path_aux": # path to the data for section title scaffold,
  "train_data_path_aux2": # path to the data for citation worthiness scaffold,
  "mixing_ratio": parameter \lambda_1 in the paper (sensitivity of loss to the first scaffold)
  "mixing_ratio2": parameter \lambda_2 in the paper (sensitivity of loss to the second scaffold)
``` 

After downloading the data, edit the configuration file with the correct paths.
You also need to pass in an environment variable specifying whether to use `[ELMO](https://allennlp.org/elmo)` or not.

`export elmo=true`

Note that with elmo training speed will be significantly slower.

After making sure you have the correct configuration file, start training the model.

```
python citation_intent/run_model.py train_multitask_2 [path-to-config-file.json] \
-s [path-to-serialization-dir/] 
--include-package scicite
```

Where the model output and logs will be stored in `[path-to-serialization-dir/]`

## Citing

Please cite our NAACL 2019 paper:

```
@InProceedings{Cohan2019Structural,
  author={Arman Cohan and Waleed Ammar and Madeleine Van Zuylen and Field Cady},
  title={{Structural Scaffolds for Citation Intent Classification in Scientific Publications}},
  booktitle="NAACL",
  year="2019"
}
```
