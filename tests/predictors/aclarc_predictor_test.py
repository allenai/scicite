# pylint: disable=no-self-use,invalid-name,unused-import
import tempfile
from pathlib import Path
from unittest import TestCase

from allennlp.common import Params
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import Model
from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

import sys
sys.path.append(str(Path('.').absolute()))

from scicite.training.train_multitask_two_tasks import train_model_from_file
from scicite.constants import root_path

sys.path.append(root_path)

from scicite.models.scaffold_bilstm_attention_classifier import ScaffoldBilstmAttentionClassifier
from scicite.dataset_readers.citation_data_reader_scicite import SciciteDatasetReader
from scicite.dataset_readers.citation_data_reader_scicite_aux import SciciteSectitleDatasetReader, SciCiteWorthinessDataReader
from scicite.predictors.predictor_scicite import PredictorSciCite
from scicite.dataset_readers.citation_data_reader_aclarc import AclarcDatasetReader
from scicite.dataset_readers.citation_data_reader_aclarc_aux import AclSectionTitleDatasetReader, AclCiteWorthinessDatasetReader
from scicite.predictors.predictor_acl_arc import CitationIntentPredictorACL

# required so that our custom model + predictor + dataset reader
# will be registered by name

class TestCitationClassifierPredictor(TestCase):
    def test_uses_named_inputs(self):
        input0 = {"text": "More recently , ( Sebastiani , 2002 ) has performed a good survey of "
                          "document categorization ; recent works can also be found in ( Joachims , 2002 ) "
                          ", ( Crammer and Singer , 2003 ) , and ( Lewis et al. , 2004 ) .",
                  "citing_paper_id": "W04-1610",
                  "cited_paper_id": "External_9370",
                  "citing_paper_year": 2004, "cited_paper_year": 2002, "citing_paper_title":
                      "a holistic lexiconbased approach to opinion mining",
                  "cited_paper_title": "learning to classify text using svm",
                  "cited_author_ids": ["T Joachims"],
                  "citation_id": "W04-1610_5"}
        
        input1 = {"text": "We use method introduced by ( Sebastiani , 2002).",
                  "citing_paper_id": "W04-1610",
                  "cited_paper_id": "External_9370",
                  "citing_paper_year": 2004, "cited_paper_year": 2002, "citing_paper_title":
                      "a holistic lexiconbased approach to opinion mining",
                  "cited_paper_title": "learning to classify text using svm",
                  "cited_author_ids": ["T Joachims"],
                  "citation_id": "W04-1610_5"}

        param_file = 'tests/fixtures/test-config-aclarc.json'
        dataset_file = 'tests/fixtures/aclarc-train.jsonl'

        self.param_file = param_file
        params = Params.from_file(self.param_file)

        reader = DatasetReader.from_params(params['dataset_reader'])
        # The dataset reader might be lazy, but a lazy list here breaks some of our tests.
        instances = list(reader.read(str(dataset_file)))
        # Use parameters for vocabulary if they are present in the config file, so that choices like
        # "non_padded_namespaces", "min_count" etc. can be set if needed.
        if 'vocabulary' in params:
            vocab_params = params['vocabulary']
            vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
        else:
            vocab = Vocabulary.from_instances(instances)
        self.vocab = vocab
        self.instances = instances
        model = Model.from_params(vocab=self.vocab, params=params['model'])
        model.predict_mode = True

        predictor = Predictor.by_name("predictor_aclarc")(model=model, dataset_reader=reader)

        result = predictor.predict_json(input0)
        prediction = result['prediction']
        assert prediction in {'Background', 'CompareOrContrast', 'Extends', 'Future', 'Motivation', 'Uses'}

        result = predictor.predict_json(input1)
        prediction = result['prediction']
        assert prediction in {'Background', 'CompareOrContrast', 'Extends', 'Future', 'Motivation', 'Uses'}

        input0 = {"source": "explicit", "citeEnd": 68.0, "sectionName": "Discussion", "citeStart": 64.0,
                  "string": "These results are in contrast with the findings of Santos et al.(16), who reported a significant association between low sedentary time and healthy CVF among Portuguese",
                  "label": "result",
                  "label2": "supportive",
                  "citingPaperId": "8f1fbe460a901d994e9b81d69f77bfbe32719f4c", "citedPaperId": "5e413c7872f5df231bf4a4f694504384560e98ca",
                  "id": "8f1fbe460a901d994e9b81d69f77bfbe32719f4c>5e413c7872f5df231bf4a4f694504384560e98ca",
                  "unique_id": "8f1fbe460a901d994e9b81d69f77bfbe32719f4c>5e413c7872f5df231bf4a4f694504384560e98ca_0",
                  "excerpt_index": 0}

        input1 = {"source": "explicit", "citeEnd": 68.0, "sectionName": "Discussion", "citeStart": 64.0,
                  "string": "ssss",
                  "label": "result",
                  "label2": "supportive",
                  "citingPaperId": "8f1fbe460a901d994e9b81d69f77bfbe32719f4c", "citedPaperId": "5e413c7872f5df231bf4a4f694504384560e98ca",
                  "id": "8f1fbe460a901d994e9b81d69f77bfbe32719f4c>5e413c7872f5df231bf4a4f694504384560e98ca",
                  "unique_id": "8f1fbe460a901d994e9b81d69f77bfbe32719f4c>5e413c7872f5df231bf4a4f694504384560e98ca_0",
                  "excerpt_index": 0}

        param_file = 'tests/fixtures/test-config-scicite.json'
        dataset_file = 'tests/fixtures/scicite-train.jsonl'

        self.param_file = param_file
        params = Params.from_file(self.param_file)

        reader = DatasetReader.from_params(params['dataset_reader'])
        # The dataset reader might be lazy, but a lazy list here breaks some of our tests.
        instances = list(reader.read(str(dataset_file)))
        # Use parameters for vocabulary if they are present in the config file, so that choices like
        # "non_padded_namespaces", "min_count" etc. can be set if needed.
        if 'vocabulary' in params:
            vocab_params = params['vocabulary']
            vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
        else:
            vocab = Vocabulary.from_instances(instances)
        self.vocab = vocab
        self.instances = instances
        model = Model.from_params(vocab=self.vocab, params=params['model'])
        model.predict_mode = True

        predictor = Predictor.by_name("predictor_scicite")(model=model, dataset_reader=reader)

        result = predictor.predict_json(input0)
        prediction = result['prediction']
        assert prediction in {'background', 'method', 'result'}

        result = predictor.predict_json(input1)
        prediction = result['prediction']
        assert prediction in {'background', 'method', 'result', ''}
