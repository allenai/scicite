# pylint: disable=invalid-name,protected-access
import copy
from typing import Dict, Union, Any, Set

from allennlp.common import Params
from allennlp.common.testing import ModelTestCase

import sys

import os
from pathlib import Path

from allennlp.data import DatasetReader, DataIterator
from numpy.testing import assert_allclose

from allennlp.models import load_archive

sys.path.append(str(Path('.').absolute()))

from scicite.training.train_multitask_two_tasks import train_model_from_file
from scicite.constants import root_path
from scicite.models.scaffold_bilstm_attention_classifier import ScaffoldBilstmAttentionClassifier
from scicite.dataset_readers.citation_data_reader_scicite import SciciteDatasetReader
from scicite.dataset_readers.citation_data_reader_scicite_aux import SciciteSectitleDatasetReader, SciCiteWorthinessDataReader


class CitationClassifierTest(ModelTestCase):

    def setUp(self):
        super(CitationClassifierTest, self).setUp()
        self.set_up_model(str(Path(root_path).parent) + '/tests/fixtures/test-config-scicite.json',
                          str(Path(root_path).parent) + '/tests/fixtures/scicite-train.jsonl')

    def test_model_can_train_save_and_load(self):
        param_file = self.param_file
        cuda_device = -1
        tolerance = 1e-4

        save_dir = self.TEST_DIR / "save_and_load_test"
        archive_file = save_dir / "model.tar.gz"
        model = train_model_from_file(param_file, save_dir)
        loaded_model = load_archive(archive_file, cuda_device=cuda_device).model
        state_keys = model.state_dict().keys()
        loaded_state_keys = loaded_model.state_dict().keys()
        assert state_keys == loaded_state_keys
        # First we make sure that the state dict (the parameters) are the same for both models.
        for key in state_keys:
            assert_allclose(model.state_dict()[key].cpu().numpy(),
                            loaded_model.state_dict()[key].cpu().numpy(),
                            err_msg=key)
        params = Params.from_file(param_file)
        reader = DatasetReader.from_params(params['dataset_reader'])

        # Need to duplicate params because Iterator.from_params will consume.
        iterator_params = params['iterator']
        iterator_params2 = Params(copy.deepcopy(iterator_params.as_dict()))

        iterator = DataIterator.from_params(iterator_params)
        iterator2 = DataIterator.from_params(iterator_params2)

        # We'll check that even if we index the dataset with each model separately, we still get
        # the same result out.
        model_dataset = reader.read(params['validation_data_path'])
        iterator.index_with(model.vocab)
        model_batch = next(iterator(model_dataset, shuffle=False))

        loaded_dataset = reader.read(params['validation_data_path'])
        iterator2.index_with(loaded_model.vocab)
        loaded_batch = next(iterator2(loaded_dataset, shuffle=False))

        # ignore auxiliary task parameters
        params_to_ignore = {'classifier_feedforward_2._linear_layers.0.weight',
                            'classifier_feedforward_2._linear_layers.0.bias',
                            'classifier_feedforward_2._linear_layers.1.weight',
                            'classifier_feedforward_2._linear_layers.1.bias',
                            'classifier_feedforward_3._linear_layers.0.weight',
                            'classifier_feedforward_3._linear_layers.0.bias',
                            'classifier_feedforward_3._linear_layers.1.weight',
                            'classifier_feedforward_3._linear_layers.1.bias'
                            }
        # Check gradients are None for non-trainable parameters and check that
        # trainable parameters receive some gradient if they are trainable.
        self.check_model_computes_gradients_correctly(model, model_batch,
                                                      params_to_ignore=params_to_ignore)

        # The datasets themselves should be identical.
        assert model_batch.keys() == loaded_batch.keys()
        for key in model_batch.keys():
            self.assert_fields_equal(model_batch[key], loaded_batch[key], key, 1e-6)

        # Set eval mode, to turn off things like dropout, then get predictions.
        model.eval()
        loaded_model.eval()
        # Models with stateful RNNs need their states reset to have consistent
        # behavior after loading.
        for model_ in [model, loaded_model]:
            for module in model_.modules():
                if hasattr(module, 'stateful') and module.stateful:
                    module.reset_states()
        model_predictions = model(**model_batch)
        loaded_model_predictions = loaded_model(**loaded_batch)

        # Check loaded model's loss exists and we can compute gradients, for continuing training.
        loaded_model_loss = loaded_model_predictions["loss"]
        assert loaded_model_loss is not None
        loaded_model_loss.backward()

        # Both outputs should have the same keys and the values for these keys should be close.
        for key in model_predictions.keys():
            self.assert_fields_equal(model_predictions[key],
                                     loaded_model_predictions[key],
                                     name=key,
                                     tolerance=tolerance)

        return model, loaded_model
