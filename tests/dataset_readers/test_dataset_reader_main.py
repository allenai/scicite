# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

import sys

from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from scicite.dataset_readers.citation_data_reader_aclarc import AclarcDatasetReader
from scicite.dataset_readers.citation_data_reader_aclarc_aux import AclSectionTitleDatasetReader, AclCiteWorthinessDatasetReader
from scicite.dataset_readers.citation_data_reader_scicite import SciciteDatasetReader
from scicite.dataset_readers.citation_data_reader_scicite_aux import SciciteSectitleDatasetReader, SciCiteWorthinessDataReader


class TestDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = AclarcDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/aclarc-train.jsonl'))
        instance1 = {"citation_text": ['Typical', 'examples', 'are', 'Bulgarian']}
        assert len(instances) == 10
        fields = instances[0].fields
        assert isinstance(instances, list)
        assert [t.text for t in fields['citation_text'].tokens][:4] == instance1['citation_text']

        reader = AclSectionTitleDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/aclarc-section-title.jsonl'))
        instance1 = {"section_name": 'related work', "citation_text": ['With', 'C99']}
        assert len(instances) == 10
        fields = instances[1].fields
        assert isinstance(instances, list)
        assert [t.text for t in fields['citation_text'].tokens][:2] == instance1['citation_text']
        assert fields['section_label'].label == instance1['section_name']

        reader = AclCiteWorthinessDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/aclarc-cite-worthiness.jsonl'))
        instance1 = {"is_citation": 'False'}
        fields = instances[1].fields
        assert isinstance(instances, list)
        assert fields['is_citation'].label == instance1['is_citation']

    def test_read_from_file_scicite(self):
        reader = SciciteDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/scicite-train.jsonl'))
        instance1 = {"citation_text": ['These', 'results', 'are', 'in']}
        assert len(instances) == 10
        fields = instances[0].fields
        assert isinstance(instances, list)
        assert [t.text for t in fields['citation_text'].tokens][:4] == instance1['citation_text']
        print(fields.keys())
        assert fields['labels'].label == "result"

        reader = SciciteSectitleDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/scicite-section-title.jsonl'))
        instance1 = {"section_name": 'introduction', "citation_text": ['SVM', 'and']}
        assert len(instances) == 10
        fields = instances[0].fields
        assert isinstance(instances, list)
        assert [t.text for t in fields['citation_text'].tokens][:2] == instance1['citation_text']
        assert fields['section_label'].label == instance1['section_name']
        assert 'is_citation' not in fields

        reader = SciCiteWorthinessDataReader()
        instances = ensure_list(reader.read('tests/fixtures/scicite-cite-worthiness.jsonl'))
        instance1 = {"is_citation": 'True'}
        fields = instances[0].fields
        assert isinstance(instances, list)
        assert fields['is_citation'].label == instance1['is_citation']
        assert 'section_name' not in fields.keys()
