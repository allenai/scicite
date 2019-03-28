""" Data reader for AllenNLP """


from typing import Dict, List
import json
import logging

import torch
from allennlp.data import Field
from overrides import overrides
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer

from scicite.resources.lexicons import ALL_ACTION_LEXICONS, ALL_CONCEPT_LEXICONS
from scicite.data import DataReaderJurgens
from scicite.data import DataReaderS2, DataReaderS2ExcerptJL
from scicite.compute_features import is_in_lexicon

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from scicite.constants import S2_CATEGORIES, NONE_LABEL_NAME


@DatasetReader.register("scicite_datasetreader")
class SciciteDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing citations from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        citation_text: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the methodology/comparison labels.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    reader_format : can be `flat` or `nested`. `flat` for flat json format and nested for
        Json format where the each object contains multiple excerpts
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 use_lexicon_features: bool=False,
                 use_sparse_lexicon_features: bool = False,
                 multilabel: bool = False,
                 with_elmo: bool = False,
                 reader_format: str = 'flat') -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        if with_elmo:
            # self._token_indexers = {"tokens": SingleIdTokenIndexer()}
            self._token_indexers = {"elmo": ELMoTokenCharactersIndexer(),
                                    "tokens": SingleIdTokenIndexer()}
        else:
            self._token_indexers = {"tokens": SingleIdTokenIndexer()}

        self.use_lexicon_features = use_lexicon_features
        self.use_sparse_lexicon_features = use_sparse_lexicon_features
        if self.use_lexicon_features or self.use_sparse_lexicon_features:
            self.lexicons = {**ALL_ACTION_LEXICONS, **ALL_CONCEPT_LEXICONS}
        self.multilabel = multilabel
        self.reader_format = reader_format

    @overrides
    def _read(self, jsonl_file: str):
        if self.reader_format == 'flat':
            reader_s2 = DataReaderS2ExcerptJL(jsonl_file)
        elif self.reader_format == 'nested':
            reader_s2 = DataReaderS2(jsonl_file)
        for citation in reader_s2.read():
            yield self.text_to_instance(
                citation_text=citation.text,
                intent=citation.intent,
                citing_paper_id=citation.citing_paper_id,
                cited_paper_id=citation.cited_paper_id,
                citation_excerpt_index=citation.citation_excerpt_index
            )

    @overrides
    def text_to_instance(self,
                         citation_text: str,
                         citing_paper_id: str,
                         cited_paper_id: str,
                         intent: List[str] = None,
                         citing_paper_title: str = None,
                         cited_paper_title: str = None,
                         citing_paper_year: int = None,
                         cited_paper_year: int = None,
                         citing_author_ids: List[str] = None,
                         cited_author_ids: List[str] = None,
                         extended_context: str = None,
                         section_number: int = None,
                         section_title: str = None,
                         cite_marker_begin: int = None,
                         cite_marker_end: int = None,
                         sents_before: List[str] = None,
                         sents_after: List[str] = None,
                         cleaned_cite_text: str = None,
                         citation_excerpt_index: str = None,
                         venue: str = None) -> Instance:  # type: ignore

        citation_tokens = self._tokenizer.tokenize(citation_text)

        fields = {
            'citation_text': TextField(citation_tokens, self._token_indexers),
        }

        if self.use_sparse_lexicon_features:
            # convert to regular string
            sent = [token.text.lower() for token in citation_tokens]
            lexicon_features, _ = is_in_lexicon(self.lexicons, sent)
            fields["lexicon_features"] = ListField([LabelField(feature, skip_indexing=True)
                                                    for feature in lexicon_features])

        if intent:
            if self.multilabel:
                fields['labels'] = MultiLabelField([S2_CATEGORIES[e] for e in intent], skip_indexing=True,
                                                   num_labels=len(S2_CATEGORIES))
            else:
                if not isinstance(intent, str):
                    raise TypeError(f"Undefined label format. Should be a string. Got: f'{intent}'")
                fields['labels'] = LabelField(intent)

        if citing_paper_year and cited_paper_year and \
                citing_paper_year > -1 and cited_paper_year > -1:
            year_diff = citing_paper_year - cited_paper_year
        else:
            year_diff = -1
        fields['year_diff'] = ArrayField(torch.Tensor([year_diff]))
        fields['citing_paper_id'] = MetadataField(citing_paper_id)
        fields['cited_paper_id'] = MetadataField(cited_paper_id)
        fields['citation_excerpt_index'] = MetadataField(citation_excerpt_index)
        fields['citation_id'] = MetadataField(f"{citing_paper_id}>{cited_paper_id}")
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'SciciteDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        use_lexicon_features = params.pop_bool("use_lexicon_features", False)
        use_sparse_lexicon_features = params.pop_bool("use_sparse_lexicon_features", False)
        multilabel = params.pop_bool("multilabel")
        with_elmo = params.pop_bool("with_elmo", False)
        reader_format = params.pop("reader_format", 'flat')
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer,
                   use_lexicon_features=use_lexicon_features,
                   use_sparse_lexicon_features=use_sparse_lexicon_features,
                   multilabel=multilabel,
                   with_elmo=with_elmo,
                   reader_format=reader_format)
