""" Data reader for AllenNLP """


from typing import Dict, List
import json
import jsonlines
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
from scicite.data import read_jurgens_jsonline
from scicite.compute_features import is_in_lexicon

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from scicite.constants import S2_CATEGORIES, NONE_LABEL_NAME


@DatasetReader.register("aclarc_dataset_reader")
class AclarcDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    where the ``label`` is derived from the citation intent

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
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_lexicon_features: bool = False,
                 use_sparse_lexicon_features: bool = False,
                 with_elmo: bool = False
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        if with_elmo:
            self._token_indexers = {"elmo": ELMoTokenCharactersIndexer(),
                                    "tokens": SingleIdTokenIndexer()}
        else:
            self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.use_lexicon_features = use_lexicon_features
        self.use_sparse_lexicon_features = use_sparse_lexicon_features
        if self.use_lexicon_features or self.use_sparse_lexicon_features:
            self.lexicons = {**ALL_ACTION_LEXICONS, **ALL_CONCEPT_LEXICONS}

    @overrides
    def _read(self, file_path):
        for ex in jsonlines.open(file_path):
            citation = read_jurgens_jsonline(ex)
            yield self.text_to_instance(
                citation_text=citation.text,
                intent=citation.intent,
                citing_paper_id=citation.citing_paper_id,
                cited_paper_id=citation.cited_paper_id
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
                         sents_before: List[str] = None,
                         sents_after: List[str] = None,
                         cite_marker_begin: int = None,
                         cite_marker_end: int = None,
                         cleaned_cite_text: str = None,
                         citation_excerpt_index: str = None,
                         citation_id: str = None,
                         venue: str = None) -> Instance:  # type: ignore

        citation_tokens = self._tokenizer.tokenize(citation_text)
        # tok_cited_title = self._tokenizer.tokenize(cited_paper_title)
        # tok_citing_title = self._tokenizer.tokenize(citing_paper_title)
        # tok_extended_context = self._tokenizer.tokenize(extended_context)

        fields = {
            'citation_text': TextField(citation_tokens, self._token_indexers),
        }

        if self.use_sparse_lexicon_features:
            # convert to regular string
            sent = [token.text.lower() for token in citation_tokens]
            lexicon_features, _ = is_in_lexicon(self.lexicons, sent)
            fields["lexicon_features"] = ListField([LabelField(feature, skip_indexing=True)
                                                    for feature in lexicon_features])

        if intent is not None:
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
        fields['citation_id'] = MetadataField(citation_id)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'AclarcDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        use_lexicon_features = params.pop_bool("use_lexicon_features", False)
        use_sparse_lexicon_features = params.pop_bool("use_sparse_lexicon_features", False)
        with_elmo = params.pop_bool("with_elmo", False)
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer,
                   use_lexicon_features=use_lexicon_features,
                   use_sparse_lexicon_features=use_sparse_lexicon_features,
                   with_elmo=with_elmo)
