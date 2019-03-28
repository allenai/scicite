import json
import jsonlines
import glob
from typing import Dict, Any, Union
import re
import Levenshtein
import logging
import numpy as np

from scicite.constants import CITATION_TOKEN, NEGATIVE_CLASS_PREFIX
from scicite.helper import regex_find_citation

logger = logging.getLogger()


class Citation(object):
    """ Class representing a citation object """

    def __init__(self,
                 text,
                 citing_paper_id,
                 cited_paper_id,
                 citing_paper_title=None,
                 cited_paper_title=None,
                 citing_paper_year=None,
                 cited_paper_year=None,
                 citing_author_ids=None,
                 cited_author_ids=None,
                 extended_context=None,
                 section_number=None,
                 section_title=None,
                 intent=None,
                 cite_marker_offset=None,
                 sents_before=None,
                 sents_after=None,
                 cleaned_cite_text=None,
                 citation_excerpt_index=0,
                 citation_id=None
                 ):
        """
        Args:
            citation_text: the citation excerpt text
            citing_paper_id
            cited_paper_id
            citing_paper_title
            cited_paper_title
            citing_paper_year
            cited_paper_year
            citing_author_ids: may be used to identify self-cites
            cited_author_ids
            extended_context: additional context beyond the excerpt
            section_number: number of the section in which the citation occurs
            section_title: title of the section in which the citation occurs
            intent(list): the labels associated with the citation excerpt
            cite_marker_offset: the offsets of the citation marker
            sents_before: list of sentences before the excerpt
            sents_after: list of sentences after the excerpt
            cleaned_cite_text: instead of citation marker, a dummy token is used
            citation_excerpt_index: index of the excerpt in the citation blob
                for reconstructing the output
            citation_id: id of the citation
        """
        self.text = text
        self.citing_paper_id = citing_paper_id
        self.cited_paper_id = cited_paper_id
        self.citing_paper_year = citing_paper_year
        self.cited_paper_year = cited_paper_year
        self.citing_paper_title = citing_paper_title
        self.cited_paper_title = cited_paper_title
        self.cited_author_ids = cited_author_ids
        self.citing_author_ids = citing_author_ids
        self.extended_context = extended_context
        self.section_number = section_number
        self.section_title = section_title
        self.intent = intent
        self.cite_marker_offset = cite_marker_offset
        self.sents_before = sents_before
        self.sents_after = sents_after
        self.cleaned_cite_text = cleaned_cite_text
        self.citation_id = citation_id
        self.citation_excerpt_index = citation_excerpt_index


class BaseReader:

    def read_data(self):
        raise NotImplementedError()


class DataReaderS2(BaseReader):
    """
    Reads raw data from the json blob format from s2 and returns a Citation class
    The S2 citation data have the following format
    {'id': str,
     'citedPaper': {'title': str,
      'authors': [{'name': str, 'ids': [str, '...']}, '...'],
      'venue': str,
      'id': str,
      'year': int},
     'citingPaper': {'title': str,
      'authors': [{'name': str, 'ids': [str, '...']}, '...'],
      'venue': str,
      'id': str,
      'year': int},
     'isKeyCitation': bool,
     'keyCitationTypes': [],
     'citationRelevance': int,
     'context': [{'source': str,
       'citeStart': int,
       'sectionName': str,
       'string': str,
       'citeEnd': int,
       'intents' (optional): [{'intent': str, 'score': float}, '...']},
      '...']}
    """

    METHOD = 'method'
    COMPARISON = 'comparison'

    def __init__(self, data_path, evaluate_mode=True, clean_citation=True, multilabel=False):
        """
        Constructor
        Args:
            data_path: path to the json file containing all the blobs
            evaluate_mode: is this evaluate more (train, or test) or inference mode
                If inference, it returns every context as a new citation object
            clean_citation: clean citation marker (e.g., "(Peters, et al) ..." -> "..."
        """
        self.data_path = data_path
        self.evaluate_mode = evaluate_mode
        self.clean_citation = clean_citation
        self.multilabel = multilabel

    def read(self):
        """ Reads the input data and yields a citation object"""
        data = [json.loads(line) for line in open(self.data_path)]
        num_returned_citations = 0
        num_not_annotated = 0
        for ex in data:
            try:
                citing_paper_year = ex['citingPaper']['year']
            except KeyError:
                citing_paper_year = -1
            try:
                cited_paper_year = ex['citedPaper']['year']
            except KeyError:
                cited_paper_year = -1

            # authors is like: [{'name': 'S Pandav', 'ids': ['2098534'], ...}]
            try:
                citing_author_ids = [author['ids'][0] if author['ids'] else 'n/a'
                                     for author in ex['citingPaper']['authors']]
            except KeyError:  # authors do not exist in the context:
                citing_author_ids = []
            try:
                cited_author_ids = [author['ids'][0] if author['ids'] else 'n/a'
                                    for author in ex['citedPaper']['authors']]
            except KeyError:
                cited_author_ids = []

            for excerpt_index, excerpt_obj in enumerate(ex['context']):
                if self.evaluate_mode:  # only consider excerpts that are annotated
                    if 'intents' not in excerpt_obj:
                        num_not_annotated += 1
                        continue

                try:
                    offsets = [excerpt_obj['citeStart'], excerpt_obj['citeEnd']]
                except KeyError:  # context does not have citeStart or citeEnd
                    offsets = [-1, -1]

                if self.clean_citation:
                    # remove citation markers (e.g., things like [1,4], (Peters, et al 2018), etc)
                    citation_text = regex_find_citation.sub("", excerpt_obj['string'])
                else:
                    citation_text = excerpt_obj['string']
                section_name = excerpt_obj['sectionName']

                # in case of multilabel add all possible labels and their negative prefix
                if self.multilabel:
                    intents = [e['intent'] if e['score'] > 0.0
                               else NEGATIVE_CLASS_PREFIX + e['intent'] for e in excerpt_obj['intents']]
                else:
                    intents = [e['intent'] for e in excerpt_obj['intents'] if e['score'] > 0.0]

                citation = Citation(
                    text=citation_text,
                    citing_paper_id=ex['citingPaper']['id'],
                    cited_paper_id=ex['citedPaper']['id'],
                    citing_paper_title=ex['citingPaper']['title'],
                    cited_paper_title=ex['citedPaper']['title'],
                    citing_paper_year=citing_paper_year,
                    cited_paper_year=cited_paper_year,
                    citing_author_ids=citing_author_ids,
                    cited_author_ids=cited_author_ids,
                    extended_context=None,  # Not available for s2 data
                    section_number=None,  # Not available for s2 data
                    section_title=section_name,
                    intent=intents,
                    cite_marker_offset=offsets,  # Not useful here
                    sents_before=None,  # not available for s2 data
                    sents_after=None,  # not available for s2 data
                    citation_excerpt_index=excerpt_index,
                    cleaned_cite_text=citation_text
                )
                num_returned_citations += 1
                yield citation

        logger.info(f'Total annotated citation texts returned: {num_returned_citations}; '
                    f'not annotated {num_not_annotated}')

    def read_data(self):
        data = [ex for ex in self.read()]
        return data


class DataReaderJurgens(BaseReader):
    """
    class for reading the data from Jurgens dataset
    See https://transacl.org/ojs/index.php/tacl/article/download/1266/313 for details
        on the dataset
    The dataset format in the Jurgens dataset is according to the following:
    The dataset contains about 186 json files.
    Each file is a json file from a paper containing all the citations
    Some citation contexts (excerpts) are labeled
    The format of the Jurgens dataset looks like the following:
    {'year': int,
    'sections': list of sections (each a dictionary)
    'paper_id': str
    'citation_contexts': list of citation contexts (each a dictionary)
        {'info': {'authors': ['Anonymous'],
          'year': '1999',
          'title': 'sunrainnet  englishchinese dictionary'},
          'citing_string': 'Anonymous , 1999',
          'sentence': 11,
          'section': 5,
          'citation_id': 'A00-1004_0',
          'raw_string': 'Anonymous. 1999a. Sunrain.net - English-Chinese dictionary.',
          'subsection': 0,
          'cite_context': 'into Chinese and then translated back ...',
          'is_self_cite': False,
          'cited_paper_id': 'External_21355'
        }
    """

    def __init__(self, data_path, teufel_data_path=None):
        """ Args:
            data_path: path to the json data
            teufel_data_path: path to the teufel data files
        """
        self.data_path = data_path
        self.teufel_data_path = teufel_data_path

    def read(self):
        files = [elem for elem
                 in glob.glob('{}/**/*.json'.format(self.data_path), recursive=True)]
        for file in files:
            paper = json.load(open(file))
            for citation in paper['citation_contexts']:
                if 'citation_role' not in citation and 'citation_function' not in citation:
                    continue
                cit_obj = self._get_citation_obj(paper, citation)
                if cit_obj is not None:
                    yield cit_obj
        # add additional data from Teufel
        if self.teufel_data_path:
            files = [elem for elem in glob.glob(
                '{}/**/*.json'.format(self.teufel_data_path), recursive=True)]
            for file in files:
                paper = json.load(open(file))
                for citation in paper['citation_contexts']:
                    # Jurgens ignored the `background` labels because they are already dominant
                    if 'citation_role' not in citation or \
                            citation['citation_role'].lower() == 'background':
                        continue
                    cit_obj = self._get_citation_obj(paper, citation)
                    if cit_obj is not None:
                        yield cit_obj

    def read_data(self):
        data = [ex for ex in self.read()]
        return data

    def read_batch_paper(self):
        """ Returns a list of lists
        each list corresponds to data.Citations in one paper"""
        files = [elem for elem
                 in glob.glob('{}/**/*.json'.format(self.data_path), recursive=True)]
        print('reading files')
        data_by_paper = []
        for file in files:
            paper = json.load(open(file))
            batch = []
            for citation in paper['citation_contexts']:
                if 'citation_role' not in citation:
                    continue
                cit_obj = self._get_citation_obj(paper, citation)
                if cit_obj is not None:
                    batch.append(cit_obj)
            data_by_paper.append(batch)
        if self.teufel_data_path:
            files = [elem for elem in glob.glob(
                '{}/**/*.json'.format(self.teufel_data_path), recursive=True)]
            for file in files:
                paper = json.load(open(file))
                batch = []
                for citation in paper['citation_contexts']:
                    # Jurgens ignored the `background` labels because they are already dominant
                    if 'citation_role' not in citation or \
                            citation['citation_role'].lower() == 'background':
                        continue
                    cit_obj = self._get_citation_obj(paper, citation)
                    if cit_obj is not None:
                        batch.append(cit_obj)
                data_by_paper.append(batch)
        return data_by_paper

    def _get_citance(self, citation_context, parsed_doc) -> str:
        # pylint: disable=bad-continuation
        return parsed_doc['sections'][citation_context['section']][
            'subsections'][citation_context['subsection']][
            'sentences'][citation_context['sentence']]['text']

    def _get_subsection(self, citation_context, parsed_doc):
        return parsed_doc['sections'][citation_context['section']][
            'subsections'][citation_context['subsection']]

    def _get_citance_context(self, citation_context, parsed_doc, window=1) -> str:
        """ Get a context window around the citation text
        i.elem., a window of sentences around the citation sentence
        will be returned """
        # pylint: disable=bad-continuation
        subsection = parsed_doc['sections'][
            citation_context['section']][
            'subsections'][citation_context['subsection']]
        max_len = len(subsection['sentences'])
        if citation_context['sentence'] + window + 1 > max_len:
            up_bound = max_len
        else:
            up_bound = citation_context['sentence'] + window + 1
        if citation_context['sentence'] - window < 0:
            low_bound = 0
        else:
            low_bound = citation_context['sentence'] - window
        return ' '.join(
            [elem['text']
             for elem in subsection['sentences'][low_bound:up_bound]])

    def _get_citation_obj(self, paper, citation) -> Dict:
        try:
            intent = citation['citation_role'].split('-')[0]
        except KeyError:  # sometimes the key is citation_function instead of citation_role
            intent = citation['citation_function'].split('-')[0]
        # Jurgens results are based on fewer categories than in the dataset.
        # We merge the categories accordingly
        if intent == 'Prior':
            intent = 'Extends'
        elif intent == 'Compares' or intent == 'Contrasts':
            intent = 'CompareOrContrast'

        citation_excerpt = self._get_citance(citation, paper)

        citation_marker = citation['citing_string']
        cite_beg = citation_excerpt.find(citation_marker)
        if cite_beg < 0:
            logger.warning(r'Citation marker does not appear in the citation text'
                           ':\n citation context:{}\n citation marker:{}'
                           ''.format(citation_excerpt, citation_marker))
        cite_end = cite_beg + len(citation_marker)

        sent_index: int = citation['sentence']
        sent: str = citation_excerpt

        text_before: str = sent[:cite_beg]
        text_after: str = sent[cite_end:]
        replaced_text = text_before + CITATION_TOKEN + text_after

        subsection = self._get_subsection(citation, paper)
        # Use the 2 sentences before as pre-citance context
        preceding_sents = []
        for prev in range(sent_index, max(-1, sent_index - 3), -1):
            preceding_sents.append(subsection['sentences'][prev]['tokens'])

        # Use the 4 sentences after as post-citance context
        following_sents = []
        for foll in range(sent_index + 1, min(sent_index + 5, len(subsection['sentences']))):
            following_sents.append(subsection['sentences'][foll]['tokens'])

        citation_obj = Citation(
            text=self._get_citance(citation, paper),
            citing_paper_id=paper['paper_id'],
            cited_paper_id=citation['cited_paper_id'],
            citing_paper_title=list(paper['sections'][0].keys())[0],
            cited_paper_title=citation['info']['title'],
            citing_paper_year=paper['year'],
            cited_paper_year=int(citation['info']['year']),
            citing_author_ids=None,
            cited_author_ids=citation['info']['authors'],
            extended_context=self._get_citance_context(citation, paper, 1),
            section_number=citation['section'],
            intent=intent,
            cite_marker_offset=[cite_beg, cite_end],
            sents_before=preceding_sents,
            sents_after=following_sents,
            cleaned_cite_text=replaced_text,
            citation_id=citation['citation_id']
        )

        return citation_obj


def read_s2_jsonline(ex, evaluate_mode=False, clean_citation=True, multilabel=False):
    """ reads a json lines object (citation blob)
    This is a separate function to be used in the predictor
     Args:
        ex: input Example
        evaluate_mode: If we are evaluating only consider annotated excerpts
    """
    citations = []
    num_not_annotated = 0
    try:
        citing_paper_year = ex['citingPaper']['year']
    except KeyError:
        citing_paper_year = -1
    try:
        cited_paper_year = ex['citedPaper']['year']
    except KeyError:
        cited_paper_year = -1

        # authors is like: [{'name': 'S Pandav', 'ids': ['2098534'], ...}]
    try:
        citing_author_ids = [author['ids'][0] if author['ids'] else 'n/a'
                             for author in ex['citingPaper']['authors']]
    except KeyError:  # authors do not exist in the context:
        citing_author_ids = []
    try:
        cited_author_ids = [author['ids'][0] if author['ids'] else 'n/a'
                            for author in ex['citedPaper']['authors']]
    except KeyError:
        cited_author_ids = []

    for excerpt_index, excerpt_obj in enumerate(ex['context']):
        if evaluate_mode:  # only consider excerpts that are annotated
            if 'intents' not in excerpt_obj:
                num_not_annotated += 1
                continue

        try:
            offsets = [excerpt_obj['citeStart'], excerpt_obj['citeEnd']]
        except KeyError:  # context does not have citeStart or citeEnd
            offsets = [-1, -1]

        if clean_citation:
            # remove citation markers (e.g., things like [1,4], (Peters, et al 2018), etc)
            citation_text = regex_find_citation.sub("", excerpt_obj['string'])
        else:
            citation_text = excerpt_obj['string']
        section_name = excerpt_obj['sectionName']

        # intents = [e['intent'] for e in excerpt_obj['intents'] if e['score'] > 0.0]

        if 'intents' in excerpt_obj:
            if multilabel:
                intents = [e['intent'] if e['score'] > 0.0
                           else NEGATIVE_CLASS_PREFIX + e['intent'] for e in excerpt_obj['intents']]
            else:
                intents = [e['intent'] for e in excerpt_obj['intents'] if e['score'] > 0.0]
        else:
            intents = None



        citation = Citation(
            text=citation_text,
            citing_paper_id=ex['citingPaper']['id'],
            cited_paper_id=ex['citedPaper']['id'],
            citing_paper_title=ex['citingPaper']['title'],
            cited_paper_title=ex['citedPaper']['title'],
            citing_paper_year=citing_paper_year,
            cited_paper_year=cited_paper_year,
            citing_author_ids=citing_author_ids,
            cited_author_ids=cited_author_ids,
            extended_context=None,  # Not available for s2 data
            section_number=None,  # Not available for s2 data
            section_title=section_name,
            intent=intents,
            cite_marker_offset=offsets,  # Not useful here
            sents_before=None,  # not available for s2 data
            sents_after=None,  # not available for s2 data
            citation_excerpt_index=excerpt_index,
            cleaned_cite_text=citation_text
        )
        citations.append(citation)
    return citations


def read_jurgens_jsonline(ex):
    citation_obj = Citation(
        text=ex.get('text'),
        citing_paper_id=ex.get('citing_paper_id'),
        cited_paper_id=ex.get('cited_paper_id'),
        citing_paper_title=ex.get('citing_paper_title'),
        cited_paper_title=ex.get('cited_paper_title'),
        citing_paper_year=ex.get('citing_paper_year'),
        cited_paper_year=ex.get('cited_paper_year'),
        cited_author_ids=ex.get('cited_author_ids'),
        extended_context=ex.get('extended_context'),
        section_number=ex.get('section_number'),
        section_title=ex.get('section_title'),
        intent=ex.get('intent'),
        cite_marker_offset=ex.get('cite_marker_offset'),
        sents_before=ex.get('sents_before'),
        sents_after=ex.get('sents_after'),
        cleaned_cite_text=ex.get('cleaned_cite_text'),
        citation_id=ex.get('citation_id'),
    )
    return citation_obj


class DataReaderJurgensJL(BaseReader):

    def __init__(self, data_path):
        super(DataReaderJurgensJL, self).__init__()
        self.data_path = data_path

    def read(self):
        for e in jsonlines.open(self.data_path):
            yield read_jurgens_jsonline(e)

    def read_data(self):
        data = [ex for ex in self.read()]
        return data


def read_s2_excerpt(ex):
    """
    Reads excerpts in a jsonlines format
    In this format each citation excerpt is one json line
    It is the flattened format from the original s2 data
    Args:
        ex: citation excerpt blob

    Returns:
        Citation object
    """
    citation = Citation(
        text=ex['string'],
        citing_paper_id=ex['citingPaperId'],
        cited_paper_id=ex['citedPaperId'],
        # citing_paper_title=ex['citingPaper']['title'],
        # cited_paper_title=ex['citedPaper']['title'],
        # citing_paper_year=citing_paper_year,
        # cited_paper_year=cited_paper_year,
        # citing_author_ids=citing_author_ids,
        # cited_author_ids=cited_author_ids,
        extended_context=None,  # Not available for s2 data
        section_number=None,  # Not available for s2 data
        section_title=ex['sectionName'],
        intent=ex['label'],
        # cite_marker_offset=offsets,  # Not useful here
        sents_before=None,  # not available for s2 data
        sents_after=None,  # not available for s2 data
        citation_excerpt_index=ex['excerpt_index'],
        cleaned_cite_text=regex_find_citation.sub('', ex['string'])
    )
    return citation


class DataReaderS2ExcerptJL(BaseReader):

    def __init__(self, data_path):
        super(DataReaderS2ExcerptJL, self).__init__()
        self.data_path = data_path

    def read(self):
        for e in jsonlines.open(self.data_path):
            yield read_s2_excerpt(e)

    def read_data(self):
        data = [ex for ex in self.read()]
        return data
