""" Module for computing features """
import re
from collections import Counter, defaultdict
from typing import List, Optional, Tuple, Type

import functools
from spacy.tokens.token import Token as SpacyToken

import scicite.constants as constants
from scicite.constants import CITATION_TOKEN
from scicite.resources.lexicons import (AGENT_PATTERNS, ALL_ACTION_LEXICONS,
                                        ALL_CONCEPT_LEXICONS, FORMULAIC_PATTERNS)
from scicite.data import Citation
import logging

logger = logging.getLogger('classifier')

NETWORK_WEIGHTS_FILE = constants.root_path + '/resources/arc-network-weights.tsv'


def load_patterns(filename, p_dict, label):
    with open(filename) as f:
        class_counts = Counter()
        for line in f:
            if not '@' in line:
                continue
            cols = line.split("\t")
            pattern = cols[0].replace("-lrb-", "(").replace('-rrb-', ')')
            category = cols[1]
            if category == 'Background':
                continue
            class_counts[category] += 1
            p_dict[category + '_' + label + '_' + str(class_counts[category])] \
                = pattern.split()
            # p_dict[clazz + '_' + label].append(pattern.split())


def get_values_from_list(inplst, key, is_class=True):
    """ gets a value of an obj for a list of dicts (inplst)
    Args:
        inplst: list of objects
        key: key of interest
        is_class: ist the input object a class or a dictionary obj
    """
    return [getattr(elem, key) for elem in inplst] if is_class \
        else [elem[key] for elem in inplst]


def is_in_lexicon(lexicon: dict,
                  sentence: str,
                  count: Optional[bool] = False) -> Tuple[List[str], List[str]]:
    """ checks if the words in a lexicon exist in the sentence """
    features = []
    feature_names = []
    if count:
        cnt = 0
    for key, word_list in lexicon.items():
        exists = False
        for word in word_list:
            if word in sentence:
                if not count:
                    exists = True
                    break
                else:
                    cnt += 1
        if not count:
            features.append(exists)
        else:
            features.append(cnt)
        feature_names.append(key)
    return features, feature_names
