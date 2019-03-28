""" Module including helper functions for feature extraction and other stuff including metrics, jsonhandler, etc"""
import json
from collections import Counter
import logging
import re
import numpy as np
import string

logger = logging.getLogger('classifier')

regex_find_citation = re.compile(r"\(\s?(([A-Za-z\-]+\s)+([A-Za-z\-]+\.?)?,?\s\d{2,4}[a-c]?(;\s)?)+\s?\)|"
                                 r"\[(\d{1,3},\s?)+\d{1,3}\]|"
                                 r"\[[\d,-]+\]|(\([A-Z][a-z]+, \d+[a-c]?\))|"
                                 r"([A-Z][a-z]+ (et al\.)? \(\d+[a-c]?\))|"
                                 r"[A-Z][a-z]+ and [A-Z][a-z]+ \(\d+[a-c]?\)]")

def print_top_words(model, feature_names, n_top_words):
    """ Prints top words in each topics for an LDA topic model"""
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def get_values_from_list(inplst, key, is_class=True):
    """ gets a value of an obj for a list of dicts (inplst)
    Args:
        inplst: list of objects
        key: key of interest
        is_class: ist the input object a class or a dictionary obj
    """
    return [getattr(elem, key) for elem in inplst] if is_class \
        else [elem[key] for elem in inplst]


def partial_fmeasure_multilabel(y_true,
                                y_pred,
                                pos_labels_index: list,
                                neg_labels_index: list):
    """Calculate F-measure when partial annotations for each class is available
     In calculating the f-measure, this function only considers examples that are annotated for each class
     and ignores instances that are not annotated for that class
     A set of positive and negative labels identify the samples that are annotated
     This functions expects the input to be one hot encoding of labels plus negative labels
     For example if labels set are ['cat', 'dog'] we also want to encode ['not-cat', 'not-dog']
     This is because if an annotator only examines an instance for `cat` and says this is `cat`
     we want to ignore this instance in calculation of f-score for `dog` category.
     Therefore, the input should have a shape of (num_instances, 2 * num_classes)
     e.g., A one hot encoding corresponding to [`cat`, `dog`] would be:
        [`cat`, `dog`, `not-cat`, `not-dog`]
    A list of pos_labels_index and negative_labels_index identify the corresponding pos and neg labels
    e.g., For our example above, pos_labels_index=[0,1] and neg_labels_idnex=[2,3]

     Args:
        y_true: A 2D array of true class, shape = (num_instances, 2*num_classes)
            e.g. [[1,0,0,0], [0,0,1,0], [0,0,0,1], [0,1,0,0], [0,1,0,0], ...]
        y_pred: A 2D array of precitions, shape = (num_instances, 2*num_classes)
        pos_labels_index: 1D array of shape (num_classes)
        neg_labels_index: 1D array of shape (num_classes)

    returns:
        list of precision scores, list of recall scores, list of F1 scores
        The list is in the order of positive labels for each class
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    precisions = []
    recalls = []
    f1s = []
    supports = []
    for pos_class, neg_class in zip(pos_labels_index, neg_labels_index):
        predictions_pos = y_pred[:, pos_class]
        predictions_neg = y_pred[:, neg_class]

        gold_pos = y_true[:, pos_class]
        gold_neg = y_true[:, pos_class]

        # argmax_predictions = predictions.max(-1)[1].float().squeeze(-1)
        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (predictions_neg == gold_neg).astype(float) * gold_neg
        _true_negatives = (correct_null_predictions.astype(float)).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (predictions_pos == gold_pos).astype(np.float) * predictions_pos
        _true_positives = correct_non_null_predictions.sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (predictions_pos != gold_pos).astype(np.float) * gold_pos
        _false_negatives = incorrect_null_predictions.sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (predictions_pos != gold_pos).astype(np.float) * predictions_pos
        _false_positives = incorrect_non_null_predictions.sum()

        precision = float(_true_positives) / float(_true_positives + _false_positives + 1e-13)
        recall = float(_true_positives) / float(_true_positives + _false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))

        support = (gold_pos + gold_neg).sum()

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1_measure)
        supports.append(support)

    return precisions, recalls, f1s, supports

def format_classification_report(precisions, recalls, f1s, supports, labels, digits=4):
    last_line_heading = 'avg / total'

    if labels is None:
        target_names = [u'%s' % l for l in labels]
    else:
        target_names = labels
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
    rows = zip(labels, precisions, recalls, f1s, supports)
    for row in rows:
        report += row_fmt.format(*row, width=5, digits=digits)

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(precisions, weights=supports),
                             np.average(recalls, weights=supports),
                             np.average(f1s, weights=supports),
                             np.sum(supports),
                             width=width, digits=digits)
    return report


class JsonFloatEncoder(json.JSONEncoder):
    """ numpy floats are not json serializable
    This class is a json encoder that enables dumping
    json objects that have numpy numbers in them
    use: json.dumps(obj, cls=JsonFLoatEncoder)"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonFloatEncoder, self).default(obj)


MIN_TOKEN_COUNT = 8
MAX_TOKEN_COUNT = 200
MIN_WORD_TOKENS_RATIO = 0.40
MIN_LETTER_CHAR_RATIO = 0.50
MIN_PRINTABLE_CHAR_RATIO = 0.95


# util function adopted from github.com/allenai/relex
def is_sentence(sentence: str) -> bool:
    if not isinstance(sentence, str):
        return False
    num_chars = len(sentence)
    tokens = sentence.split(' ')
    num_tokens = len(tokens)
    if num_tokens < MIN_TOKEN_COUNT:
        return False
    if num_tokens > MAX_TOKEN_COUNT:
        return False
    # Most tokens should be words
    if sum([t.isalpha() for t in tokens]) / num_tokens < MIN_WORD_TOKENS_RATIO:
        return False
    # Most characters should be letters, not numbers and not special characters
    if sum([c in string.ascii_letters for c in sentence]) / num_chars < MIN_LETTER_CHAR_RATIO:
        return False
    # Most characters should be printable
    if sum([c in string.printable for c in sentence]) / num_chars < MIN_PRINTABLE_CHAR_RATIO:
        return False
    return True
