import operator
from copy import deepcopy
from distutils.version import StrictVersion
from typing import Dict, Optional

import allennlp
import numpy as np
import torch
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder, Embedding, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides
from torch.nn import Parameter, Linear

from scicite.constants import  Scicite_Format_Nested_Jsonlines


import torch.nn as nn


@Model.register("scaffold_bilstm_attention_classifier")
class ScaffoldBilstmAttentionClassifier(Model):
    """
    This ``Model`` performs text classification for citation intents.  We assume we're given a
    citation text, and we predict some output label.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 citation_text_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 classifier_feedforward_2: FeedForward,
                 classifier_feedforward_3: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 report_auxiliary_metrics: bool = False,
                 predict_mode: bool = False,
                 ) -> None:
        """
        Additional Args:
            lexicon_embedder_params: parameters for the lexicon attention model
            use_sparse_lexicon_features: whether to use sparse (onehot) lexicon features
            multilabel: whether the classification is multi-label
            data_format: s2 or jurgens
            report_auxiliary_metrics: report metrics for aux tasks
            predict_mode: predict unlabeled examples
        """
        super(ScaffoldBilstmAttentionClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.num_classes_sections = self.vocab.get_vocab_size("section_labels")
        self.num_classes_cite_worthiness = self.vocab.get_vocab_size("cite_worthiness_labels")
        self.citation_text_encoder = citation_text_encoder
        self.classifier_feedforward = classifier_feedforward
        self.classifier_feedforward_2 = classifier_feedforward_2
        self.classifier_feedforward_3 = classifier_feedforward_3

        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}
        self.label_f1_metrics_sections = {}
        self.label_f1_metrics_cite_worthiness = {}
        # for i in range(self.num_classes):
        #     self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] =\
        #         F1Measure(positive_label=i)

        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] =\
                F1Measure(positive_label=i)
        for i in range(self.num_classes_sections):
            self.label_f1_metrics_sections[vocab.get_token_from_index(index=i, namespace="section_labels")] =\
                F1Measure(positive_label=i)
        for i in range(self.num_classes_cite_worthiness):
            self.label_f1_metrics_cite_worthiness[vocab.get_token_from_index(index=i, namespace="cite_worthiness_labels")] =\
                F1Measure(positive_label=i)
        self.loss = torch.nn.CrossEntropyLoss()

        self.attention_seq2seq = Attention(citation_text_encoder.get_output_dim())

        self.report_auxiliary_metrics = report_auxiliary_metrics
        self.predict_mode = predict_mode

        initializer(self)

    @overrides
    def forward(self,
                citation_text: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                lexicon_features: Optional[torch.IntTensor] = None,
                year_diff: Optional[torch.Tensor] = None,
                citing_paper_id: Optional[str] = None,
                cited_paper_id: Optional[str] = None,
                citation_excerpt_index: Optional[str] = None,
                citation_id: Optional[str] = None,
                section_label: Optional[torch.Tensor] = None,
                is_citation: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model
        Args:
            citation_text: citation text of shape (batch, sent_len, embedding_dim)
            labels: labels
            lexicon_features: lexicon sparse features (batch, lexicon_feature_len)
            year_diff: difference between cited and citing years
            citing_paper_id: id of the citing paper
            cited_paper_id: id of the cited paper
            citation_excerpt_index: index of the excerpt
            citation_id: unique id of the citation
            section_label: label of the section
            is_citation: citation worthiness label
        """
        # pylint: disable=arguments-differ
        citation_text_embedding = self.text_field_embedder(citation_text)
        citation_text_mask = util.get_text_field_mask(citation_text)

        # shape: [batch, sent, output_dim]
        encoded_citation_text = self.citation_text_encoder(citation_text_embedding, citation_text_mask)

        # shape: [batch, output_dim]
        attn_dist, encoded_citation_text = self.attention_seq2seq(encoded_citation_text, return_attn_distribution=True)

        # In training mode, labels are the citation intents
        # If in predict_mode, predict the citation intents
        if labels is not None:
            logits = self.classifier_feedforward(encoded_citation_text)
            class_probs = F.softmax(logits, dim=1)

            output_dict = {"logits": logits}

            loss = self.loss(logits, labels)
            output_dict["loss"] = loss

            # compute F1 per label
            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(class_probs, labels)
            output_dict['labels'] = labels

        if section_label is not None:  # this is the first scaffold task
            logits = self.classifier_feedforward_2(encoded_citation_text)
            class_probs = F.softmax(logits, dim=1)
            output_dict = {"logits": logits}
            loss = self.loss(logits, section_label)
            output_dict["loss"] = loss
            for i in range(self.num_classes_sections):
                metric = self.label_f1_metrics_sections[self.vocab.get_token_from_index(index=i, namespace="section_labels")]
                metric(logits, section_label)

        if is_citation is not None:  # second scaffold task
            logits = self.classifier_feedforward_3(encoded_citation_text)
            class_probs = F.softmax(logits, dim=1)
            output_dict = {"logits": logits}
            loss = self.loss(logits, is_citation)
            output_dict["loss"] = loss
            for i in range(self.num_classes_cite_worthiness):
                metric = self.label_f1_metrics_cite_worthiness[
                    self.vocab.get_token_from_index(index=i, namespace="cite_worthiness_labels")]
                metric(logits, is_citation)

        if self.predict_mode:
            logits = self.classifier_feedforward(encoded_citation_text)
            class_probs = F.softmax(logits, dim=1)
            output_dict = {"logits": logits}

        output_dict['citing_paper_id'] = citing_paper_id
        output_dict['cited_paper_id'] = cited_paper_id
        output_dict['citation_excerpt_index'] = citation_excerpt_index
        output_dict['citation_id'] = citation_id
        output_dict['attn_dist'] = attn_dist  # also return attention distribution for analysis
        output_dict['citation_text'] = citation_text['tokens']
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                 for x in argmax_indices]
        output_dict['probabilities'] = class_probabilities
        output_dict['positive_labels'] = labels
        output_dict['prediction'] = labels
        citation_text = []
        for batch_text in output_dict['citation_text']:
            citation_text.append([self.vocab.get_token_from_index(token_id.item()) for token_id in batch_text])
        output_dict['citation_text'] = citation_text
        output_dict['all_labels'] = [self.vocab.get_index_to_token_vocabulary(namespace="labels")
                                     for _ in range(output_dict['logits'].shape[0])]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}

        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            metric_dict[name + '_P'] = metric_val[0]
            metric_dict[name + '_R'] = metric_val[1]
            metric_dict[name + '_F1'] = metric_val[2]
            if name != 'none':  # do not consider `none` label in averaging F1
                sum_f1 += metric_val[2]

        names = list(self.label_f1_metrics.keys())
        total_len = len(names) if 'none' not in names else len(names) - 1
        average_f1 = sum_f1 / total_len
        # metric_dict['combined_metric'] = (accuracy + average_f1) / 2
        metric_dict['average_F1'] = average_f1

        if self.report_auxiliary_metrics:
            sum_f1 = 0.0
            for name, metric in self.label_f1_metrics_sections.items():
                metric_val = metric.get_metric(reset)
                metric_dict['aux-sec--' + name + '_P'] = metric_val[0]
                metric_dict['aux-sec--' + name + '_R'] = metric_val[1]
                metric_dict['aux-sec--' + name + '_F1'] = metric_val[2]
                if name != 'none':  # do not consider `none` label in averaging F1
                    sum_f1 += metric_val[2]
            names = list(self.label_f1_metrics_sections.keys())
            total_len = len(names) if 'none' not in names else len(names) - 1
            average_f1 = sum_f1 / total_len
            # metric_dict['combined_metric'] = (accuracy + average_f1) / 2
            metric_dict['aux-sec--' + 'average_F1'] = average_f1

            sum_f1 = 0.0
            for name, metric in self.label_f1_metrics_cite_worthiness.items():
                metric_val = metric.get_metric(reset)
                metric_dict['aux-worth--' + name + '_P'] = metric_val[0]
                metric_dict['aux-worth--' + name + '_R'] = metric_val[1]
                metric_dict['aux-worth--' + name + '_F1'] = metric_val[2]
                if name != 'none':  # do not consider `none` label in averaging F1
                    sum_f1 += metric_val[2]
            names = list(self.label_f1_metrics_cite_worthiness.keys())
            total_len = len(names) if 'none' not in names else len(names) - 1
            average_f1 = sum_f1 / total_len
            # metric_dict['combined_metric'] = (accuracy + average_f1) / 2
            metric_dict['aux-worth--' + 'average_F1'] = average_f1

        return metric_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ScaffoldBilstmAttentionClassifier':
        with_elmo = params.pop_bool("with_elmo", False)
        if with_elmo:
            embedder_params = params.pop("elmo_text_field_embedder")
        else:
            embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(embedder_params, vocab=vocab)
        # citation_text_encoder = Seq2VecEncoder.from_params(params.pop("citation_text_encoder"))
        citation_text_encoder = Seq2SeqEncoder.from_params(params.pop("citation_text_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))
        classifier_feedforward_2 = FeedForward.from_params(params.pop("classifier_feedforward_2"))
        classifier_feedforward_3 = FeedForward.from_params(params.pop("classifier_feedforward_3"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        use_lexicon = params.pop_bool("use_lexicon_features", False)
        use_sparse_lexicon_features = params.pop_bool("use_sparse_lexicon_features", False)
        data_format = params.pop('data_format')

        report_auxiliary_metrics = params.pop_bool("report_auxiliary_metrics", False)

        predict_mode = params.pop_bool("predict_mode", False)
        print(f"pred mode: {predict_mode}")

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   citation_text_encoder=citation_text_encoder,
                   classifier_feedforward=classifier_feedforward,
                   classifier_feedforward_2=classifier_feedforward_2,
                   classifier_feedforward_3=classifier_feedforward_3,
                   initializer=initializer,
                   regularizer=regularizer,
                   report_auxiliary_metrics=report_auxiliary_metrics,
                   predict_mode=predict_mode)


def new_parameter(*size):
    out = Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


class Attention(nn.Module):
    """ Simple multiplicative attention"""
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in, reduction_dim=-2, return_attn_distribution=False):
        """
        return_attn_distribution: if True it will also return the original attention distribution

        this reduces the one before last dimension in x_in to a weighted sum of the last dimension
        e.g., x_in.shape == [64, 30, 100] -> output.shape == [64, 100]
        Usage: You have a sentence of shape [batch, sent_len, embedding_dim] and you want to
            represent sentence to a single vector using attention [batch, embedding_dim]

        Here we use it to aggregate the lexicon-aware representation of the sentence
        In two steps we convert [batch, sent_len, num_words_in_category, num_categories] into [batch, num_categories]
        """
        # calculate attn weights
        attn_score = torch.matmul(x_in, self.attention).squeeze()
        # add one dimension at the end and get a distribution out of scores
        attn_distrib = F.softmax(attn_score.squeeze(), dim=-1).unsqueeze(-1)
        scored_x = x_in * attn_distrib
        weighted_sum = torch.sum(scored_x, dim=reduction_dim)
        if return_attn_distribution:
            return attn_distrib.reshape(x_in.shape[0], -1), weighted_sum
        else:
            return weighted_sum
