""" A predictor that works with the flat format of s2 data
i.e., one excerpt per dictionary instead of a nested dict"""
import json
import operator
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

from scicite.data import Citation, read_s2_excerpt
from scicite.helper import JsonFloatEncoder
from scicite.constants import NONE_LABEL_NAME
from scicite.constants import S2_CATEGORIES_MULTICLASS
from scicite.helper import is_sentence


@Predictor.register('predictor_scicite')
class PredictorSciCite(Predictor):
    """"Predictor wrapper for the CitationIntentClassifier"""
    
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        return_dict = {}

        citation = read_s2_excerpt(inputs)
        # skip if the excerpt is not a valid sentence
        if len(citation.text) < 5 or not is_sentence(citation.text):
            return_dict['citingPaperId'] = inputs['citingPaperId']
            return_dict['citedPaperId'] = inputs['citedPaperId']
            return_dict['prediction'] = ''
        else:
            instance = self._dataset_reader.text_to_instance(
                citation_text=citation.text,
                intent=citation.intent,
                citing_paper_id=citation.citing_paper_id,
                cited_paper_id=citation.cited_paper_id,
                citation_excerpt_index=citation.citation_excerpt_index
            )
            outputs = self._model.forward_on_instance(instance)
            predictions = {}

            label_to_index = {v: k for k, v in outputs['all_labels'].items()}
            for i, prob in enumerate(outputs['class_probs']):
                predictions[outputs['all_labels'][i]] = prob

            label = max(predictions.items(), key=operator.itemgetter(1))[0]
            return_dict['citation_id'] = citation.citation_id
            return_dict['citingPaperId'] = outputs['citing_paper_id']
            return_dict['citedPaperId'] = outputs['cited_paper_id']
            return_dict['citation_id'] = citation.citation_id
            return_dict['probabilities'] = predictions
            return_dict['prediction'] = label
            return_dict['original_label'] = citation.intent
            return_dict['citation_text'] = outputs['citation_text']
            return_dict['original_citation_text'] = citation.text
            return_dict['attention_dist'] = outputs['attn_dist']
        return return_dict

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        keys = ['citedPaperId', 'citingPaperId', 'excerptCitationIntents']
        for k in outputs.copy():
            if k not in keys:
                outputs.pop(k)
        return json.dumps(outputs, cls=JsonFloatEncoder) + "\n"
