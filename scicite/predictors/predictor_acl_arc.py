import json
import operator
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

from scicite.data import read_jurgens_jsonline
from scicite.helper import JsonFloatEncoder
from scicite.constants import NONE_LABEL_NAME


@Predictor.register('predictor_aclarc')
class CitationIntentPredictorACL(Predictor):
    """"Predictor wrapper for the CitationIntentClassifier"""
    
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        return_dict = {}
        citation = read_jurgens_jsonline(inputs)
        if len(citation.text) == 0:
            print('empty context, skipping')
            return {}
        print(self._dataset_reader)
        instance = self._dataset_reader.text_to_instance(
            citation_text=citation.text,
            intent=citation.intent,
            citing_paper_id=citation.citing_paper_id,
            cited_paper_id=citation.cited_paper_id,
            citation_excerpt_index=citation.citation_excerpt_index,
            citation_id=citation.citation_id
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
        return_dict['probabilities'] = predictions
        return_dict['prediction'] = label
        return_dict['original_label'] = citation.intent
        # return_dict['attention_dist'] = outputs['attn_dist']
        return return_dict

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        keys = ['citation_id', 'prediction', 'probabilities', 'original_label', 'citation_text', 'attention_dist', 'original_citation_text']
        for k in outputs.copy():
            if k not in keys:
                outputs.pop(k)
        return json.dumps(outputs, cls=JsonFloatEncoder) + "\n"
