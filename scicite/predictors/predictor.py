import json
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

from scicite.data import Citation, read_s2_jsonline
from scicite.helper import JsonFloatEncoder
from scicite.constants import NONE_LABEL_NAME
from scicite.constants import S2_CATEGORIES_MULTICLASS

@Predictor.register('citation_classifier')
class CitationIntentPredictor(Predictor):
    """"Predictor wrapper for the CitationIntentClassifier"""
    
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        return_dict = {}

        for citation in read_s2_jsonline(inputs, multilabel=self._dataset_reader.multilabel):
            if len(citation.text) == 0:
                continue
            if not self._dataset_reader.multilabel and len(citation.intent) > 1:
                raise(f"specified not multilabel but multiple positive labels found")
            intents = citation.intent[0] if not self._dataset_reader.multilabel else citation.intent
            instance = self._dataset_reader.text_to_instance(
                citation_text=citation.text,
                intent=intents,
                citing_paper_id=citation.citing_paper_id,
                cited_paper_id=citation.cited_paper_id,
                citation_excerpt_index=citation.citation_excerpt_index
            )
            outputs = self._model.forward_on_instance(instance)
            if self._dataset_reader.multilabel:
                predictions = {outputs['positive_labels'][i]: outputs['class_probs'][i]
                                 for i in range(len(outputs['positive_labels']))}
            else:
                predictions = {}
                for label, idx in S2_CATEGORIES_MULTICLASS.items():
                    if label != NONE_LABEL_NAME:
                        predictions[label] = outputs['class_probs'][idx]
                # label_to_index = {v: k for k, v in outputs['all_labels'].items()}
                # for i, prob in enumerate(outputs['class_probs']):
                #     if NONE_LABEL_NAME not in outputs['all_labels'][i]:
                #         predictions.append({outputs['all_labels'][i]: prob})
                # prediction_index = label_to_index[outputs['positive_labels']]
                # predictions = {outputs['positive_labels']: outputs['class_probs'][prediction_index]}
            return_dict['citingPaperId'] = outputs['citing_paper_id']
            return_dict['citedPaperId'] = outputs['cited_paper_id']
            if 'excerptCitationIntents' not in return_dict:
                return_dict['excerptCitationIntents'] = []
            return_dict['excerptCitationIntents'].append(predictions)
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
