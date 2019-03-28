from pathlib import Path

Aclarc_Format_Nested_Jsonlines = 'jurgens'
Aclarc_Format_Flat_Jsonlines = 'jurgens_jl'
Scicite_Format_Flat_Jsonlines = 'scicite_flat_jsonlines'
Scicite_Format_Nested_Jsonlines = 's2'
CITATION_TOKEN = '@@CITATION'  # a word to replace citation marker

# get absolute path of the project source
root_path = str(Path(__file__).resolve().parent)
patterns_path = root_path + '/resources/patterns/'

# default file extensions for saving the files
MODEL_DEFAULT_NAME = 'model.pkl'
LDA_DEFAULT_NAME = 'lda.pkl'
LDA_VECTORIZER_DEFAULT_NAME = 'ldavec.pkl'
VECTORIZER_DEFAULT_NAME = 'vec.pkl'
BASELINE_DEFAULT_NAME = 'baseline'

NONE_LABEL_NAME = 'other'

NEGATIVE_CLASS_PREFIX = 'not--'

S2_CATEGORIES = {"methodology": 0,
               NEGATIVE_CLASS_PREFIX + "methodology": 1,
               "comparison": 2,
               NEGATIVE_CLASS_PREFIX + "comparison": 3
                 }

S2_CATEGORIES_BINARY = {"methodology": 0,
                       NONE_LABEL_NAME: 1}

S2_CATEGORIES_MULTICLASS = {"background": 0, "method": 1, "result": 2}

JURGENS_FEATURES_PATH = root_path + '/resources/jurgens_features.pkl'
