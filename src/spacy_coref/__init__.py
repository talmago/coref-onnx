"""
coref_onnx
~~~~~~~~~~

Lightweight crosslingual coreference resolution using ONNX Runtime
and distilled transformer models.
"""

__version__ = "0.1.0"
__author__ = "Tal Almagor"
__license__ = "MIT"
__email__ = "almagoric@gmail.com"

from spacy_coref.coref_resolver import (  # noqa: F401
    CoreferenceResolver,
    decode_clusters,
)
from spacy_coref.spacy_component import (  # noqa: F401
    SpaCyCorefComponent,
    create_coref_minilm_component,
)
