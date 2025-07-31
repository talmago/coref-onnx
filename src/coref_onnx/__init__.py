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

import json
import os
from typing import (
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np
import onnxruntime
from huggingface_hub import snapshot_download
from spacy.language import Language
from spacy.tokens import Doc
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Metaspace


class CoreferenceResolver:
    """
    A lightweight coreference resolution model using ONNX runtime.

    This class loads a pre-exported ONNX model and SentencePiece tokenizer
    to perform span-based coreference resolution.
    """

    def __init__(self, model_dir: str, max_spans: int = 512, max_span_width: int = 5):
        """
        Initialize the coreference resolver.

        Args:
            model_dir (str): Path to the directory containing the ONNX model and tokenizer.
            max_span_width (int): Maximum width (in tokens) of candidate spans to consider.
        """
        model_path = os.path.join(model_dir, "model.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        self.session = onnxruntime.InferenceSession(model_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.pre_tokenizer = Metaspace(
            replacement="▁", prepend_scheme="always", split=True
        )
        self.max_spans = max_spans
        self.max_span_width = max_span_width

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Instantiate a CoreferenceResolver from a model name or local path.

        Args:
            model_name_or_path (str): Local path or HuggingFace model repo ID.

        Returns:
            CoreferenceResolver: An initialized instance.
        """
        local_dir = kwargs.get("local_dir", None)

        if os.path.isdir(model_name_or_path):
            model_dir = model_name_or_path
        else:
            model_dir = snapshot_download(
                repo_id=model_name_or_path,
                local_dir=local_dir,
                allow_patterns=[
                    "*.onnx",
                    "*.json",
                ],
            )

        # Load configuration
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Could not find model configuration in {model_dir}/config.json"
            )

        with open(config_path, "r") as f:
            config = json.load(f)

        max_span_width = config.get("max_span_width", 5)
        max_spans = config.get("max_spans", 5)

        return cls(model_dir=model_dir, max_spans=max_spans, max_span_width=max_span_width)

    def __call__(
        self,
        text: list[str] | list[list[str]],
        spans: Optional[list[tuple[int, int]]] = None,
    ):
        """
        Resolve coreference clusters for the input text.

        Args:
            text (List[str] | List[List[str]]): A list of tokens or list of list of tokens (sentences).
            spans (Optional[List[Tuple[int, int]]]): Predefined spans to consider. If not provided, spans will be enumerated.

        Returns:
            dict: {
                "clusters": List of resolved clusters (as span index tuples),
                "top_spans": Top spans considered by the model,
                "antecedent_indices": Index of each span's antecedent candidates,
                "predicted_antecedents": Index of selected antecedents (or -1 if none)
            }
        """
        inputs = self._prepare_inputs(text, spans=spans)

        top_spans, antecedent_indices, predicted_antecedents = self.session.run(
            None, inputs
        )

        clusters = self._agg_clusters(top_spans, antecedent_indices, predicted_antecedents)

        return {
            "clusters": clusters,
            "top_spans": top_spans,
            "antecedent_indices": antecedent_indices,
            "predicted_antecedents": predicted_antecedents,
        }

    def _prepare_inputs(
        self,
        text: list[str] | list[list[str]],
        spans: Optional[list[tuple[int, int]]] = None,
    ):
        """
        Tokenize and format input text and spans into ONNX input format.

        Args:
            text (List[str] | List[List[str]]): Input text tokens.
            spans (Optional[List[Tuple[int, int]]]): Optional list of spans.

        Returns:
            Dict[str, np.ndarray]: Dictionary of input tensors for the ONNX model.
        """
        if isinstance(text, Iterable) and isinstance(text[0], list):
            flat_text = [token for sent in text for token in sent]
        else:
            flat_text = text

        encoding = self.tokenizer.encode(flat_text, is_pretokenized=True)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        word_ids = encoding.word_ids

        # Map original words to subword token indices
        orig_to_subword = []
        current_word = None
        start = None
        for i, word_id in enumerate(word_ids):
            if word_id != current_word:
                if current_word is not None:
                    orig_to_subword.append([start, i - 1])
                if word_id is not None:
                    start = i
                current_word = word_id
        if current_word is not None and start is not None:
            orig_to_subword.append([start, len(word_ids) - 1])

        seq_len = len(orig_to_subword)
        offsets = np.array([orig_to_subword], dtype=np.int64)  # shape: (1, seq_len, 2)
        mask = np.ones((1, seq_len), dtype=bool)
        segment_concat_mask = np.ones_like(input_ids, dtype=bool)

        spans = spans or self._enumerate_spans(seq_len)
        span_arr = np.array([spans], dtype=np.int64)
        spans_tensor = self.pad_or_truncate_spans(span_arr, max_spans=self.max_spans)

        return {
            "token_ids": input_ids,
            "mask": mask,
            "segment_concat_mask": segment_concat_mask,
            "offsets": offsets,
            "spans": spans_tensor,
        }

    def pad_or_truncate_spans(self, spans, max_spans=128):
        """
        Pad or truncate the span tensor to match a fixed max length.

        Args:
            spans (np.ndarray): Span tensor of shape (1, num_spans, 2).
            max_spans (int): Desired fixed length.

        Returns:
            np.ndarray: Padded/truncated span tensor of shape (1, max_spans, 2).
        """
        batch_size, num_spans, _ = spans.shape
        if num_spans > max_spans:
            spans = spans[:, :max_spans, :]
        elif num_spans < max_spans:
            pad_len = max_spans - num_spans
            pad = np.full((batch_size, pad_len, 2), fill_value=-1, dtype=spans.dtype)
            spans = np.concatenate([spans, pad], axis=1)
        return spans

    def _enumerate_spans(self, seq_len: int) -> list[list[int]]:
        """
        Generate all possible spans up to max_span_width.

        Args:
            seq_len (int): Number of tokens in the sequence.

        Returns:
            List[Tuple[int, int]]: Candidate spans.
        """
        spans = []
        for start in range(seq_len):
            for end in range(start, min(start + self.max_span_width, seq_len)):
                spans.append((start, end))
        return spans

    def _agg_clusters(self, top_spans, antecedent_indices, predicted_antecedents):
        """
        Construct coreference clusters based on top spans and antecedents.

        Args:
            top_spans (np.ndarray): Top spans selected by the model.
            antecedent_indices (np.ndarray): Candidate antecedents for each span.
            predicted_antecedents (np.ndarray): Final predicted antecedents.

        Returns:
            List[List[Tuple[int, int]]]: Clustered span indices per document.
        """
        batch_clusters = []
        batch_size = top_spans.shape[0]

        for b in range(batch_size):
            spans_to_cluster_id = {}
            clusters = []

            for i, predicted_antecedent in enumerate(predicted_antecedents[b]):
                if predicted_antecedent < 0:
                    continue  # No antecedent: skip

                predicted_index = int(antecedent_indices[b, i, predicted_antecedent])
                antecedent_span = tuple(top_spans[b, predicted_index].tolist())

                if antecedent_span in spans_to_cluster_id:
                    cluster_id = spans_to_cluster_id[antecedent_span]
                else:
                    cluster_id = len(clusters)
                    clusters.append([antecedent_span])
                    spans_to_cluster_id[antecedent_span] = cluster_id

                current_span = tuple(top_spans[b, i].tolist())
                clusters[cluster_id].append(current_span)
                spans_to_cluster_id[current_span] = cluster_id

            # Sort spans in each cluster by start index
            for cluster in clusters:
                cluster.sort(key=lambda span: span[0])

            # Remove singleton clusters
            clusters = [c for c in clusters if len(c) > 1]
            batch_clusters.append(clusters)

        return batch_clusters


@Language.factory("coref")
def create_coref_component(nlp, name, model_name_or_path):  # noqa: ARG001
    return SpaCyCorefComponent(model_name_or_path=model_name_or_path)


@Language.factory("coref_minilm")
def create_coref_minilm_component(nlp, name):  # noqa: ARG001
    return SpaCyCorefComponent(
        model_name_or_path="talmago/allennlp-coref-onnx-mMiniLMv2-L12-H384-distilled-from-XLMR-Large"
    )


class SpaCyCorefComponent:
    def __init__(self, model_name_or_path: str, **kwargs):
        self.resolver = CoreferenceResolver.from_pretrained(model_name_or_path, **kwargs)

        Doc.set_extension("coref_clusters", default=None)
        Doc.set_extension("resolved_text", default=None)
        Doc.set_extension("cluster_heads", default=None)

    def __call__(self, doc: Doc) -> Doc:
        tokens = [token.text for token in doc]
        clusters = self.resolver([tokens])["clusters"][0]

        doc._.coref_clusters = clusters
        doc._.resolved_text = self._resolve_text(doc, clusters)
        doc._.cluster_heads = self._get_cluster_heads(doc, clusters)

        return doc

    def _resolve_text(self, doc: Doc, clusters: List[List[Tuple[int, int]]]) -> str:
        """
        Resolves coreference clusters in a spaCy Doc and returns a string with coreferent mentions
        replaced by their cluster heads, preserving original whitespace and punctuation.

        Args:
            doc (Doc): The spaCy document to resolve.
            clusters (List[List[Tuple[int, int]]]): A list of coreference clusters,
                each containing spans as (start_token_index, end_token_index).

        Returns:
            str: The coreference-resolved text.
        """
        token_to_head = {}

        for cluster in clusters:
            if not cluster:
                continue
            head_span = cluster[0]
            head_text = doc[head_span[0] : head_span[1] + 1].text

            for span in cluster[1:]:
                # Replace the first token in the span with the head text
                token_to_head[span[0]] = head_text
                # Mark the rest for removal
                for i in range(span[0] + 1, span[1] + 1):
                    token_to_head[i] = ""

        resolved_text = "".join(
            token_to_head.get(i, token.text) + token.whitespace_
            for i, token in enumerate(doc)
            if token_to_head.get(i, token.text) != ""
        )

        return resolved_text.strip()

    def _get_cluster_heads(self, doc: Doc, clusters: List[List[Tuple[int, int]]]):
        heads = {}
        for cluster in clusters:
            if not cluster:
                continue
            span = cluster[0]
            head = doc[span[0] : span[1] + 1]
            heads[head.text] = [span[0], span[1]]
        return heads
