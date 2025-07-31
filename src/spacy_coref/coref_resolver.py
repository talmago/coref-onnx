import json
import os
from typing import (
    Iterable,
    Optional,
)

import numpy as np
import onnxruntime
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Metaspace


def init_onnxruntime_session(path_or_bytes: str | bytes | os.PathLike):
    """Init ONNXRuntime inference session.

    Args:
        path_or_bytes: Filename or serialized ONNX or ORT format model in a byte string.

    Returns:
        InferenceSession
    """
    available_providers = onnxruntime.get_available_providers()

    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = onnxruntime.InferenceSession(path_or_bytes, providers=providers)
    return session


class CoreferenceResolver:
    """
    A lightweight coreference resolution model using ONNX runtime.

    This class loads a pre-exported ONNX model and SentencePiece tokenizer to perform span-based coreference resolution.

    Usage example:

    ```python
    from coref_onnx import CoreferenceResolver, decode_clusters

    resolver = CoreferenceResolver.from_pretrained("talmago/allennlp-coref-onnx-mMiniLMv2-L12-H384-distilled-from-XLMR-Large")

    sentences = [
        ["Barack", "Obama", "was", "the", "44th", "President", "of", "the", "United", "States", "."],
        ["He", "was", "born", "in", "Hawaii", "."]
    ]

    pred = resolver(sentences)

    print(decode_clusters(sentences, pred["clusters"][0]))

    # Output is:
    # [['Barack Obama', 'He']]
    ```
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

        self.session = init_onnxruntime_session(model_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.pre_tokenizer = Metaspace(
            replacement="▁", prepend_scheme="always", split=True
        )
        self.max_spans = max_spans
        self.max_span_width = max_span_width

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "CoreferenceResolver":
        """
        Instantiate a resolver from a model name or local path.

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
    ) -> dict:
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

        clusters = self.agg_clusters(top_spans, antecedent_indices, predicted_antecedents)

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

    @staticmethod
    def pad_or_truncate_spans(spans, max_spans=128):
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

    @staticmethod
    def agg_clusters(top_spans, antecedent_indices, predicted_antecedents):
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


def decode_clusters(sentences, clusters):
    flat_tokens = [token for sent in sentences for token in sent]
    decoded_clusters = []
    max_index = len(flat_tokens) - 1
    for cluster in clusters:
        decoded_spans = []
        for span in cluster:
            start, end = span
            if not (0 <= start <= end <= max_index):
                continue
            decoded_spans.append(" ".join(flat_tokens[start : end + 1]))
        decoded_clusters.append(decoded_spans)
    return decoded_clusters
