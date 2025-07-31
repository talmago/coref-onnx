from typing import (
    List,
    Tuple,
)

from spacy import Language
from spacy.tokens import Doc, Span

from coref_onnx import CoreferenceResolver


class SpaCyCorefComponent:
    """

    ```python
    import spacy

    from coref_onnx import create_coref_minilm_component

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("coref_minilm")

    doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")

    print(doc._.coref_clusters[0])
    print(doc._.cluster_heads)
    print(doc._.resolved_text)

    # Output is:
    # [Barack Obama, He]
    # {'Barack Obama': Barack Obama}
    # Barack Obama was born in Hawaii. Barack Obama was elected president in 2008.
    ```
    """
    def __init__(self, model_name_or_path: str, **kwargs):
        self.resolver = CoreferenceResolver.from_pretrained(model_name_or_path, **kwargs)

        Doc.set_extension("coref_clusters", default=None)
        Doc.set_extension("resolved_text", default=None)
        Doc.set_extension("cluster_heads", default=None)

    def __call__(self, doc: Doc) -> Doc:
        tokens = [token.text for token in doc]
        clusters = self.resolver([tokens])["clusters"][0]

        span_clusters = self._convert_clusters_to_spans(doc, clusters)
        doc._.coref_clusters = span_clusters
        doc._.resolved_text = self._resolve_text(doc, span_clusters)
        doc._.cluster_heads = self._get_cluster_heads(span_clusters)

        return doc

    @staticmethod
    def _convert_clusters_to_spans(doc: Doc, clusters: List[List[Tuple[int, int]]]) -> List[List[Span]]:
        """Convert index-based clusters to spaCy span clusters."""
        span_clusters = []
        for cluster in clusters:
            span_cluster = []
            for start, end in cluster:
                # Ensure valid range (inclusive end -> exclusive)
                span = doc[start : end + 1]
                span_cluster.append(span)
            span_clusters.append(span_cluster)
        return span_clusters

    @staticmethod
    def _resolve_text(doc: Doc, clusters: List[List[Span]]) -> str:
        token_to_head = {}

        for cluster in clusters:
            if not cluster:
                continue
            head_text = cluster[0].text

            for span in cluster[1:]:
                token_to_head[span.start] = head_text
                for i in range(span.start + 1, span.end):
                    token_to_head[i] = ""

        resolved_text = "".join(
            token_to_head.get(i, token.text) + token.whitespace_
            for i, token in enumerate(doc)
            if token_to_head.get(i, token.text) != ""
        )

        return resolved_text.strip()

    @staticmethod
    def _get_cluster_heads(clusters: List[List[Span]]):
        """Return a dict mapping head span text to the Span itself."""
        heads = {}
        for cluster in clusters:
            if not cluster:
                continue
            head_span = cluster[0]
            heads[head_span.text] = head_span
        return heads


@Language.factory("coref_minilm")
def create_coref_minilm_component(nlp, name):  # noqa: ARG001
    return SpaCyCorefComponent(
        model_name_or_path="talmago/allennlp-coref-onnx-mMiniLMv2-L12-H384-distilled-from-XLMR-Large"
    )
