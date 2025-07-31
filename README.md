# Coreference Resolution with ONNX (AllenNLP-based)

Lightweight, fast coreference resolution component using a distilled version of AllenNLP's coreference model, exported to ONNX. 

## ✨ Features

- 🧠 Cross-lingual coreference resolution
- 🪶 Lightweight model based AllenNLP’s coreference resolution architecture
- ⚡ Fast inference via ONNX
- 🔌 Easy integration with spaCy

---

## 📦 Installation

```bash
$ pip install coref-onnx
```

## 🚀 Quickstart

Basic spaCy integration

```python
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("coref_minilm")

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")

print(doc._.coref_clusters)
# Output: [[(0, 1), (7, 7)]]

print(doc._.resolved_text)
# Output: Barack Obama was born in Hawaii. Barack Obama was elected president in 2008.

print(doc._.cluster_heads)
# Output: {'Barack Obama': [0, 1]}
```

You can also choose to export other coreference models yourself (e.g. spanBERT) and run with:

```python
nlp.add_pipe("coref", config={
    "model_name_or_path": "..."
})
```

The `model_name_or_path` is either a local or remote directory that includes:
  * **model.onnx** - model weigths exported to onnx graph
  * **tokenizer.json** - tokenizer file
  * **config.json** — metadata including *max_span_width* and *max_spans*
