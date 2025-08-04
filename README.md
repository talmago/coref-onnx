# spacy-coref

Lightweight, fast co-reference resolution using a distilled version of AllenNLP's coreference model (exported to ONNX). 

## ✨ Features

- 🧠 Cross-lingual coreference resolution
- 🪶 Lightweight model based on AllenNLP’s coref modeling
- ⚡  Fast inference via ONNX
- 🔌 Easy integration with spaCy

---

## 📦 Installation

```bash
$ pip install spacy-coref
```

## 🚀 Quickstart

Usage as a standalone component

```python
from spacy_coref import CoreferenceResolver, decode_clusters

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

Usage with spaCy

```python
import spacy
import spacy_coref

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

## 🛠️ Development

Set up virtualenv

```sh
$ make env
```

Set PYTHONPATH

```sh
$ export PYTHONPATH=$PYTHONPATH:/Users/talmago/git/spacy-coref/src
```

Code formatting

```sh
$ make format
```

### 📚 References

This project builds on the work of the following repositories:

- **[crosslingual-coreference](https://github.com/davidberenstein1957/crosslingual-coreference)**  
  David Berenstein's implementation of multilingual coreference resolution, adapted from the original AllenNLP coref model.  
  GitHub: [davidberenstein1957/crosslingual-coreference](https://github.com/davidberenstein1957/crosslingual-coreference)

- **[AllenNLP coreference model](https://github.com/allenai/allennlp-models/tree/b1f372248c17ad12684d344955fbcd98e957e77e/allennlp_models/coref)**  
  Official AllenNLP implementation of coreference resolution.  
  GitHub: [allenai/allennlp-models](https://github.com/allenai/allennlp-models)
