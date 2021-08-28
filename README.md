### Introduction

An elementary NLP kata to solve for hidden words in a matrix. Words are limited to a natural language context.

### Pre-Requisite

- Python 3.9+

- A supercool package manager [PDM](https://pdm.fming.dev/),

- Python libraries :
    - Wikipedia library
    - nltk library

- nltk errors are thrown if there are no corpora support resources like stopwords

  `nltk.download(stopwords)` should solve the issue

### Code structure
```
.
|-- README.md
|-- __init__.py
|-- data
|-- main.py
|-- pyproject.toml
`-- wiki_corpus.py
```

### How to run

- 1. Install project dependencies
    `pdm install`

- 2. Build the corpus

    `pdm run python wiki_corpus.py`

- 3. Run the solver
    `pdm run python main.py`

- 4. Performance evaluation

     Use time utility to get performance insights.

    `time pdm run python main.py`
