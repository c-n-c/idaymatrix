### Introduction

A small NLP kata to solve for hidden words in a matrix limited to a natural language context.

### Pre-Requisite

- you need to have PDM, Wikipedia library and nltk library

- nltk errors are thrown if there are not supporting ntlk resources

nltk.download(<resources>) should solve the issues

### How to run

- 1. Build the corpus

    `pdm run python wiki_corpus.py`

- 2. Run the solver
    `pdm run python main.py`

- 3. Performance evaluation

    evaluate performance using time:

    `time pdm run python main.py`
