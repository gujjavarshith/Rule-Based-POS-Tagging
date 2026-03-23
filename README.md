# Rule-Based-POS-Tagging

## Dataset

This project uses the **English Web Treebank (en_ewt)** dataset from **Universal Dependencies (UD)** for developing and evaluating the POS tagging models. The dataset is located in the `Dataset/` directory.

The dataset is provided in the **CoNLL-U** format (`.conllu`), which contains rich annotations for each sentence, including word forms, lemmas, part-of-speech (POS) tags, morphological features, and dependency relations.

### CoNLL-U Format Overview

Each word or token in a sentence is represented on a single line with 10 tab-separated fields:

1. **ID**: Word index (integer starting at 1 for each new sentence).
2. **FORM**: Word form or punctuation symbol.
3. **LEMMA**: Lemma or stem of the word form.
4. **UPOS**: Universal part-of-speech tag.
5. **XPOS**: Language-specific part-of-speech tag.
6. **FEATS**: List of morphological features.
7. **HEAD**: Head of the current word, which is either a value of ID or zero (0).
8. **DEPREL**: Universal dependency relation to the HEAD.
9. **DEPS**: Enhanced dependency graph in the form of a list of head-deprel pairs.
10. **MISC**: Any other annotation.
