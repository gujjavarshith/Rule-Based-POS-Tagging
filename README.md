# Rule-Based POS Tagger

A cascading rule-based Part-of-Speech (POS) tagger for English, built on the [Universal Dependencies English Web Treebank (UD EWT)](https://universaldependencies.org/treebanks/en_ewt/index.html). The tagger assigns one of 17 Universal POS (UPOS) tags to each token using a priority-ordered pipeline of handcrafted linguistic rules — no machine learning required.

> **Overall Accuracy:** **87.51%** on the UD EWT test set (with innovation modules enabled)

---

## Table of Contents

- [Overview](#overview)
- [Universal POS Tag Set](#universal-pos-tag-set)
- [Architecture](#architecture)
  - [Tagging Pipeline](#tagging-pipeline)
  - [Innovation Modules](#innovation-modules)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Setup](#data-setup)
- [Usage](#usage)
  - [Full Pipeline](#full-pipeline)
  - [Tag a Sentence](#tag-a-sentence)
  - [Evaluate on Dev/Test](#evaluate-on-devtest)
  - [Ablation Study](#ablation-study)
- [Results](#results)
  - [Per-Tag F1 Scores](#per-tag-f1-scores)
  - [Ablation Study Results](#ablation-study-results)
- [Notebooks](#notebooks)
- [License](#license)

---

## Overview

This project implements a **rule-based POS tagger** that uses a cascading pipeline of six rule layers to assign Universal POS tags. Each token passes through the rules in priority order, and the **first rule that fires** determines the tag. If no rule matches, the token defaults to `NOUN`.

The system includes four **innovation modules** that extend the baseline with prefix-based tagging, web-token detection, multi-word compound recognition, and post-hoc context correction — all derived from error analysis on the dev set.

---

## Universal POS Tag Set

The tagger uses the **17 UPOS tags** defined by the [Universal Dependencies](https://universaldependencies.org/u/pos/) project:

| Tag | Category | Description | Examples |
|-----|----------|-------------|----------|
| `ADJ` | Open class | Adjective | *big, old, green, first* |
| `ADP` | Closed class | Adposition (preposition/postposition) | *in, to, during, of* |
| `ADV` | Open class | Adverb | *very, tomorrow, down, where* |
| `AUX` | Closed class | Auxiliary verb | *is, has, will, should, can* |
| `CCONJ` | Closed class | Coordinating conjunction | *and, or, but* |
| `DET` | Closed class | Determiner | *a, the, this, every* |
| `INTJ` | Open class | Interjection | *oh, wow, yes, hello* |
| `NOUN` | Open class | Noun | *girl, cat, tree, air* |
| `NUM` | Open class | Numeral | *1, 2025, one, seventy-seven* |
| `PART` | Closed class | Particle | *not, 's, to* (infinitive marker) |
| `PRON` | Closed class | Pronoun | *I, you, he, myself, who* |
| `PROPN` | Open class | Proper noun | *Mary, London, NASA* |
| `PUNCT` | Other | Punctuation | *. , ; : ! ?* |
| `SCONJ` | Closed class | Subordinating conjunction | *if, while, that, because* |
| `SYM` | Other | Symbol | *$, %, §, ©, +, :), #* |
| `VERB` | Open class | Verb | *run, eat, runs, ate, running* |
| `X` | Other | Other (foreign words, typos, etc.) | *guten, xfgh, etc.* |

---

## Architecture

### Tagging Pipeline

The tagger applies rules in **four passes** over each sentence. Within each pass, rules are checked in strict priority order — the first rule to return a tag wins.

```
                        ┌─────────────────────────┐
                        │   Input Sentence         │
                        │ ["The", "cat", "sat"]    │
                        └────────────┬────────────┘
                                     │
            ╔════════════════════════╧════════════════════════╗
            ║          PASS 1 — Deterministic Rules          ║
            ║  ┌──────────────────────────────────────────┐  ║
            ║  │ 1. Punctuation / Numeral (PUNCT, SYM, NUM)│  ║
            ║  │ 2. Closed-class words (DET, PRON, AUX..) │  ║
            ║  └──────────────────────────────────────────┘  ║
            ╚════════════════════════╤════════════════════════╝
                                     │
            ╔════════════════════════╧════════════════════════╗
            ║       PASS 2 — Lexicon + Form-based Rules      ║
            ║  ┌──────────────────────────────────────────┐  ║
            ║  │ 3. Lexicon lookup (most-frequent tag)    │  ║
            ║  │ 4. Capitalization → PROPN                │  ║
            ║  │ 5. Morphology (suffix rules)             │  ║
            ║  │    + Innovation rules (if enabled)       │  ║
            ║  └──────────────────────────────────────────┘  ║
            ╚════════════════════════╤════════════════════════╝
                                     │
            ╔════════════════════════╧════════════════════════╗
            ║         PASS 3 — Context Rules                 ║
            ║  ┌──────────────────────────────────────────┐  ║
            ║  │ 6. Neighbouring-tag context heuristics    │  ║
            ║  │ 7. Default fallback → NOUN               │  ║
            ║  └──────────────────────────────────────────┘  ║
            ╚════════════════════════╤════════════════════════╝
                                     │
            ╔════════════════════════╧════════════════════════╗
            ║   PASS 4 — Context Override (Innovation only)  ║
            ║  ┌──────────────────────────────────────────┐  ║
            ║  │ Brill-style post-hoc correction rules    │  ║
            ║  │ (fixes AUX↔VERB, PART↔ADP, PRON↔SCONJ)  │  ║
            ║  └──────────────────────────────────────────┘  ║
            ╚════════════════════════╤════════════════════════╝
                                     │
                        ┌────────────┴────────────┐
                        │   Output Tags            │
                        │ ["DET", "NOUN", "VERB"]  │
                        └─────────────────────────┘
```

### Innovation Modules

Four modules extend the baseline tagger (located in `innovation/`):

| Module | File | What It Does |
|--------|------|--------------|
| **Prefix Rules** | `prefix_rules.py` | Tags OOV words using English prefixes (`un-` → ADJ, `re-` → VERB, `anti-` → ADJ, etc.) |
| **Web Token Rules** | `web_token_rules.py` | Detects URLs, emails, hashtags, @mentions, timestamps, dates, and decorative punctuation |
| **Compound Context** | `compound_context.py` | Recognises multi-word proper nouns (*"New York"*, *"Wall Street"*), compound adpositions (*"because of"*), and sentence-initial proper nouns |
| **Context Override** | `context_override.py` | Post-processing Brill-style rules that fix systematic errors (e.g., possessive *'s* AUX→PART, *to* PART→ADP before nouns, *have/do* AUX→VERB as main verbs) |

---

## Project Structure

```
Rule-Based-POS-Tagging/
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── README.md                     # This file
│
├── data/                         # UD English Web Treebank (CoNLL-U)
│   ├── en_ewt-ud-train.conllu    #   Training set (~12,543 sentences)
│   ├── en_ewt-ud-dev.conllu      #   Development set (~2,002 sentences)
│   └── en_ewt-ud-test.conllu     #   Test set (~2,077 sentences)
│
├── src/                          # Core tagger source code
│   ├── __init__.py
│   ├── parser.py                 #   CoNLL-U file parser
│   ├── lexicon.py                #   Word → most-frequent-tag lexicon
│   ├── tagger.py                 #   Main cascading rule tagger
│   ├── evaluate.py               #   Accuracy, F1, confusion matrix
│   └── rules/                    #   Baseline rule modules
│       ├── __init__.py
│       ├── punct_num.py          #     Punctuation, symbol, numeral detection
│       ├── closed_class.py       #     Closed-class word lists (DET, PRON, AUX, etc.)
│       ├── capitalization.py     #     Casing-based PROPN detection
│       ├── morphology.py         #     Suffix-based tagging (-tion → NOUN, -ly → ADV)
│       └── context.py            #     Neighbouring-tag context heuristics
│
├── innovation/                   # Extended rule modules (innovation layer)
│   ├── __init__.py
│   ├── prefix_rules.py           #   Prefix-based OOV tagging
│   ├── web_token_rules.py        #   Web/social token detection
│   ├── compound_context.py       #   Multi-word expression rules
│   └── context_override.py       #   Post-hoc Brill-style corrections
│
├── notebooks/                    # Jupyter analysis notebooks
│   ├── 01_eda.ipynb              #   Exploratory data analysis
│   ├── 02_baseline_eval.ipynb    #   Baseline evaluation & visualisation
│   ├── 03_error_analysis.ipynb   #   Error analysis & confusion pairs
│   ├── 04_ablation.ipynb         #   Ablation study
│   ├── 05_rule_coverage_and_oov_analysis.ipynb  # OOV & coverage analysis
│   └── 06_innovation.ipynb       #   Innovation module evaluation
│
├── outputs/                      # Generated results & plots
│   ├── lexicon.pkl               #   Serialised lexicon
│   ├── results_baseline_test.json#   Baseline per-tag metrics (test)
│   ├── results_final_test.json   #   Final per-tag metrics (test)
│   ├── ablation_table.csv        #   Ablation study table
│   ├── confusion_matrix.png      #   Confusion matrix heatmap
│   ├── error_analysis.csv        #   Sample error examples
│   └── *.png                     #   Various analysis plots
│
└── report/
    └── references.bib            # Bibliography
```

---

## Getting Started

### Prerequisites

- **Python 3.8+**
- pip (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/Rule-Based-POS-Tagging.git
   cd Rule-Based-POS-Tagging
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Data Setup

The tagger uses the **UD English Web Treebank v2.x** in CoNLL-U format. Place the three data files in the `data/` directory:

```
data/
├── en_ewt-ud-train.conllu
├── en_ewt-ud-dev.conllu
└── en_ewt-ud-test.conllu
```

You can download them from: https://universaldependencies.org/treebanks/en_ewt/

---

## Usage

All commands are run from the project root directory.

### Full Pipeline

Build the lexicon, evaluate on both dev and test splits (baseline + innovation), and run the ablation study — all in one command:

```bash
python main.py run-all
```

This produces:
- `outputs/lexicon.pkl` — serialised word-tag lexicon
- `outputs/results_baseline_test.json` — baseline per-tag metrics
- `outputs/results_final_test.json` — innovation per-tag metrics
- `outputs/confusion_matrix.png` — confusion matrix heatmap
- `outputs/error_analysis.csv` — sample misclassified tokens
- `outputs/ablation_table.csv` — ablation study results

### Tag a Sentence

Interactively tag a sentence using the full tagger (innovation enabled):

```bash
python main.py tag "The quick brown fox jumps over the lazy dog"
```

**Example output:**

```
  Word                 Tag
  ----------------------------
  The                  DET
  quick                ADJ
  brown                ADJ
  fox                  NOUN
  jumps                NOUN
  over                 ADP
  the                  DET
  lazy                 ADJ
  dog                  NOUN
```

### Evaluate on Dev/Test

Run evaluation on a specific split:

```bash
# Baseline on dev set
python main.py evaluate --split dev

# Baseline on test set
python main.py evaluate --split test

# With innovation modules enabled
python main.py evaluate --split test --innovation
```

### Ablation Study

Disable one rule module at a time to measure its contribution:

```bash
python main.py ablation
```

### Build Lexicon Only

If you only want to build and save the lexicon:

```bash
python main.py build-lexicon
```

---

## Results

### Per-Tag F1 Scores

Performance on the **UD EWT test set** (25,094 tokens):

| Tag | Precision | Recall | F1 Score | Support |
|-----|--------:|-------:|-------:|--------:|
| PUNCT | 0.9835 | 0.9803 | **0.9819** | 3,096 |
| CCONJ | 0.9902 | 0.9633 | **0.9766** | 736 |
| PRON | 0.9639 | 0.9251 | **0.9441** | 2,164 |
| DET | 0.9169 | 0.9594 | **0.9377** | 1,897 |
| AUX | 0.8183 | 0.9955 | **0.8982** | 1,543 |
| ADP | 0.8685 | 0.8840 | **0.8762** | 2,025 |
| ADJ | 0.9041 | 0.8484 | **0.8754** | 1,788 |
| ADV | 0.9476 | 0.7750 | **0.8527** | 1,191 |
| NOUN | 0.8196 | 0.8748 | **0.8463** | 4,123 |
| NUM | 0.8767 | 0.8137 | **0.8440** | 542 |
| VERB | 0.8723 | 0.8131 | **0.8416** | 2,605 |
| PART | 0.7210 | 0.8921 | **0.7975** | 649 |
| PROPN | 0.8804 | 0.7166 | **0.7901** | 2,075 |
| SCONJ | 0.6059 | 0.5885 | **0.5971** | 384 |
| INTJ | 0.4533 | 0.8430 | **0.5896** | 121 |
| SYM | 0.4476 | 0.5664 | **0.5000** | 113 |
| X | 0.0533 | 0.0952 | **0.0684** | 42 |

> **Overall Accuracy: 87.51%** (final, with innovation) vs **87.63%** (baseline)

### Ablation Study Results

Each row shows accuracy when a specific rule module is **disabled**:

| Rule Disabled | Accuracy | Drop from Full |
|---------------|-------:|-------:|
| None (full system) | **87.22%** | — |
| `context` | 87.20% | −0.02% |
| `morphology` | 86.87% | −0.35% |
| `capitalization` | 85.73% | −1.49% |
| `lexicon` | 56.56% | −30.66% |

**Key Insight:** The **lexicon** is by far the most critical component, contributing ~30% of overall accuracy. **Capitalization** is the second most impactful rule, adding ~1.5% by detecting proper nouns.

---

## Notebooks

The `notebooks/` directory contains Jupyter notebooks documenting the full analysis workflow:

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory data analysis — tag distributions, sentence lengths, word ambiguity |
| `02_baseline_eval.ipynb` | Baseline tagger evaluation with visualisations |
| `03_error_analysis.ipynb` | Detailed error analysis — top confusion pairs, error examples |
| `04_ablation.ipynb` | Ablation study — contribution of each rule module |
| `05_rule_coverage_and_oov_analysis.ipynb` | Rule coverage distribution and OOV word analysis |
| `06_innovation.ipynb` | Evaluation of innovation modules and comparison with baseline |

To run the notebooks:

```bash
jupyter notebook notebooks/
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

**Author:** Gujja Srivarshith Rao
