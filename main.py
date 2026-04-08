#!/usr/bin/env python3
"""
main.py — CLI entry point for the Rule-Based POS Tagger.

Usage:
    python main.py run-all          # full pipeline: build lexicon → tag → evaluate
    python main.py build-lexicon    # build and save lexicon only
    python main.py evaluate         # evaluate on dev/test (lexicon must exist)
    python main.py ablation         # run ablation study
    python main.py tag "Some text"  # tag a single sentence interactively
"""

import argparse
import json
import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.parser import parse_conllu, get_forms_and_tags
from src.lexicon import Lexicon
from src.tagger import RuleBasedTagger
from src.evaluate import Evaluator

# Paths 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LEXICON_PATH = os.path.join(OUTPUT_DIR, "lexicon.pkl")

TRAIN_FILE = os.path.join(DATA_DIR, "en_ewt-ud-train.conllu")
DEV_FILE = os.path.join(DATA_DIR, "en_ewt-ud-dev.conllu")
TEST_FILE = os.path.join(DATA_DIR, "en_ewt-ud-test.conllu")


def build_lexicon():
    """Parse training data and build the word-tag lexicon."""
    print("=" * 60)
    print("BUILDING LEXICON")
    print("=" * 60)

    print(f"  Parsing {TRAIN_FILE} ...")
    train_sents = parse_conllu(TRAIN_FILE)
    print(f"  Loaded {len(train_sents):,} training sentences")
    print(f"  Total tokens: {sum(len(s) for s in train_sents):,}")

    lexicon = Lexicon().build(train_sents)
    print(f"  {lexicon}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lexicon.save(LEXICON_PATH)
    print(f"  Saved lexicon → {LEXICON_PATH}")

    return lexicon, train_sents


def evaluate(split="dev", use_innovation=False, label=None):
    """Evaluate the tagger on dev or test split."""
    # Load lexicon
    if not os.path.exists(LEXICON_PATH):
        print("[!] Lexicon not found. Building first ...")
        lexicon, _ = build_lexicon()
    else:
        lexicon = Lexicon.load(LEXICON_PATH)

    # Pick split
    split_file = DEV_FILE if split == "dev" else TEST_FILE
    if label is None:
        label = "baseline" if not use_innovation else "final"

    print()
    print("=" * 60)
    print(f"EVALUATING on {split.upper()} split ({label})")
    print("=" * 60)

    print(f"  Parsing {split_file} ...")
    sents = parse_conllu(split_file)
    corpus = get_forms_and_tags(sents)
    print(f"  Loaded {len(corpus):,} sentences, "
          f"{sum(len(w) for w, _ in corpus):,} tokens")

    # Lexicon coverage
    known, total, pct = lexicon.coverage(sents)
    print(f"  Lexicon coverage: {known:,}/{total:,} ({pct:.1f}%)")

    # Tag
    tagger = RuleBasedTagger(lexicon, use_innovation=use_innovation)
    start = time.time()
    tagged = tagger.tag_corpus(corpus)
    elapsed = time.time() - start

    # Evaluate
    evaluator = Evaluator()
    results = evaluator.run_full_evaluation(tagged, label=label, output_dir=OUTPUT_DIR)

    print(f"\n  Overall accuracy : {results['overall_accuracy']:.4f}")
    print(f"  Tagging speed    : {sum(len(w) for w, _ in corpus) / elapsed:,.0f} tokens/sec")
    print(f"  Time elapsed     : {elapsed:.2f}s")
    print()

    # Per-tag table
    print(f"  {'Tag':<8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-'*40}")
    for tag, m in sorted(results["per_tag"].items()):
        print(f"  {tag:<8} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['support']:>8,}")

    print(f"\n  Results → {OUTPUT_DIR}/results_{label}.json")
    print(f"  Errors  → {OUTPUT_DIR}/error_analysis.csv")
    print(f"  Matrix  → {OUTPUT_DIR}/confusion_matrix.png")
    print()

    return results


def ablation():
    """Run ablation study: disable rules one at a time."""
    if not os.path.exists(LEXICON_PATH):
        print("[!] Lexicon not found. Building first ...")
        build_lexicon()

    lexicon = Lexicon.load(LEXICON_PATH)

    print()
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    sents = parse_conllu(DEV_FILE)
    corpus = get_forms_and_tags(sents)
    tagger = RuleBasedTagger(lexicon)

    rules_to_disable = [
        "none", "punct_num", "closed_class", "lexicon",
        "capitalization", "morphology", "context",
    ]

    ablation_results = {}
    for rule in rules_to_disable:
        print(f"  Testing with '{rule}' disabled ...", end=" ", flush=True)
        if rule == "none":
            tagged = tagger.tag_corpus(corpus)
        else:
            tagged = []
            for words, gold in corpus:
                pred = tagger.tag_sentence_ablation(words, disable=rule)
                tagged.append((words, gold, pred))

        acc = Evaluator.accuracy(tagged)
        ablation_results[rule] = acc
        print(f"accuracy = {acc:.4f}")

    Evaluator.export_ablation_table(ablation_results, OUTPUT_DIR)
    print(f"\n  Ablation table → {OUTPUT_DIR}/ablation_table.csv")

    # Show summary
    full_acc = ablation_results["none"]
    print(f"\n  {'Rule':<20} {'Accuracy':>10} {'Drop':>10}")
    print(f"  {'-'*40}")
    for rule in rules_to_disable:
        acc = ablation_results[rule]
        drop = full_acc - acc
        marker = " ← FULL" if rule == "none" else ""
        print(f"  {rule:<20} {acc:>10.4f} {drop:>10.4f}{marker}")
    print()


def tag_interactive(text):
    """Tag a single sentence."""
    if not os.path.exists(LEXICON_PATH):
        print("[!] Lexicon not found. Building first ...")
        build_lexicon()

    lexicon = Lexicon.load(LEXICON_PATH)
    tagger = RuleBasedTagger(lexicon, use_innovation=True)

    words = text.split()
    tags = tagger.tag_sentence(words)

    print(f"\n  {'Word':<20} {'Tag':<8}")
    print(f"  {'-'*28}")
    for w, t in zip(words, tags):
        print(f"  {w:<20} {t:<8}")
    print()


def run_all():
    """Full pipeline: build lexicon → baseline eval → innovation eval → ablation."""
    build_lexicon()

    # Baseline (without innovation)
    evaluate(split="dev", use_innovation=False, label="baseline")
    evaluate(split="test", use_innovation=False, label="baseline_test")

    # With innovation
    evaluate(split="dev", use_innovation=True, label="final")
    evaluate(split="test", use_innovation=True, label="final_test")

    # Ablation
    ablation()

    print("=" * 60)
    print("ALL DONE — check the outputs/ directory for results.")
    print("=" * 60)


# CLI 
def main():
    parser = argparse.ArgumentParser(
        description="Rule-Based POS Tagger for UD English Web Treebank"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("build-lexicon", help="Build and save word-tag lexicon")
    sub.add_parser("run-all", help="Full pipeline (build + eval + ablation)")

    eval_p = sub.add_parser("evaluate", help="Evaluate on dev or test")
    eval_p.add_argument("--split", default="dev", choices=["dev", "test"])
    eval_p.add_argument("--innovation", action="store_true")

    sub.add_parser("ablation", help="Ablation study")

    tag_p = sub.add_parser("tag", help="Tag a sentence")
    tag_p.add_argument("text", help="Sentence to tag")

    args = parser.parse_args()

    if args.command == "build-lexicon":
        build_lexicon()
    elif args.command == "run-all":
        run_all()
    elif args.command == "evaluate":
        evaluate(split=args.split, use_innovation=args.innovation)
    elif args.command == "ablation":
        ablation()
    elif args.command == "tag":
        tag_interactive(args.text)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
