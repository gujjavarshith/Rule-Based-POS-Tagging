"""
Evaluation module — accuracy, per-tag precision/recall/F1, confusion matrix.
"""

import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


# All 17 UPOS tags
UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
]


class Evaluator:
    """Compute POS tagging metrics."""

    def __init__(self):
        self.results: Dict = {}

    # Core metrics
    @staticmethod
    def accuracy(
        tagged_corpus: List[Tuple[List[str], List[str], List[str]]]
    ) -> float:
        """Overall token-level accuracy."""
        correct = 0
        total = 0
        for _words, gold, pred in tagged_corpus:
            for g, p in zip(gold, pred):
                total += 1
                if g == p:
                    correct += 1
        return correct / total if total else 0.0

    @staticmethod
    def per_tag_metrics(
        tagged_corpus: List[Tuple[List[str], List[str], List[str]]],
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Precision, recall, F1 for each tag.

        Returns
        -------
        dict : {tag: {"precision": …, "recall": …, "f1": …, "support": …}}
        """
        if tags is None:
            tags = UPOS_TAGS

        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        support = defaultdict(int)

        for _words, gold, pred in tagged_corpus:
            for g, p in zip(gold, pred):
                support[g] += 1
                if g == p:
                    tp[g] += 1
                else:
                    fp[p] += 1
                    fn[g] += 1

        metrics = {}
        for tag in tags:
            p = tp[tag] / (tp[tag] + fp[tag]) if (tp[tag] + fp[tag]) else 0.0
            r = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            metrics[tag] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "support": support[tag],
            }

        return metrics

    @staticmethod
    def confusion_matrix(
        tagged_corpus: List[Tuple[List[str], List[str], List[str]]],
        tags: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Build a confusion matrix (rows = gold, cols = predicted).

        Returns
        -------
        np.ndarray of shape (n_tags, n_tags)
        """
        if tags is None:
            tags = UPOS_TAGS
        tag2idx = {t: i for i, t in enumerate(tags)}
        n = len(tags)
        cm = np.zeros((n, n), dtype=int)

        for _words, gold, pred in tagged_corpus:
            for g, p in zip(gold, pred):
                gi = tag2idx.get(g)
                pi = tag2idx.get(p)
                if gi is not None and pi is not None:
                    cm[gi, pi] += 1

        return cm

    # Error analysis helpers
    @staticmethod
    def error_examples(
        tagged_corpus: List[Tuple[List[str], List[str], List[str]]],
        max_per_pair: int = 10,
    ) -> List[Dict]:
        """
        Collect example errors grouped by (gold, pred) tag pair.

        Returns list of dicts with keys: word, gold, pred, context.
        """
        errors = defaultdict(list)
        for words, gold, pred in tagged_corpus:
            for i, (g, p) in enumerate(zip(gold, pred)):
                if g != p:
                    key = (g, p)
                    if len(errors[key]) < max_per_pair:
                        ctx_start = max(0, i - 2)
                        ctx_end = min(len(words), i + 3)
                        errors[key].append({
                            "word": words[i],
                            "gold": g,
                            "pred": p,
                            "context": " ".join(words[ctx_start:ctx_end]),
                        })

        flat = []
        for _key, examples in errors.items():
            flat.extend(examples)
        return flat

    # Export
    def run_full_evaluation(
        self,
        tagged_corpus: List[Tuple[List[str], List[str], List[str]]],
        label: str = "baseline",
        output_dir: str = "outputs",
    ) -> Dict:
        """
        Run all metrics and export to files.

        Produces:
        - results_{label}.json  – accuracy + per-tag metrics
        - error_analysis.csv    – sample errors
        - confusion_matrix.png  – heatmap
        """
        os.makedirs(output_dir, exist_ok=True)

        acc = self.accuracy(tagged_corpus)
        per_tag = self.per_tag_metrics(tagged_corpus)
        cm = self.confusion_matrix(tagged_corpus)

        results = {
            "label": label,
            "overall_accuracy": round(acc, 4),
            "per_tag": per_tag,
        }

        # Save JSON
        json_path = os.path.join(output_dir, f"results_{label}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save error analysis CSV
        errors = self.error_examples(tagged_corpus)
        csv_path = os.path.join(output_dir, "error_analysis.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["word", "gold", "pred", "context"])
            writer.writeheader()
            writer.writerows(errors)

        # Save confusion matrix plot
        self._plot_confusion_matrix(cm, UPOS_TAGS, output_dir)

        self.results = results
        return results

    @staticmethod
    def _plot_confusion_matrix(
        cm: np.ndarray, labels: List[str], output_dir: str
    ) -> None:
        """Save a confusion matrix heatmap."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(14, 11))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                xticklabels=labels,
                yticklabels=labels,
                cmap="Blues",
                ax=ax,
                linewidths=0.5,
            )
            ax.set_xlabel("Predicted", fontsize=13)
            ax.set_ylabel("Gold", fontsize=13)
            ax.set_title("POS Tag Confusion Matrix", fontsize=15)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "confusion_matrix.png"), dpi=150
            )
            plt.close()
        except ImportError:
            print("[WARN] matplotlib/seaborn not installed; skipping plot.")

    # Ablation
    @staticmethod
    def export_ablation_table(
        ablation_results: Dict[str, float], output_dir: str = "outputs"
    ) -> None:
        """
        Write ablation results to CSV.

        Parameters
        ----------
        ablation_results : dict
            {rule_name: accuracy_when_disabled, …}
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "ablation_table.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["disabled_rule", "accuracy", "accuracy_drop"])

            full_acc = ablation_results.get("none", 0.0)
            for rule, acc in sorted(
                ablation_results.items(), key=lambda x: x[1]
            ):
                writer.writerow([rule, f"{acc:.4f}", f"{full_acc - acc:.4f}"])
