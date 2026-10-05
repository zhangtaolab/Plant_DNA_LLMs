"""Benchmark classical CPU baselines using existing H3K27ac CSV splits.

Strand symmetry is motivated by Caduceus (arXiv:2403.03234). This code
does not implement Caduceus or compare against a pretrained foundation model.
"""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    VOCABULARY,
    canonical_5mers,
    composition_features,
    reverse_complement,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def metrics(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "threshold": float(threshold),
    }


def choose_threshold(labels, probabilities):
    return float(max(
        np.linspace(0.1, 0.9, 81),
        key=lambda threshold: matthews_corrcoef(labels, probabilities >= threshold),
    ))


def read_splits(data_dir):
    audit, splits, hashes, seen = {}, {}, {}, set()
    # Earlier splits have priority, preventing exact/RC train-evaluation overlap.
    for name in ["train", "dev", "test"]:
        path = data_dir / f"{name}.csv"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows, removed, invalid = [], 0, 0
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"sequence", "label"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{path} requires sequence and label columns")
            for row in reader:
                sequence = (row["sequence"] or "").strip().upper()
                if (len(sequence) < 5 or set(sequence) - set("ACGTN")
                        or row["label"] not in ("0", "1")):
                    invalid += 1
                    continue
                key = min(sequence, reverse_complement(sequence))
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
                rows.append((sequence, int(row["label"])))
        if not rows:
            raise ValueError(f"{name} has no usable sequences")
        sequences, labels = zip(*rows)
        labels = np.array(labels)
        if len(np.unique(labels)) != 2:
            raise ValueError(f"{name} must contain both classes for this benchmark")
        splits[name] = (list(sequences), labels)
        audit[name] = {
            "kept": len(labels),
            "positive_fraction": float(labels.mean()),
            "exact_or_RC_duplicates_removed": removed,
            "invalid_removed": invalid,
            "min_length": min(map(len, sequences)),
            "max_length": max(map(len, sequences)),
        }
        print(name, audit[name], flush=True)
    return splits, audit, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    start = time.monotonic()
    splits, audit, hashes = read_splits(args.data_dir)
    train, dev, test = (splits[name] for name in ["train", "dev", "test"])

    baseline = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=500, random_state=args.seed)
    )
    baseline.fit(composition_features(train[0]), train[1])
    dev_base = baseline.predict_proba(composition_features(dev[0]))[:, 1]
    threshold_base = choose_threshold(dev[1], dev_base)

    # Exhaustive fixed vocabulary; TF-IDF statistics use training data only.
    vectorizer = CountVectorizer(
        analyzer=canonical_5mers, vocabulary=VOCABULARY, dtype=np.float32
    )
    tfidf = TfidfTransformer(sublinear_tf=True)
    train_features = tfidf.fit_transform(vectorizer.fit_transform(train[0]))
    dev_features = tfidf.transform(vectorizer.transform(dev[0]))
    best, candidates = None, []
    for regularization in [0.1, 1.0, 10.0]:
        classifier = LogisticRegression(
            C=regularization, max_iter=500, solver="liblinear", random_state=args.seed
        )
        classifier.fit(train_features, train[1])
        probabilities = classifier.predict_proba(dev_features)[:, 1]
        score = float(roc_auc_score(dev[1], probabilities))
        candidates.append({
            "C": regularization, "dev_auroc": score,
            "iterations": classifier.n_iter_.tolist(),
        })
        print("candidate", candidates[-1], flush=True)
        if best is None or score > best[0]:
            best = (score, classifier, probabilities, regularization)
    threshold = choose_threshold(dev[1], best[2])

    # Evaluate test only after model/threshold selection is complete.
    test_base = baseline.predict_proba(composition_features(test[0]))[:, 1]
    test_features = tfidf.transform(vectorizer.transform(test[0]))
    test_probabilities = best[1].predict_proba(test_features)[:, 1]
    forward = vectorizer.transform(test[0][:10]).toarray()
    reverse = vectorizer.transform([
        reverse_complement(sequence) for sequence in test[0][:10]
    ]).toarray()
    if not np.array_equal(forward, reverse):
        raise RuntimeError("Reverse-complement feature invariance failed")
    report = {
        "task": "H3K27ac binary classification; not expression prediction",
        "seed": args.seed,
        "package_versions": {
            name: version(name) for name in ["numpy", "scipy", "scikit-learn", "joblib"]
        },
        "data_sha256": hashes,
        "audit": audit,
        "model_selection": candidates,
        "selected_C": best[3],
        "dev": {
            "composition_baseline": metrics(dev[1], dev_base, threshold_base),
            "canonical_5mer_tfidf": metrics(dev[1], best[2], threshold),
        },
        "test": {
            "composition_baseline": metrics(test[1], test_base, threshold_base),
            "canonical_5mer_tfidf": metrics(test[1], test_probabilities, threshold),
        },
        "limits": [
            "Exact and reverse-complement duplicates removed, but homologous/overlapping fragments may remain.",
            "Source CSV files lack chromosome, species, tissue, and coordinate metadata: chromosome/species split cannot be established.",
            "This result is not a comparison against a pretrained PDLLM, DNABERT-2, or Caduceus.",
            "H3K27ac labels cannot serve as gene expression or accession/trait labels.",
        ],
        "seconds": time.monotonic() - start,
    }
    report["test_auroc_gain"] = (
        report["test"]["canonical_5mer_tfidf"]["auroc"]
        - report["test"]["composition_baseline"]["auroc"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    joblib.dump({
        "tfidf": tfidf, "classifier": best[1], "threshold": threshold,
        "canonical_vocabulary": VOCABULARY, "k": 5,
    }, args.output_dir / "canonical_5mer_model.joblib")
    with (args.output_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "retained_test_row", "label", "composition_probability",
            "canonical_5mer_probability",
        ])
        writer.writerows(zip(
            range(len(test[1])), test[1], test_base, test_probabilities
        ))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
