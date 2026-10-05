"""CPU prediction using a trusted local canonical 5-mer model and sequence CSV."""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from features import canonical_5mers

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path,
        default=HERE / "artifacts" / "canonical_5mer_model.joblib",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.output.exists():
        parser.error("Output already exists; choose a new output path")
    # joblib uses pickle: load only this repository's or your own trusted model.
    model = joblib.load(args.model)
    if model["k"] != 5:
        raise ValueError("This predictor requires a canonical 5-mer model")
    vectorizer = CountVectorizer(
        analyzer=canonical_5mers, vocabulary=model["canonical_vocabulary"],
        dtype=np.float32,
    )
    count = 0
    with args.input.open(newline="") as source:
        reader = csv.DictReader(source)
        if "sequence" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must have a sequence column")
        with args.output.open("x", newline="") as destination:
            writer = csv.writer(destination, lineterminator="\n")
            writer.writerow(["row", "H3K27ac_probability", "prediction"])

            def emit(sequences):
                nonlocal count
                features = model["tfidf"].transform(vectorizer.transform(sequences))
                probabilities = model["classifier"].predict_proba(features)[:, 1]
                for probability in probabilities:
                    writer.writerow([
                        count, float(probability),
                        int(probability >= model["threshold"]),
                    ])
                    count += 1

            batch = []
            for row in reader:
                sequence = (row["sequence"] or "").strip().upper()
                if len(sequence) < 5 or set(sequence) - set("ACGTN"):
                    raise ValueError(f"Invalid DNA sequence at input row {count + len(batch)}")
                batch.append(sequence)
                if len(batch) == 256:
                    emit(batch)
                    batch = []
            if batch:
                emit(batch)
    print(f"Predicted {count} sequences on CPU; threshold={model['threshold']}")


if __name__ == "__main__":
    main()
