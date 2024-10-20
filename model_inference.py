import sys
from os import path, makedirs
import argparse
from pdllib import ModelInference, check_sequence, load_seqfile

def get_options():
    parser = argparse.ArgumentParser(description="Script for Plant DNA Large Language Models (LLMs) inference")
    parser.add_argument('-m', dest='model', required=True, type=str, help='Model path')
    parser.add_argument('-ms', dest='source', type=str, choices=['huggingface', 'modelscope', 'local'], default='huggingface', help='Model source')
    parser.add_argument('-f', dest='file', default=None, type=str, help='File contains sequences')
    parser.add_argument('-s', dest='sequence', type=str, default=None, help='Single sequence')
    parser.add_argument('-t', dest='threshold', type=float, default=0.5, help='Threshold for classification')
    parser.add_argument('-l', dest='max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('-d', dest='device', type=str, choices=['cpu','gpu','mps','auto'], default='auto', help='Device for inference')
    parser.add_argument('-o', dest='outfile', type=str, default=None, help='Output file')
    return parser.parse_args()

def main():
    args = get_options()

    model = ModelInference(
        model_path=args.model,
        source=args.source,
        device=args.device,
        max_token=args.max_length
    )

    if args.file:
        results = model.predict_file(args.file, threshold=args.threshold)
        sequences = load_seqfile(args.file)
    else:
        results = model.predict([args.sequence], threshold=args.threshold)
        sequences = [args.sequence]

    if args.outfile:
        with open(args.outfile, 'w') as f:
            print("Sequence", "Length", "Label", "Probability", sep="\t", file=f)
            for seq, result in zip(sequences, results):
                print(seq, len(seq), result['label'], result['probability'], sep="\t", file=f)
    else:
        for seq, result in zip(sequences, results):
            print(f"Sequence: {seq}")
            print(f"Length: {len(seq)}")
            print(f"Label: {result['label']}")
            print(f"Probability: {result['probability']}")
            print()

if __name__ == '__main__':
    main()
