"""Fixed DNA features shared by training and inference."""
import itertools

import numpy as np

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


CANONICAL_KMERS = {
    kmer: min(kmer, reverse_complement(kmer))
    for bases in itertools.product("ACGT", repeat=5)
    for kmer in ["".join(bases)]
}
VOCABULARY = {
    kmer: index
    for index, kmer in enumerate(sorted(set(CANONICAL_KMERS.values())))
}


def canonical_5mers(sequence):
    # Ambiguous windows containing N are omitted; strand orientation is ignored.
    return [
        CANONICAL_KMERS[window]
        for index in range(len(sequence) - 4)
        for window in [sequence[index:index + 5]]
        if window in CANONICAL_KMERS
    ]


def composition_features(sequences):
    return np.array([
        [
            (sequence.count("G") + sequence.count("C")) / len(sequence),
            sequence.count("N") / len(sequence),
            np.log1p(len(sequence)),
        ]
        for sequence in sequences
    ])
