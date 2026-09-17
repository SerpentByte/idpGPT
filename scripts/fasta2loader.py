#!/usr/bin/env python3
"""
Convert a FASTA file into a .lod file: a pickled RandomChunkLoader dataset
ready to pass to train.py's --train-data / --val-data.

Sequences are joined with a single space separator (consistent with the
tokenizer's vocabulary, which includes a space token as a sequence
boundary), tokenized character-by-character, and wrapped in a
RandomChunkLoader over the resulting linear token array.
"""

import argparse
import pickle

import numpy as np
import yaml

from idpgpt import readFasta, RandomChunkLoader


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert a FASTA file to a .lod (pickled RandomChunkLoader) dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--fasta", type=str, required=True, help="Path to input FASTA file")
    p.add_argument("--tokenizer", type=str, required=True, dest="tokenizer_path", 
                   help="Path to tokenizer YAML file")
    p.add_argument("--chunk-length", type=int, required=True,
                    help="Chunk length for RandomChunkLoader (max-len + 1)")
    p.add_argument("--output", type=str, required=True, help="Output .lod path")

    return p.parse_args()


def main():
    args = parse_args()

    with open(args.tokenizer_path, "r") as f:
        tokenizer = yaml.safe_load(f)

    sequences = list(readFasta(args.fasta).values())
    joined = " ".join(sequences)

    missing = sorted(set(joined) - set(tokenizer.keys()))
    if missing:
        raise ValueError(
            f"Characters in FASTA not found in tokenizer: {missing}. "
            "Check that --tokenizer-path matches this dataset."
        )

    tokens = np.array([tokenizer[c] for c in joined], dtype=np.int64)
    print(f"Tokenized {len(sequences)} sequences into {len(tokens)} tokens")

    dataset = RandomChunkLoader(tokens, length=args.chunk_length)

    with open(args.output, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved dataset ({len(dataset)} chunks) to {args.output}")


if __name__ == "__main__":
    main()
