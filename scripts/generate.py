#!/usr/bin/env python3
"""
Generate protein sequences from a trained idpGPT model.

Takes a weights file (as saved by train.py: state_dict + architecture
bundled together), a target length range, and how many sequences to
generate.
"""

import argparse

import torch
import yaml

from idpgpt import decoderGPT, predict, saveFasta


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate protein sequences from a trained idpGPT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--weights", type=str, required=True,
                    help="Path to .pt file saved by train.py (state_dict + model_args + vocab_len)")
    p.add_argument("--tokenizer", type=str, required=True, dest="tokenizer_path", 
                   help="Path to tokenizer YAML file")

    p.add_argument("--min-length", type=int, required=True, help="Minimum generated sequence length")
    p.add_argument("--max-length", type=int, required=True, help="Maximum generated sequence length")
    p.add_argument("--n-seq", type=int, required=True, dest="n_sequences", help="Number of sequences to generate")

    p.add_argument("--output", type=str, default="generated.fasta", help="Output FASTA path")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Generation knobs (sensible defaults; rarely need changing)
    p.add_argument("--batch-size", type=int, default=64, help="Sequences generated per forward pass")
    p.add_argument("--context", type=int, default=None,
                    help="Attention context window during generation (defaults to model's max_len)")
    p.add_argument("--sampling", type=str, default="multinomial", choices=["multinomial", "greedy"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # Load tokenizer
    with open(args.tokenizer_path, "r") as f:
        tokenizer = yaml.safe_load(f)

    # Load weights + architecture
    ckpt = torch.load(args.weights, map_location="cpu")
    model = decoderGPT(vocab_len=ckpt["vocab_len"], **ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    context = args.context if args.context is not None else ckpt["model_args"]["max_len"]

    # Generate in batches until we have n_sequences
    sequences = []
    remaining = args.n_sequences
    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        batch = predict(
            models=[model],
            minLength=args.min_length,
            maxLength=args.max_length,
            tokenizer=tokenizer,
            context=context,
            batch_size=batch_size,
            sampling=args.sampling,
            T=args.temperature,
        )
        sequences.extend(batch)
        remaining -= batch_size

    saveFasta(sequences, args.output)


if __name__ == "__main__":
    main()
