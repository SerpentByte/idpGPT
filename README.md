# idpGPT

Compact, decoder-only transformer language models trained to generate protein
sequences prone to liquid-liquid phase separation (LLPS). Two models are
provided: **LLPS+ GPT**, trained on phase-separating IDPs, and **PDB\* GPT**,
trained on well-folded, non-condensing proteins.

## Example

Generate 10 sequences (20–40 residues) from the LLPS+ model:

```bash
python scripts/generate.py \
  --weights models/llps_plus_weights.pt \
  --tokenizer-path models/tokenizer.yaml \
  --min-length 20 \
  --max-length 40 \
  --n-sequences 10 \
  --output generated.fa
```

This loads the trained weights, autoregressively samples sequences in the
given length range, and writes them to `generated.fa`. Swap in
`models/pdb_star_weights.pt` to generate from the PDB\* model instead.

To train a model from scratch:

```bash
python scripts/train.py \
  --tokenizer-path models/tokenizer.yaml \
  --train-data data/llps_plus_train.fa \
  --val-data data/llps_plus_val.fa \
  --nheads 8 --nlayers 8 --embed-dim 256 --ff-dim 1024 --max-len 128 \
  --best-weights-out models/llps_plus_weights.pt
```

## Repository structure

```
.
├── data
│   ├── File-S1.fa ... File-S5.fa    # supporting sequence sets (S1–S5)
│   ├── llps_plus_train.fa           # LLPS+ training sequences
│   └── pdb_star_train.fa            # PDB* training sequences
├── models
│   ├── llps_plus_weights.pt         # trained LLPS+ GPT weights + architecture
│   ├── pdb_star_weights.pt          # trained PDB* GPT weights + architecture
│   └── tokenizer.yaml               # shared amino-acid tokenizer
├── scripts
│   ├── idpgpt.py                    # library: model, dataset, prediction utilities
│   ├── train.py                     # training entry point
│   └── generate.py                  # generation entry point
├── requirements.txt
└── LICENSE
```

- **`data/`** — FASTA files used for training and the supplementary generated
  sequence sets (File-S1–S5) referenced in the manuscript.
- **`models/`** — checkpoints ready to use with `generate.py`. Each `.pt` file
  bundles the model's weights together with the architecture it was trained
  with, so no extra configuration is needed at generation time. Both models
  share the same tokenizer.
- **`scripts/idpgpt.py`** — the core library (transformer decoder, Lightning
  training wrapper, autoregressive sampling, FASTA I/O). Imported by
  `train.py` and `generate.py`; not run directly.
- **`scripts/train.py`** — trains a model and saves the best checkpoint (by
  validation loss) as a `.pt` file for use with `generate.py`.
- **`scripts/generate.py`** — loads a trained `.pt` file and generates new
  sequences within a specified length range.

An easier to use Google Colab is available [here](https://tinyurl.com/3kmy5xbt)

## Citation

If you use this code or the provided models, please cite:

```bibtex
@article{wasim2024harnessing,
  title={Harnessing Transformers to Generate Protein Sequences Prone to Liquid Liquid Phase Separation},
  author={Wasim, Abdul and Pramanik, Ushasi and Das, Anirban and Latua, Pikaso and Rudra, Jai S and Mondal, Jagannath},
  journal={bioRxiv},
  pages={2024--03},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
