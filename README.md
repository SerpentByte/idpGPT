# idpGPT

Compact, decoder-only transformer language models trained to generate protein
sequences prone to liquid-liquid phase separation (LLPS). Two models are
provided: **LLPS+ GPT**, trained on phase-separating IDPs, and **PDB\* GPT**,
trained on well-folded, non-condensing proteins.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

`scripts/idpgpt.py` is the shared library imported by both `train.py` and
`generate.py`. Since it lives inside `scripts/` rather than being an
installed package, that directory needs to be on your `PYTHONPATH` before
running either script.

**Linux / macOS:**

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/scripts"
```

**Windows (PowerShell):**

```powershell
$env:PYTHONPATH += ";$(Get-Location)\scripts"
```

**Windows (Command Prompt):**

```cmd
set PYTHONPATH=%PYTHONPATH%;%cd%\scripts
```

Run these from the repository root. This only needs to be set once per
terminal session (re-run it if you open a new terminal).

## Example

Generate 10 sequences (20–40 residues) from the LLPS+ model:

```bash
python scripts/generate.py \
  --weights models/llps_plus_weights.pt \
  --tokenizer models/tokenizer.yaml \
  --min-length 20 \
  --max-length 40 \
  --n-seq 10 \
  --output generated.fa
```

This loads the trained weights, autoregressively samples sequences in the
given length range, and writes them to `generated.fa`. Swap in
`models/pdb_star_weights.pt` to generate from the PDB\* model instead.

To train a model from scratch:

```bash
python scripts/train.py \
  --tokenizer models/tokenizer.yaml \
  --train-data data/llps_plus_train0.lod \
  --val-data data/llps_plus_val0.lod \
  --nheads 2 --nlayers 2 --embed-dim 1024 --ff-dim 2048 --max-len 128 \
  --best-weights-out models/llps_plus_weights.pt
```

`--train-data`/`--val-data` each take a `.lod` file — a pickled
`RandomChunkLoader` dataset (already tokenized and chunked), not a raw FASTA
file. `data/llps_plus_full.lod` and `data/noPS_full.lod` are the full,
unsplit datasets (i.e. before the train/val split) for LLPS+ and the
non-phase-separating set respectively, provided for reference/reproducibility.

If you have your own train/val FASTA files instead of `.lod` files, convert
them first:

```bash
python scripts/fasta2loader.py \
  --fasta my_train.fa \
  --tokenizer models/tokenizer.yaml \
  --chunk-length 129 \
  --output my_train.lod
```

Run once for each of your train and val FASTA files. `--chunk-length` should
be one more than the `--max-len` you plan to train with (`RandomChunkLoader`
splits each chunk into an input/target pair shifted by one position).

## Repository structure

```
.
├── data
│   ├── File-S1.fa ... File-S5.fa    # supporting sequence sets (S1–S5)
│   ├── llps_plus_full.fa            # full LLPS+ dataset in FASTA (readable)
│   ├── llps_plus_full.lod           # full LLPS+ dataset (pickled RandomChunkLoader), unsplit
│   ├── llps_plus_train0.lod         # LLPS+ training split
│   ├── llps_plus_val0.lod           # LLPS+ validation split
│   ├── noPS_full.fa                 # full non-phase-separating dataset in FASTA (readable)
│   └── noPS_full.lod                # full non-phase-separating dataset (pickled RandomChunkLoader)
├── models
│   ├── llps_plus_weights.pt         # trained LLPS+ GPT weights + architecture
│   ├── pdb_star_weights.pt          # trained PDB* GPT weights + architecture
│   └── tokenizer.yaml               # shared amino-acid tokenizer
├── scripts
│   ├── idpgpt.py                    # library: model, dataset, prediction utilities
│   ├── fasta2loader.py              # converts FASTA -> .lod (pickled RandomChunkLoader)
│   ├── train.py                     # training entry point
│   └── generate.py                  # generation entry point
├── requirements.txt
└── LICENSE
```

- **`data/`** — `.lod` files are pickled `RandomChunkLoader` datasets (already
  tokenized and chunked) used directly by `train.py`; `llps_plus_full.fa` and
  `noPS_full.fa` are the same full datasets in plain FASTA form (their `.lod`
  counterparts are what `fasta2loader.py` produces from them); `File-S1.fa`–
  `File-S5.fa` are the supplementary generated sequence sets referenced in
  the manuscript.
- **`models/`** — checkpoints ready to use with `generate.py`. Each `.pt` file
  bundles the model's weights together with the architecture it was trained
  with, so no extra configuration is needed at generation time. Both models
  share the same tokenizer.
- **`scripts/idpgpt.py`** — the core library (transformer decoder, Lightning
  training wrapper, autoregressive sampling, FASTA I/O). Imported by
  `train.py` and `generate.py`; not run directly.
- **`scripts/fasta2loader.py`** — tokenizes a FASTA file and packages it as a
  `.lod` file for use with `train.py`.
- **`scripts/train.py`** — trains a model and saves the best checkpoint (by
  validation loss) as a `.pt` file for use with `generate.py`.
- **`scripts/generate.py`** — loads a trained `.pt` file and generates new
  sequences within a specified length range.

An easier to use Google Colab is available [here](https://tinyurl.com/3kmy5xbt).

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
