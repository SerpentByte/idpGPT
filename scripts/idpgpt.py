"""
idpGPT library
===============
Single-file library combining the model architecture, data loading,
prediction/generation utilities, and I/O helpers needed to train and
sample from the idpGPT transformer decoder.

Sections:
    1. Helpers      - seeding, FASTA I/O
    2. Loaders      - PyTorch Dataset for chunked sequence training
    3. Model         - decoderGPT (transformer) + LightningGPT wrapper
    4. Prediction   - checkpoint loading + autoregressive generation

Used by: trainGPT.py, generate.py
"""

import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from natsort import natsorted
from tqdm.auto import trange
from Bio import SeqIO


# ============================================================
# 1. Helpers
# ============================================================

def setSeeds(seed, deterministic=False):
    """Seed random, numpy, and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def readFasta(file):
    """Read a FASTA file into a dict of {record_id: sequence}."""
    data = {}
    with open(file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            data[record.id] = str(record.seq)
    return data


def saveFasta(sequences, output_path):
    """Save a list of sequences to a FASTA file."""
    with open(output_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">prot{i}\n{seq}\n")
    print(f"Saved {len(sequences)} sequences to {output_path}")


# ============================================================
# 2. Loaders
# ============================================================

class RandomChunkLoader(torch.utils.data.Dataset):
    """
    Dataset that slices fixed-length overlapping chunks from a single
    long tokenized sequence array `X`, returning (input, target) pairs
    shifted by one position for next-token prediction.
    """

    def __init__(self, X, length):
        self.X = X
        self.length = length
        self.data_length = X.shape[0]

    def __len__(self):
        return self.data_length - self.length

    def __getitem__(self, idx):
        chunk = self.X[idx:idx + self.length]
        return (
            torch.tensor(chunk[:-1], dtype=torch.long),
            torch.tensor(chunk[1:], dtype=torch.long),
        )


# ============================================================
# 3. Model: decoderGPT + LightningGPT
# ============================================================

class MHA(nn.Module):
    """Multi-Head Attention with optional causal masking."""

    def __init__(self, nheads, embed_dim, dropout, is_causal, batch_first):
        super().__init__()
        self.nheads = nheads
        self.embed_dim = embed_dim
        self.is_causal = is_causal
        self.batch_first = batch_first
        self.dropout = dropout

        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.nheads,
            dropout=self.dropout,
            batch_first=self.batch_first,
        )

    @staticmethod
    def generate_causal_mask(shape, device=None):
        Lq, Lk = shape
        mask = torch.triu(torch.ones(Lq, Lk, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, Q, K, V, padding_mask=None):
        if self.is_causal:
            lenQ = Q.size(1) if self.batch_first else Q.size(0)
            lenK = K.size(1) if self.batch_first else K.size(0)
            attn_mask = self.generate_causal_mask((lenQ, lenK), device=Q.device)
        else:
            attn_mask = None

        attn_out, attn_weights = self.mha(
            query=Q, key=K, value=V,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
        )
        return attn_out, attn_weights


class EncoderBlock(nn.Module):
    """Pre-norm Transformer block: self-attention + feed-forward, each with a residual."""

    def __init__(self, nheads, embed_dim, ff_dim, dropout, batch_first, is_causal, attn_dropout):
        super().__init__()
        self.nheads = nheads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.batch_first = batch_first
        self.is_causal = is_causal

        self.mha = MHA(
            nheads=self.nheads,
            embed_dim=self.embed_dim,
            is_causal=self.is_causal,
            dropout=self.attn_dropout,
            batch_first=self.batch_first,
        )

        self.ff = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, self.embed_dim),
        )

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = nn.Dropout(self.dropout)

    def forward(self, X, padding=None):
        h = self.norm1(X)
        attn_out, weights = self.mha(h, h, h, padding_mask=padding)
        X = X + self.drop1(attn_out)

        h = self.norm2(X)
        X = X + self.drop2(self.ff(h))

        return X, weights


class decoderGPT(nn.Module):
    """
    GPT-style causal transformer decoder for protein sequence modeling.

    Args:
        vocab_len (int): Vocabulary size.
        nheads (int): Attention heads per block.
        nlayers (int): Number of stacked EncoderBlocks.
        embed_dim (int): Embedding / model dimension.
        ff_dim (int): Feed-forward hidden dimension.
        max_len (int): Maximum sequence length (positional embedding table size).
        dropout (float): Residual dropout.
        attn_dropout (float): Attention-weight dropout.
        emb_dropout (float): Dropout after embedding + layernorm.
        batch_first (bool): Whether tensors are (batch, seq, dim).
        is_causal (bool): Whether to apply causal masking (True for autoregressive LM).
    """

    def __init__(self, vocab_len, nheads, nlayers, embed_dim, ff_dim, max_len,
                 dropout=0.1, attn_dropout=0.1, emb_dropout=0.1,
                 batch_first=True, is_causal=True):
        super().__init__()

        self.nheads = nheads
        self.nlayers = nlayers
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.batch_first = batch_first
        self.is_causal = is_causal
        self.attn_dropout = attn_dropout
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.emb_dropout = emb_dropout

        self.decoder = nn.ModuleList(
            [
                EncoderBlock(
                    nheads=self.nheads,
                    embed_dim=self.embed_dim,
                    ff_dim=self.ff_dim,
                    dropout=self.dropout,
                    batch_first=self.batch_first,
                    is_causal=self.is_causal,
                    attn_dropout=self.attn_dropout,
                )
                for _ in range(self.nlayers)
            ]
        )

        self.projection = nn.Linear(self.embed_dim, self.vocab_len)

        self.register_buffer("positions", torch.arange(self.max_len))
        self.tok_emb_table = nn.Embedding(self.vocab_len, self.embed_dim)
        self.pos_emb_table = nn.Embedding(self.max_len, self.embed_dim)
        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.emb_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, X, return_weights=False, mask_id=None):
        """
        Args:
            X (torch.Tensor): Token IDs, shape (batch, seq_len).
            return_weights (bool): Also return per-layer attention weights.
            mask_id (int, optional): Token ID to treat as padding (masked in attention).

        Returns:
            logits (torch.Tensor): (batch, seq_len, vocab_len).
            [attn_weights] (list): Only if return_weights=True.
        """
        if self.batch_first:
            batch_size, seq_len = X.shape[:2]
        else:
            seq_len, batch_size = X.shape[:2]

        tok_emb = self.tok_emb_table(X)
        pos_idx = self.positions[:seq_len]
        pos_emb = self.pos_emb_table(pos_idx)
        if self.batch_first:
            pos_emb = pos_emb.unsqueeze(0)

        x = tok_emb + pos_emb
        x = self.emb_norm(x)
        x = self.emb_drop(x)

        padding_mask = None
        if mask_id is not None:
            padding_mask = (X == mask_id)

        attn_weights = [] if return_weights else None
        for dec in self.decoder:
            x, wt = dec(x, padding=padding_mask)
            if return_weights:
                attn_weights.append(wt)

        logits = self.projection(x)

        if return_weights:
            return logits, attn_weights
        return logits


class LightningGPT(pl.LightningModule):
    """PyTorch Lightning wrapper for decoderGPT: teacher-forced next-token training."""

    def __init__(self, model, pad_idx=0, lr=1e-3, validate=True):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.pad_idx = pad_idx
        self.validate = validate
        self.criterion = (
            nn.CrossEntropyLoss(ignore_index=pad_idx)
            if pad_idx is not None else nn.CrossEntropyLoss()
        )
        self.learning_rate = lr

    def forward(self, X, mask_id=None):
        return self.model(X, mask_id=mask_id)

    def _shared_step(self, batch):
        src, tgt = batch
        src = src.to(dtype=torch.long)
        tgt = tgt.to(dtype=torch.long)

        logits = self(src, mask_id=self.pad_idx)

        B, T, V = logits.shape
        loss = self.criterion(logits.reshape(-1, V), tgt.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if not self.validate:
            return None
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


# ============================================================
# 4. Prediction / generation
# ============================================================

def load_models(config):
    """
    Load an ensemble of decoderGPT checkpoints matching a glob pattern.

    Expects config with keys: model_dir, checkpoint_pattern, device,
    vocab_size, model_args (dict of decoderGPT kwargs).
    """
    model_dir = config["model_dir"]
    ckpt_pattern = config["checkpoint_pattern"]

    ckpt_paths = natsorted(glob(f"{model_dir}/{ckpt_pattern}"))
    device = torch.device(config["device"])

    models = []
    for ckpt_path in ckpt_paths:
        state = torch.load(ckpt_path)["state_dict"]
        state = {k.replace("model.", "", 1): v for k, v in state.items()}

        model = decoderGPT(
            vocab_len=config["vocab_size"],
            **config["model_args"],
        )
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} models")
    return models, device


def predict(models, minLength, maxLength, tokenizer,
            context=128, batch_size=32, reduction="median",
            sampling="multinomial", T=1.0, verbose=True):
    """
    Autoregressively generate protein sequences from one or more models
    (an ensemble is averaged/medianed at each step).
    """
    if not isinstance(models, list):
        models = [models]
    for model in models:
        model.eval()

    inv_tokenizer = {v: k for k, v in tokenizer.items()}
    max_token_val = max(tokenizer.values())
    device = next(models[0].parameters()).device

    reduce_fn = torch.median if reduction == "median" else torch.mean

    sequences = torch.zeros(batch_size, maxLength, dtype=torch.long, device=device)
    sequences[:, 0] = torch.randint(0, max_token_val, size=(batch_size,), device=device)

    with torch.no_grad():
        iterator = trange(1, maxLength, leave=False) if verbose else range(1, maxLength)
        for step in iterator:
            context_start = sequences[:, max(0, step - context):step]

            logits_list = [model(context_start)[:, -1, :] / T for model in models]
            pooled_logits = torch.stack(logits_list)

            pooled_probs = torch.softmax(pooled_logits, dim=-1)
            probs = reduce_fn(pooled_probs, dim=0)
            if isinstance(probs, tuple):
                probs = probs[0]

            if sampling == "greedy":
                next_token = probs.argmax(dim=-1)
            else:
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            sequences[:, step] = next_token

    sequences = sequences.cpu().numpy()
    lengths = np.random.randint(minLength, maxLength, size=batch_size)

    return convertToText(tokens=sequences, lengths=lengths, inv_tokenizer=inv_tokenizer)


def convertToText(tokens, inv_tokenizer, lengths):
    """Convert token ID arrays to protein sequence strings, trimming to length and dropping spaces."""
    sequences = []
    for token_seq, length in zip(tokens, lengths):
        chars = [
            inv_tokenizer[int(t)] for t in token_seq[:length]
            if int(t) in inv_tokenizer and inv_tokenizer[int(t)] != " "
        ]
        sequences.append("".join(chars))
    return sequences
