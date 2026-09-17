#!/usr/bin/env python3
"""
Train the idpGPT transformer decoder on protein sequences.

At the end of training, the best checkpoint (by validation loss) is
re-saved as a plain .pt file containing the model's state_dict plus
the architecture kwargs needed to reconstruct it -- this is the file
generate.py expects.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from idpgpt import decoderGPT, LightningGPT, setSeeds

torch.set_float32_matmul_precision("medium")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the idpGPT transformer decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer YAML file")
    p.add_argument("--train-data", type=str, required=True, help="Path to training data pickle (RandomChunkLoader dataset)")
    p.add_argument("--val-data", type=str, required=True, help="Path to validation data pickle (RandomChunkLoader dataset)")

    # Model architecture
    p.add_argument("--nheads", type=int, required=True, help="Number of attention heads")
    p.add_argument("--nlayers", type=int, required=True, help="Number of transformer blocks")
    p.add_argument("--embed-dim", type=int, required=True, help="Embedding dimension")
    p.add_argument("--ff-dim", type=int, required=True, help="Feed-forward dimension")
    p.add_argument("--max-len", type=int, required=True, help="Maximum sequence length")

    # Training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accelerator", type=str, default="cpu", choices=["cpu", "gpu", "mps"])
    p.add_argument("--devices", type=int, nargs="+", default=[0])
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # Early stopping
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--early-stop-metric", type=str, default="val_loss")

    # Output
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for Lightning checkpoints")
    p.add_argument("--save-top-k", type=int, default=3)
    p.add_argument("--best-weights-out", type=str, default="best_model.pt",
                    help="Where to save the final best model weights (.pt) for use with generate.py")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--experiment-name", type=str, default="idpGPT")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Arguments: %s", args)

    setSeeds(args.seed)

    # Tokenizer / vocab
    with open(args.tokenizer_path, "r") as f:
        tokenizer = yaml.safe_load(f)
    vocab_len = len(tokenizer)
    logger.info("Vocabulary size: %d", vocab_len)

    # Data
    with open(args.train_data, "rb") as f:
        train_loader = DataLoader(pickle.load(f), batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)
    with open(args.val_data, "rb") as f:
        val_loader = DataLoader(pickle.load(f), batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    logger.info("Training samples: %d, Validation samples: %d",
                len(train_loader.dataset), len(val_loader.dataset))

    # Model
    model_args = dict(
        nheads=args.nheads,
        nlayers=args.nlayers,
        embed_dim=args.embed_dim,
        ff_dim=args.ff_dim,
        max_len=args.max_len,
    )
    model = decoderGPT(vocab_len=vocab_len, **model_args)
    logger.info("Model created with %d parameters", sum(p.numel() for p in model.parameters()))

    lightning_model = LightningGPT(model=model, pad_idx=None, lr=args.lr)

    # Callbacks
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="idpGPT-{epoch:02d}-{val_loss:.4f}",
        monitor=args.early_stop_metric,
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor=args.early_stop_metric,
        mode="min",
        patience=args.early_stop_patience,
        verbose=True,
    )
    csv_logger = CSVLogger(save_dir=args.log_dir, name=args.experiment_name)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=csv_logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    logger.info("Starting training")
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("Training complete. Best checkpoint: %s", checkpoint_callback.best_model_path)

    # Re-save best checkpoint as a plain .pt (state_dict + architecture) for generate.py
    best_path = checkpoint_callback.best_model_path
    if not best_path:
        logger.warning("No best checkpoint recorded (did validation ever improve?); "
                        "saving current in-memory weights instead.")
        state_dict = lightning_model.model.state_dict()
    else:
        ckpt = torch.load(best_path, map_location="cpu")
        state_dict = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}

    torch.save(
        {
            "state_dict": state_dict,
            "model_args": model_args,
            "vocab_len": vocab_len,
        },
        args.best_weights_out,
    )
    logger.info("Saved best model weights to %s", args.best_weights_out)


if __name__ == "__main__":
    main()
