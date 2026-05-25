"""Entry point for training.

Reads: configs/active.yaml
Usage: python train.py
"""
from src.engine.trainer import train


if __name__ == "__main__":
    train()
