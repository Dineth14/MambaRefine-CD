"""Entry point for training."""
import argparse

from src.engine.trainer import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/active.yaml")
    args = parser.parse_args()
    train(args.config)
