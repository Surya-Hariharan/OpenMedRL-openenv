"""Thin wrapper that calls the `triagerl.training.train` orchestration.

Keeps top-level `train.py` as the CLI entrypoint while delegating
implementation to the package-internal module.
"""
from triagerl.training.train import main


if __name__ == "__main__":
    main()
