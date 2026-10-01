"""Backward-compatible entry point for the ToxiGuard training workflow.

Use python train_model.py in project instructions. This file remains so older
commands keep working, while all training logic has one source of truth.
"""

from train_model import main


if __name__ == "__main__":
    main()
