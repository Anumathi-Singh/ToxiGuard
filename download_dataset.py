"""Download the public Jigsaw training CSV to ToxiGuard's canonical data path.

Run python download_dataset.py before python train_model.py when the source CSV
is not already present.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path


DATASET_URL = (
    "https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments/"
    "resolve/main/train.csv.gz?download=true"
)


def parse_args() -> argparse.Namespace:
    """Return command-line settings for the source CSV destination."""
    parser = argparse.ArgumentParser(description="Download the Jigsaw toxic-comment CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/toxic_comments_dataset.csv"),
        help="Destination CSV path. Default: data/toxic_comments_dataset.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Download and decompress the public CSV used by train_model.py."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {args.output} ...")
    with urllib.request.urlopen(DATASET_URL) as response:
        with gzip.GzipFile(fileobj=response) as compressed:
            with args.output.open("wb") as destination:
                shutil.copyfileobj(compressed, destination)
    print("Download complete.")


if __name__ == "__main__":
    main()
