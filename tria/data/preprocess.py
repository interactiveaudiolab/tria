import csv
import os
from pathlib import Path
from typing import Callable, Dict, Tuple, Union, Optional, Any
from rich.progress import track

import numpy as np

from audiotools.core.util import random_state
from ..util import ensure_dir

SplitType = Union[Tuple[float, float, float], Callable[[Path], str]]


def create_manifests(
    data_dir: Union[str, Path],
    ext: str,
    output_dir: Union[str, Path],
    split: SplitType,
    attributes: Dict[str, Callable[[Path], Any]],
    seed: Optional[int] = 0,
) -> Dict[str, Path]:
    """
    Create CSV manifests for audio dataset.

    Parameters
    ----------
    data_dir : str
        Dataset root directory to search recursively for files
    ext : str
        Audio file extension
    output_dir : str
        Directory to which to write manifests
    split : SplitType
        Either a 3-tuple containing (train, val, test) proportions summing to 1
        or a Callable that returns "train", "val", or "test" given a filepath
    attributes : dict
        Dictionary mapping column names to Callables for extracting values
        given filepaths; for example {'path': lambda p: str(p)}
    seed : int
        Random seed
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    all_files = sorted(
        [p for p in data_dir.rglob(f"*{ext}") if p.is_file()],
        key=lambda p: str(p).lower(),
    )

    splits = {"train": [], "val": [], "test": []}

    # Callable split: apply given function to file paths to obtain train/val/test
    # assignments
    if callable(split):
        for p in all_files:
            s = split(p)
            if s not in splits:
                raise ValueError(
                    f"Split function must return one of "
                    f"{list(splits.keys())}, got {s!r} for {p}"
                )
            splits[s].append(p)
            
    # Proportional split: randomly shuffle files and split according to given
    # values
    else:
        if not (isinstance(split, tuple) and len(split) == 3):
            raise ValueError(f"Split proportions tuple must have length 3")
        p_train, p_val, p_test = split
        total = float(p_train + p_val + p_test)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Split proportions must sum to 1.0 (got {total}).")

        rs = random_state(seed)
        idx = np.array(rs.permutation(len(all_files)))
        n = len(idx)
        n_train = int(np.floor(p_train * n))
        n_val = int(np.floor(p_val * n))
        n_test = n - n_train - n_val

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        for i in train_idx:
            splits["train"].append(all_files[int(i)])
        for i in val_idx:
            splits["val"].append(all_files[int(i)])
        for i in test_idx:
            splits["test"].append(all_files[int(i)])
    
    columns = list(attributes.keys())

    # Write CSVs
    out_paths: Dict[str, Path] = {}
    for s in ("train", "val", "test"):
        out_csv = output_dir / f"{s}.csv"
        out_paths[s] = out_csv

        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for p in track(
                splits[s], 
                description=f"Writing {s}.csv", 
                total=len(splits[s])
            ):

                try:
                    row = {}
                    for col, fn in attributes.items():
                        row[col] = fn(p)
                    writer.writerow(row)
                except Exception as e:
                    print(
                        f"Error at path {p}:\n"
                        f"{e}\n"
                        f"Skipping..."
                    )

    return out_paths
