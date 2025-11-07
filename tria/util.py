import math
import os
import sys
from pathlib import Path
from typing import Optional
from typing import Union

import librosa
import numpy as np
import rich
import soundfile as sf
import torch
from audiotools import AudioSignal
from audiotools.core.util import random_state
from flatten_dict import flatten
from flatten_dict import unflatten


################################################################################
# General utilities
################################################################################


def count_parameters(m: torch.nn.Module, trainable: bool = False):
    if trainable:
        return sum([p.shape.numel() for p in m.parameters() if p.requires_grad])
    else:
        return sum([p.shape.numel() for p in m.parameters()])


def exists(val):
    return val is not None


def print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not local_rank:
        rich.print(*args, **kwargs, file=sys.stderr)


def ensure_dir(directory: Union[str, Path]):
    directory = str(directory)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def ensure_dir_for_filename(filename: str):
    ensure_dir(os.path.dirname(filename))


def collate(list_of_dicts: list, n_splits: int = None):
    """
    Collates a list of dictionaries (e.g. as returned by a dataloader) into a
    dictionary with batched values. This function takes `n_splits` to enable
    splitting a batch into multiple sub-batches for the purposes of gradient
    accumulation, etc. Adapted from `audiotools.core.util.collate`.

    Parameters
    ----------
    list_of_dicts : list
        List of dictionaries to be collated.
    n_splits : int
        Number of splits to make when creating the batches (split into sub-
        batches). Useful for things like gradient accumulation.

    Returns
    -------
    dict
        Dictionary containing batched data.
    """

    batches = []
    list_len = len(list_of_dicts)

    return_list = False if n_splits is None else True
    n_splits = 1 if n_splits is None else n_splits
    n_items = int(math.ceil(list_len / n_splits))

    for i in range(0, list_len, n_items):
        list_of_dicts_ = [flatten(d) for d in list_of_dicts[i : i + n_items]]
        dict_of_lists = {
            k: [dic[k] for dic in list_of_dicts_] for k in list_of_dicts_[0]
        }

        batch = {}
        for k, v in dict_of_lists.items():
            if not isinstance(v, list):
                continue

            # AudioSignal → pad & batch
            if all(isinstance(s, AudioSignal) for s in v):
                batch[k] = AudioSignal.batch(v, pad_signals=True)

            # Strings / Paths → keep as list
            elif all(isinstance(s, (str, Path)) for s in v):
                batch[k] = v

            # All None → keep as list
            elif all(s is None for s in v):
                batch[k] = v

            else:
                # Fallback to torch default collate (tensors, numbers, mappings, etc.)
                try:
                    batch[k] = torch.utils.data._utils.collate.default_collate(v)
                except TypeError:
                    # Last-resort: keep raw list
                    batch[k] = v

        batches.append(unflatten(batch))

    return batches[0] if not return_list else batches


def get_info(path: Union[str, Path]):
    info = sf.info(str(path))
    return float(info.duration), int(info.samplerate)


def load_audio(
    path: Union[str, Path],
    offset: float,
    duration: float,
    file_sample_rate: Optional[int] = None,
):
    """
    SoundFile windowed loading seems to outperform `librosa.load` (used
    throughout `AudioSignal`) in limiting memory consumption; this helps avert
    crashes when training with large `num_workers`.
    """
    if file_sample_rate is None:
        _duration, sample_rate = get_info(path)
    start = int(offset * sample_rate)
    n_samples = int(duration * sample_rate)

    with sf.SoundFile(str(path), "r") as f:
        f.seek(start)
        x = f.read(
            n_samples, dtype="float32", always_2d=True
        ).T  # (n_channels, n_samples)
    x = torch.from_numpy(x)[None, :, :]  # (n_batch==1, n_channels, n_samples)

    return AudioSignal(x, sample_rate=sample_rate)


def rms_salience(
    path: str,
    duration: float,
    cutoff_db: float = -40.0,
    num_tries: int = 3,
    state: Optional[int] = None,
    file_duration: Optional[float] = None,
    file_sample_rate: Optional[int] = None,
) -> float:
    if file_duration is None or file_sample_rate is None:
        _duration, sample_rate = get_info(path)
    else:
        _duration, sample_rate = file_duration, file_sample_rate

    if not np.isfinite(_duration) or _duration <= 0 or _duration <= duration:
        return 0.0

    state = random_state(state)
    max_offset = _duration - duration
    n_samples = int(duration * sample_rate)

    tries = max(1, int(num_tries))
    best_db = -np.inf
    best_offset = None

    with sf.SoundFile(str(path), "r") as f:
        for _ in range(tries):
            offset = float(state.rand() * max_offset)
            start = int(offset * sample_rate)
            try:
                f.seek(start)
                y = f.read(
                    n_samples, dtype="float32", always_2d=True
                )  # (n_samples, n_channels)
                y = y.mean(axis=1, dtype=np.float32)  # (n_samples,)
                rms = float(np.sqrt(np.mean(y * y) + 1e-12))
                db = 20.0 * np.log10(max(rms, 1e-12))
            except Exception:
                continue

            if db >= cutoff_db:
                return offset
            if db > best_db:
                best_db, best_offset = db, offset

    return float(best_offset if best_offset is not None else state.rand() * max_offset)
