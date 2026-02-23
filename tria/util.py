import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import csv
import librosa
import numpy as np
import rich
import soundfile as sf
import torch
from audiotools import AudioSignal
from audiotools.core.util import random_state
from flatten_dict import flatten
from flatten_dict import unflatten

from . import dsp

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

    Cases are as follows:
      * AudioSignal objects, which wrap tensors of shape (b, c, t), are padded
        to t_max and batched
      * 3+ dimensional tensors of shape (b, ..., c, t) are padded to t_max and
        batched
      * 2 dimensional tensors of shape (b, c) are batched without padding,
        assuming a matching channel dimension
      * 1 dimensional tensors of shape (b,) are batched
      * Remaining data types are handled via `default_collate`

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

            # Determine valid lengths key, extracting name from flattened (tuple) key
            k_lengths = k[:-1] + (f"{k[-1]}_lengths",)
            example = v[0]

            # AudioSignal → pad & batch
            if all(isinstance(s, AudioSignal) for s in v):
                lengths = torch.tensor(
                    [s.signal_length for s in v],
                    dtype=torch.long,
                )  # (n_batch,)

                # Batch signals
                batch[k] = AudioSignal.batch(v, pad_signals=True)
                batch[k_lengths] = lengths

            # Tensor → (possibly) pad & batch
            elif all(isinstance(s, torch.Tensor) for s in v):
                # Assume matching number of dimensions across batch
                ndim = example.ndim

                # Assume (b, ..., c, t)
                if ndim >= 3:
                    lengths = torch.tensor(
                        [s.shape[-1] for s in v], dtype=torch.long
                    )  # (n_batch,)

                    # Stack along first dimension
                    tgt_shape = (len(v), *example.shape[1:-1], lengths.amax().item())

                    # Pad to max length
                    batch[k] = torch.zeros(tgt_shape, dtype=example.dtype)
                    for i in range(len(v)):
                        batch[k][i, ..., : lengths[i]] = v[i]
                    batch[k_lengths] = lengths

                # Assume (b,) or (b, c)
                elif ndim in [1, 2]:
                    batch[k] = torch.stack(v, dim=0)

                # Fall back to default
                else:
                    batch[k] = torch.utils.data._utils.collate.default_collate(v)

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
    else:
        sample_rate = file_sample_rate
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


def read_manifest(
    sources: Union[str, Path, Sequence[Union[str, Path]]],
    columns: Sequence[str],
    relative_path: Union[str, Path] = "",
    strict: bool = True,
) -> List[List[Dict]]:
    """
    Read one or more CSV manifests into a list of row dicts, each containing keys:
      * "__manifest__": the source manifest as a string path
      * "paths": any file paths in the row under the specified columns
      * "min_duration": minimum file duration across specified columns
      * "meta": any remaining metadata

    Parameters
    ----------
    sources : Sequence[str]
        One or more CSV manifest paths
    columns : Sequence[str]
        Column names holding paths to audio files
    relative_path : str
        Relative path to prepend to file paths
    strict : bool
        If `True`, drop rows where any requested column is missing

    Returns
    -------
    List[List[Dict]]
        A list of lists of row dicts, with one outer list per manifest
    """
    assert sources is not None
    cols = list(columns)
    rel = Path(relative_path).expanduser()
    csv_paths = [sources] if isinstance(sources, (str, Path)) else list(sources)

    out: List[List[Dict]] = []
    for cpath in csv_paths:
        cpath = Path(cpath).expanduser()
        rows: List[Dict] = []

        with open(cpath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {"__manifest__": str(cpath)}
                paths: Dict[str, str] = {}

                for col in cols:
                    raw = (row.get(col) or "").strip()
                    if raw:
                        p = Path(raw).expanduser()
                        if not p.is_absolute():
                            p = (rel / p).expanduser()
                        paths[col] = str(p)
                    else:
                        paths[col] = ""

                entry["paths"] = paths
                extra = {k: v for k, v in row.items() if k not in cols}
                if extra:
                    entry["meta"] = extra
                rows.append(entry)

        filtered: List[Dict] = []
        for r in rows:
            missing = [c for c, p in r["paths"].items() if not p or not Path(p).is_file()]
            if strict and missing:
                continue

            any_valid = False
            min_dur = np.inf
            for _c, p in r["paths"].items():
                if p and Path(p).is_file():
                    any_valid = True
                    try:
                        min_dur = min(min_dur, float(sf.info(p).duration))
                    except Exception:
                        if strict:
                            min_dur = -np.inf
                            break

            if not any_valid or not np.isfinite(min_dur) or min_dur <= 0:
                continue

            r["min_duration"] = float(min_dur)
            filtered.append(r)

        out.append(filtered)

    return out


def normalize_source_weights(
    source_weights: Optional[Sequence[float]],
    n_sources: int,
    kept_mask: Optional[Sequence[bool]] = None,
) -> Optional[List[float]]:
    """
    Normalize per-CSV-file sampling weights. If kept_mask is provided, drop weights for
    sources that were filtered out.
    """
    if source_weights is None:
        return None

    w = np.asarray(list(source_weights), dtype=float)
    if w.shape[0] != int(n_sources):
        raise ValueError(
            f"source_weights must match number of sources ({n_sources}), got {w.shape[0]}"
        )

    if kept_mask is not None:
        kept = np.asarray(list(kept_mask), dtype=bool)
        w = w[kept]

    w = np.clip(w, 0.0, None)
    if not np.any(w > 0):
        w = np.ones_like(w)
    w = w / w.sum()
    return w.tolist()


def pick_offset(
    path: Union[str, Path],
    duration: float,
    sample_rate: int,
    state: np.random.RandomState,
    *,
    from_start: bool = False,
    loudness_cutoff: Optional[float] = None,
    num_tries: int = 0,
    file_duration: Optional[float] = None,
    file_sample_rate: Optional[int] = None,
    max_offset: Optional[float] = None,
) -> float:
    """
    Select offset at which to excerpt audio

    Parameters
    ----------
    from_start : bool
        If `True`, load from beginning of file
    loudness_cutoff : float
        If provided and `from_start` is `False`, search for random excerpt that
        satisfies loudness criterion. If not provided and `from_start` is `False`, 
        sample offset uniformly at random
    max_offset : float
        Maximum offset to consider; useful for when dealing with aligned files 
        with different durations

    Returns
    -------
    float
        Offset in seconds
    """
    if from_start:
        return 0.0

    if file_duration is None or file_sample_rate is None:
        total_sec, sr = get_info(path)
    else:
        total_sec, sr = float(file_duration), int(file_sample_rate)

    if not np.isfinite(total_sec) or total_sec <= 0 or total_sec <= float(duration):
        return 0.0

    max_off = max(0.0, float(total_sec) - float(duration))

    if max_offset is not None:
        max_off = min(max_off, float(max_offset))

    eps = 1.0 / float(sample_rate)
    max_valid_offset = max(0.0, max_off - eps)

    if loudness_cutoff is None or not int(num_tries):
        off = float(state.rand() * max_off) if max_off > 0 else 0.0
    else:
        off = float(
            rms_salience(
                str(path),
                duration=float(duration),
                cutoff_db=float(loudness_cutoff),
                num_tries=int(num_tries),
                state=state,
                file_duration=float(total_sec),
                file_sample_rate=int(sr),
            )
        )
        # keep old behavior: clamp after salience search
        off = min(max(0.0, off), max_valid_offset)

    return float(min(max(0.0, off), max_valid_offset))


def load_excerpt(
    path: Union[str, Path],
    *,
    duration: float,
    sample_rate: int,
    state: np.random.RandomState,
    from_start: bool = False,
    loudness_cutoff: Optional[float] = None,
    num_tries: int = 0,
    offset: Optional[float] = None,
    num_channels: Optional[int] = None,
    max_offset: Optional[float] = None,
    resample: bool = True,
) -> Tuple[AudioSignal, float]:
    """
    Load excerpt from given audio file
    """
    p = str(path)
    if offset is None:
        off = pick_offset(
            p,
            duration=float(duration),
            sample_rate=int(sample_rate),
            state=state,
            from_start=bool(from_start),
            loudness_cutoff=loudness_cutoff,
            num_tries=int(num_tries),
            max_offset=max_offset,
        )
    else:
        off = float(offset)

    sig = load_audio(p, offset=off, duration=float(duration))
    
    if num_channels is not None:
        nc = int(num_channels)
        if nc == 1:
            if sig.num_channels != 1:
                sig = sig.to_mono()
        elif sig.num_channels != nc:
            if sig.num_channels == 1:
                sig.audio_data = sig.audio_data.repeat(1, nc, 1)
            else:
                # Fall back: downmix & repeat
                sig.audio_data = sig.audio_data.mean(dim=1, keepdim=True).repeat(1, nc, 1)
                 
    # Optional resample (StemDataset will keep this True; aug instantiation can set False)
    if resample:
        sig = dsp.resample(sig, int(sample_rate))
 
    return sig, float(off)
