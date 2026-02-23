import csv
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import soundfile as sf
from audiotools import AudioSignal
from audiotools.core.util import random_state
from torch.utils.data import Dataset

from ..constants import DURATION
from ..constants import SAMPLE_RATE
from ..constants import STEMS
from ..util import collate
from ..util import read_manifest, normalize_source_weights, load_excerpt



################################################################################
# Dataset for loading aligned excerpts across stem classes
################################################################################


class StemDataset(Dataset):
    """
    Load aligned excerpts from specified stem classes given paths in one or more
    CSV manifests. Based on `audiotools.data.datasets.AudioDataset`.

    Parameters
    ----------
    sources : Union[str, Path, List[Union[str, Path]]]
        CSV manifest(s) with columns for each requested stem.
    stems : List[str]
        Column names to load, e.g. ["mixture","drums","bass","vocals"].
        The **first** stem is used for salience unless `salience_on` is set.
    sample_rate : int
    duration : float
    n_examples : int
    num_channels : int
    relative_path : str
        Prepended to relative CSV paths.
    strict : bool
        Drop rows with missing stems (True) vs. fill with silence (False).
    with_replacement : bool
        Sampling strategy for rows.
    shuffle_state : int
        Seed for deterministic per-index RNG.
    loudness_cutoff : Optional[float]
        dB LUFS cutoff; if None, take random excerpt (still shared across stems).
    salience_num_tries : int
        Max tries for salient excerpt search (see `AudioSignal.salient_excerpt`).
    salience_on : Optional[str]
        Which stem to use for salience. Defaults to first of `stems`.
    from_start : bool
        If `True`, load audio from beginning of file rather than randomly excerpting
    """

    def __init__(
        self,
        stems: List[str] = STEMS,
        sample_rate: int = SAMPLE_RATE,
        duration: float = DURATION,
        sources: Union[str, Path, List[Union[str, Path]]] = None,
        source_weights: Optional[List[float]] = None,
        n_examples: int = 1000,
        num_channels: int = 1,
        relative_path: str = "",
        strict: bool = True,
        with_replacement: bool = True,
        shuffle_state: int = 0,
        loudness_cutoff: Optional[float] = -40.0,
        salience_num_tries: int = 8,
        salience_on: Optional[str] = None,
        from_start: bool = False,
    ):
        super().__init__()

        assert sources is not None
        assert len(stems) >= 1

        self.stems = list(stems)
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.num_channels = int(num_channels)
        self.relative_path = str(relative_path)
        self.strict = bool(strict)
        self.with_replacement = bool(with_replacement)
        self.length = int(n_examples)
        self.shuffle_state = int(shuffle_state)

        self.loudness_cutoff = loudness_cutoff
        self.salience_num_tries = int(salience_num_tries)
        self.salience_on = salience_on or self.stems[0]
        self.from_start = bool(from_start)

        if self.salience_on not in self.stems:
            raise ValueError(
                f"`salience_on` ('{self.salience_on}') must be one of {self.stems}"
            )

        # Read + filter manifests (per-CSV lists)
        per_source_rows = read_manifest(
            sources=sources,
            columns=self.stems,
            relative_path=self.relative_path,
            strict=self.strict,
        )

        # Keep only non-empty sources, and apply per-CSV weights
        kept_mask = [len(lst) > 0 for lst in per_source_rows]
        self.source_rows: List[List[Dict]] = [
            lst for lst in per_source_rows if len(lst) > 0
        ]
        if len(self.source_rows) == 0:
            raise RuntimeError(
                "StemDataset: no valid rows after filtering in any source."
            )

        csv_paths = [sources] if isinstance(sources, (str, Path)) else list(sources)
        self.csv_paths = [
            Path(p).expanduser() for p, keep in zip(csv_paths, kept_mask) if keep
        ]

        self._weights = normalize_source_weights(
            source_weights=source_weights,
            n_sources=len(csv_paths),
            kept_mask=kept_mask,
        )

        # Global row indexing
        lengths = [len(lst) for lst in self.source_rows]
        self._source_offsets = np.cumsum([0] + lengths[:-1])
        self._n_rows = int(sum(lengths))

    def _pick_row(self, state: np.random.RandomState):
        sidx = int(state.choice(len(self.source_rows), p=self._weights))
        n_in_source = len(self.source_rows[sidx])
        item_idx = int(state.randint(n_in_source))
        row = self.source_rows[sidx][item_idx]
        ridx_global = int(self._source_offsets[sidx] + item_idx)
        return ridx_global, row

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        state = random_state((self.shuffle_state + int(idx)) & 0x7FFFFFFF)
        ridx, row = self._pick_row(state)

        shortest = float(row.get("min_duration", np.inf))
        max_off_all = max(0.0, shortest - self.duration)
        eps = 1.0 / float(self.sample_rate)
        max_valid_offset = max(0.0, max_off_all - eps)

        primary = self.salience_on
        p0 = row["paths"].get(primary, "")

        offset = 0.0
        primary_sig = None

        if p0 and Path(p0).is_file():
            primary_sig, offset = load_excerpt(
                p0,
                duration=self.duration,
                sample_rate=self.sample_rate,
                state=state,
                from_start=self.from_start,
                loudness_cutoff=self.loudness_cutoff,
                num_tries=self.salience_num_tries,
                num_channels=self.num_channels,
                max_offset=max_off_all,
            )
            offset = min(max(0.0, offset), max_valid_offset)
        else:
            offset = 0.0

        item: Dict[str, Dict] = {}
        for s in self.stems:
            p = row["paths"].get(s, "")
            exists = bool(p) and Path(p).is_file()

            if s == primary and primary_sig is not None:
                sig = primary_sig.clone()
            elif exists:
                sig, _ = load_excerpt(
                    p,
                    duration=self.duration,
                    sample_rate=self.sample_rate,
                    state=state,
                    offset=offset,  # force alignment; offset wins
                    num_channels=self.num_channels,
                )
            else:
                sig = AudioSignal.zeros(
                    self.duration, self.sample_rate, self.num_channels
                )

            sig.metadata["path"] = p
            sig.metadata["offset"] = offset
            sig.metadata["source_row"] = ridx  # restore global id
            if "meta" in row:
                for k, v in row["meta"].items():
                    sig.metadata[k] = v

            item[s] = {"signal": sig, "path": p}

        item["idx"] = idx
        return item

    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        return collate(list_of_dicts, n_splits=n_splits)
