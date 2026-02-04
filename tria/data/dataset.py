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
from ..util import get_info
from ..util import load_audio
from ..util import rms_salience

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
    ):
        super().__init__()

        assert sources is not None
        assert len(stems) >= 1

        self.stems = list(stems)
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.num_channels = int(num_channels)
        self.relative_path = Path(relative_path)
        self.strict = strict
        self.with_replacement = with_replacement
        self.length = int(n_examples)
        self.shuffle_state = int(shuffle_state)

        self.loudness_cutoff = loudness_cutoff
        self.salience_num_tries = int(salience_num_tries)
        self.salience_on = salience_on or self.stems[0]
        if self.salience_on not in self.stems:
            raise ValueError(
                f"`salience_on` ('{self.salience_on}') must be one of {self.stems}"
            )

        # Read manifests
        csv_paths = [sources] if isinstance(sources, (str, Path)) else list(sources)
        self.source_rows: List[List[Dict]] = []
        kept_mask: List[bool] = []
        kept_csvs: List[Path] = []

        for cpath in csv_paths:
            # Read rows for source
            cpath = Path(cpath)
            raw_rows = []
            with open(cpath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entry = {"__manifest__": str(cpath)}
                    stem_paths = {}
                    for s in self.stems:
                        raw = (row.get(s) or "").strip()
                        stem_paths[s] = str(self._resolve_path(raw)) if raw else ""
                    entry["paths"] = stem_paths
                    extra = {k: v for k, v in row.items() if k not in self.stems}
                    if extra:
                        entry["meta"] = extra
                    raw_rows.append(entry)

            # Filter rows for source
            filtered = []
            for r in raw_rows:
                missing = [
                    s for s, p in r["paths"].items() if not p or not Path(p).is_file()
                ]
                if self.strict and missing:
                    continue

                min_dur = np.inf
                any_valid = False
                for s, p in r["paths"].items():
                    if p and Path(p).is_file():
                        any_valid = True
                        try:
                            total_sec = float(sf.info(p).duration)
                            min_dur = min(min_dur, float(total_sec))
                        except Exception:
                            if self.strict:
                                min_dur = -np.inf
                                break
                if not any_valid or not np.isfinite(min_dur):
                    continue

                r["min_duration"] = min_dur if np.isfinite(min_dur) else 0.0
                filtered.append(r)

            if len(filtered) > 0:
                self.source_rows.append(filtered)
                kept_mask.append(True)
                kept_csvs.append(cpath)
            else:
                kept_mask.append(False)

        if len(self.source_rows) == 0:
            raise RuntimeError(
                "StemDataset: no valid rows after filtering in any source."
            )

        self.csv_paths = kept_csvs

        lengths = [len(lst) for lst in self.source_rows]
        self._source_offsets = np.cumsum([0] + lengths[:-1])  # for global idx
        self._n_rows = int(sum(lengths))

        # Weights over non-empty sources
        if source_weights is None:
            self._weights = None
        else:
            if len(source_weights) != len(csv_paths):
                raise ValueError(
                    f"source_weights must match number of sources ({len(csv_paths)}), "
                    f"got {len(source_weights)}"
                )
            w = np.asarray(source_weights, dtype=float)

            # Keep only weights for sources that survived filtering
            w = w[np.array(kept_mask, dtype=bool)]
            w = np.clip(w, 0, None)
            if not np.any(w > 0):
                w = np.ones_like(w)
            self._weights = (w / w.sum()).tolist()

    def _resolve_path(self, p: Union[str, Path]) -> Path:
        p = Path(p).expanduser()
        if not p.is_absolute():
            p = (self.relative_path / p).expanduser()
        return p

    def _pick_row(self, state: np.random.RandomState):
        # Sample a non-empty source
        sidx = int(state.choice(len(self.source_rows), p=self._weights))
        n_in_source = len(self.source_rows[sidx])
        item_idx = int(state.randint(n_in_source))
        row = self.source_rows[sidx][item_idx]

        # Map to a global idx for metadata
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
            if self.loudness_cutoff is None or not self.salience_num_tries:
                try:
                    total_sec, _sr = get_info(p0)
                except Exception:
                    total_sec = 0.0
                max_off = max(0.0, total_sec - self.duration)
                max_off = min(max_off, max_off_all)
                offset = float(state.rand() * max_off) if max_off > 0 else 0.0
            else:
                offset = rms_salience(
                    p0,
                    duration=self.duration,
                    cutoff_db=float(self.loudness_cutoff),
                    num_tries=int(self.salience_num_tries),
                    state=state,
                )
                offset = min(max(0.0, offset), max_off_all)

            offset = min(offset, max_valid_offset)
            primary_sig = load_audio(p0, offset=offset, duration=self.duration)
        else:
            offset = 0.0

        item: Dict[str, Dict] = {}
        for s in self.stems:
            p = row["paths"][s]
            exists = bool(p) and Path(p).is_file()

            if s == primary and primary_sig is not None:
                sig = primary_sig.clone()  # Reuse window
            elif exists:
                sig = load_audio(
                    p, offset=offset, duration=self.duration
                )  # Windowed load
            else:
                sig = AudioSignal.zeros(
                    self.duration, self.sample_rate, self.num_channels
                )

            # Channel formatting
            if self.num_channels == 1:
                sig = sig.to_mono()
            elif self.num_channels != sig.num_channels:
                assert sig.num_channels == 1
                sig.audio_data = sig.audio_data.repeat(1, self.num_channels, 1)

            # Resample/pad to target sample rate and exact duration
            sig = sig.resample(self.sample_rate)

            # DEBUG: do not enforce duration in dataset, but instead handle
            # padding during collation
            # if sig.duration < self.duration:
            #    sig = sig.zero_pad_to(int(self.duration * self.sample_rate))

            # Metadata
            sig.metadata["path"] = p
            sig.metadata["offset"] = offset
            sig.metadata["source_row"] = ridx
            if "meta" in row:
                for k, v in row["meta"].items():
                    sig.metadata[k] = v

            item[s] = {"signal": sig, "path": p}

        item["idx"] = idx
        return item

    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        return collate(list_of_dicts, n_splits=n_splits)
