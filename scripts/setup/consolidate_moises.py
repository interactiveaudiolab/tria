import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from itertools import chain

from rich.progress import track
from audiotools import AudioSignal

# Allow local imports
@contextmanager
def chdir(path: str):
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_path))

with chdir(_path):
    from tria.constants import DATA_DIR
    from tria.util import ensure_dir_for_filename, get_info

warnings.filterwarnings("ignore", category=UserWarning)

################################################################################
# Consolidate MoisesDB multitracks by summing
################################################################################


def _sum(in_paths: list, out_path: Path, max_length: int = None):
    in_paths = list(in_paths)
    if not in_paths:
        return False
    ensure_dir_for_filename(str(out_path))
    sigs = [AudioSignal(p) for p in in_paths]
    batch = AudioSignal.batch(sigs, pad_signals=True)

    if max_length is not None:
        batch = batch.zero_pad_to(max_length, mode="after")
    
    batch.audio_data = batch.audio_data.sum(dim=0, keepdim=True)
    batch = batch.ensure_max_of_audio()
    batch.write(out_path)
    return True


def main():

    print("Consolidating MoisesDB stems")
    
    moises_dir = DATA_DIR / "moisesdb" / "moisesdb_v0.1"

    _stems = ["drums", "vocals", "bass"]
    stems = _stems + ["other", "mixture"]
    
    n_tracks = 0
    n_cons = {stem: 0 for stem in stems}

    for _track in track(
        list(moises_dir.iterdir()),
        description="Consolidating MoisesDB stems"
    ):  
        if not _track.is_dir() or _track.name.startswith("."):
            continue

        # Multitracks are arranged in subdirectories        
        subdirs = [
            d for d in _track.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        
        # Collect drum files
        drum_dirs = [d for d in subdirs if ("drum" in d.name.lower()) or ("percussion" in d.name.lower())]
        drum_files = sorted({p for d in drum_dirs for p in d.rglob("*.wav")})

        stem_files = {}
        stem_files["drums"] = list(drum_files)

        # Collect vocals/bass files
        for stem in ("vocals", "bass"):
            d = _track / stem
            stem_files[stem] = list(d.rglob("*.wav")) if d.is_dir() else []

        # Collect all other files
        excluded_dirs = set(drum_dirs + [(_track / "vocals"), (_track / "bass")])
        other_dirs = [d for d in subdirs if d not in excluded_dirs]
        other_files = list(chain.from_iterable(d.rglob("*.wav") for d in other_dirs))
        stem_files["other"] = other_files

        # All files for mixture
        all_stem_files = list(chain.from_iterable(stem_files.values()))

        # Determine maximum audio length
        all_metadata = [get_info(f) for f in all_stem_files]  # Duration, sample rate
        max_length = max([int(m[0] * m[1]) for m in all_metadata])
        
        # Write drum/bass/vocal files
        wrote_any = False
        for stem in _stems:
            out_f = _track / f"{stem}.wav"
            if _sum(stem_files[stem], out_f, max_length):
                n_cons[stem] += 1
                wrote_any = True

        # Write other
        out_other = _track / "other.wav"
        if _sum(stem_files["other"], out_other, max_length):
            n_cons["other"] += 1
            wrote_any = True

        # Write mixture of all files
        out_mix = _track / "mixture.wav"
        if _sum(all_stem_files, out_mix, max_length):
            n_cons["mixture"] += 1
            wrote_any = True
        else:
            pass

        if wrote_any:
            n_tracks += 1

    print(
        "Done.\n"
        f"Tracks: {n_tracks}\n"
        f"Containing 'drums':   {n_cons['drums']}\n"
        f"Containing 'vocals':  {n_cons['vocals']}\n"
        f"Containing 'bass':    {n_cons['bass']}\n"
        f"Containing 'other':   {n_cons['other']}\n"
        f"Containing 'mixture': {n_cons['mixture']}"
    )

if __name__ == "__main__":
    main()
