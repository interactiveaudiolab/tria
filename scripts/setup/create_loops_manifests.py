import json
import os
import re
import shutil
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

import argbind
from rich.progress import track


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
    from tria.data.preprocess import create_manifests
    from tria.constants import DATA_DIR, MANIFESTS_DIR
    from tria.util import get_info

warnings.filterwarnings("ignore", category=UserWarning)

################################################################################
# Create CSV manifests for consolidated MoisesDB dataset
################################################################################


@argbind.bind(without_prefix=True)
def main(seed: int = 0):
    loops_dir = DATA_DIR / "FSL10K"

    # Define valid subset of loops
    with open(loops_dir / "metadata.json") as f:
        d = json.load(f)

    all_loops = list((loops_dir / "audio").rglob("*.wav"))
    valid_loops = set()

    for i, k in track(
        enumerate(d.keys()), description=f"Filtering FSL", total=len(all_loops)
    ):
        metadata = d[k]

        # Filter out loops with non-drum instrument tags
        other_inst = ["guitar", "keyboard", "vocal", "bass", "piano"]
        contains_drums = False

        tags = [t.lower().strip() for t in metadata["tags"]]
        for t in tags:
            if any([inst in t for inst in other_inst]):
                contains_drums = False
                break
            if "drum" in t:
                contains_drums = True

        if not contains_drums:
            continue

        # Find candidate file path by stored ID
        raw_path = metadata["preview_url"].split("/")[-1]
        match = re.search(r"\d+_\d+", raw_path)
        if match:
            fid = match.group()
        else:
            continue

        path_1 = loops_dir / "audio" / "wav" / f"{fid}.wav.wav"
        path_2 = loops_dir / "audio" / "wav" / f"{fid}.aiff.wav"

        if path_1 in all_loops:
            path = path_1
        elif path_2 in all_loops:
            path = path_2
        else:
            continue

        # Filter by duration and sample rate
        duration, sample_rate = get_info(path)
        if duration >= 4.0 and sample_rate >= 22_050:
            valid_loops.add(str(path))

    def _fetch_drums(p):
        p = str(p)
        if p in valid_loops:
            return p
        else:
            raise ValueError(f"Invalid loop path {p}")

    # Fetch audio sample rate in Hz and duration in seconds; assume all fetched
    # loops contain drums only
    metadata_fns = {
        "sample_rate": lambda p: get_info(str(p))[1],
        "duration": lambda p: get_info(str(p))[0],
        "drums": _fetch_drums,
    }

    print("Creating manifests for FSL")
    out_paths = create_manifests(
        DATA_DIR / "FSL10K",
        ext=".wav",
        output_dir=MANIFESTS_DIR / "fsl",
        split=(0.8, 0.1, 0.1),
        attributes=metadata_fns,
        seed=seed,
    )
    print(
        "Created manifests at ",
        f"{[str(p.relative_to(MANIFESTS_DIR)) for p in out_paths.values()]}",
    )


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
