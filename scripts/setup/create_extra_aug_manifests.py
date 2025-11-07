import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

import argbind


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
    from tria.util import get_info
    from tria.constants import DATA_DIR, MANIFESTS_DIR

warnings.filterwarnings("ignore", category=UserWarning)

################################################################################
# Create CSV manifests for additional augmentation data
################################################################################


@argbind.bind(without_prefix=True)
def main(seed: int = 0):
    # Fetch audio sample rate in Hz and duration in seconds
    metadata_fns = {
        "sample_rate": lambda p: get_info(str(p))[1],
        "duration": lambda p: get_info(str(p))[0],
    }

    aug_attributes = {**metadata_fns, "path": lambda p: str(p)}

    ########################################
    # RIR/Noise Database
    ########################################

    print("Creating manifests for RIR Database (real)")
    out_paths = create_manifests(
        DATA_DIR / "rir-database" / "real",
        ext="wav",
        output_dir=MANIFESTS_DIR / "rir_real",
        split=(0.8, 0.1, 0.1),
        attributes=aug_attributes,
        seed=seed,
    )
    print(
        "Created manifests at ",
        f"{[str(p.relative_to(MANIFESTS_DIR)) for p in out_paths.values()]}",
    )

    print("Creating manifests for RIR Database (synthetic)")
    out_paths = create_manifests(
        DATA_DIR / "rir-database" / "synthetic",
        ext="wav",
        output_dir=MANIFESTS_DIR / "rir_synthetic",
        split=(0.8, 0.1, 0.1),
        attributes=aug_attributes,
        seed=seed,
    )
    print(
        "Created manifests at ",
        f"{[str(p.relative_to(MANIFESTS_DIR)) for p in out_paths.values()]}",
    )

    print("Creating manifests for Noise Database (pointsource)")
    out_paths = create_manifests(
        DATA_DIR / "noise-database" / "pointsource",
        ext="wav",
        output_dir=MANIFESTS_DIR / "noise_pointsource",
        split=(0.8, 0.1, 0.1),
        attributes=aug_attributes,
        seed=seed,
    )
    print(
        "Created manifests at ",
        f"{[str(p.relative_to(MANIFESTS_DIR)) for p in out_paths.values()]}",
    )

    print("Creating manifests for Noise Database (room)")
    out_paths = create_manifests(
        DATA_DIR / "noise-database" / "room",
        ext="wav",
        output_dir=MANIFESTS_DIR / "noise_room",
        split=(0.8, 0.1, 0.1),
        attributes=aug_attributes,
        seed=seed,
    )
    print(
        "Created manifests at ",
        f"{[str(p.relative_to(MANIFESTS_DIR)) for p in out_paths.values()]}",
    )

    ########################################
    # MUSAN, MIT-IR, WHAM
    ########################################

    for dataset in ["musan", "MIT-IR", "high-res-wham"]:
        out_name = dataset.lower().replace("-", "_")

        print(f"Creating manifests for {dataset}")
        out_paths = create_manifests(
            DATA_DIR / dataset,
            ext="wav",
            output_dir=MANIFESTS_DIR / out_name,
            split=(0.8, 0.1, 0.1),
            attributes=aug_attributes,
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
