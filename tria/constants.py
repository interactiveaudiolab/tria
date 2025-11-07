from pathlib import Path

MANIFESTS_DIR = Path(__file__).parent.parent / "manifests"
DATA_DIR = Path(__file__).parent.parent / "data"
PRETRAINED_DIR = Path(__file__).parent.parent / "pretrained"
ASSETS_DIR = Path(__file__).parent.parent / "assets"


STEMS = ["drums", "bass", "vocals", "other", "mixture"]
SAMPLE_RATE = 44_100
DURATION = 6.0
