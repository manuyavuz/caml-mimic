PAD_CHAR = "**PAD**"

EMBEDDING_SIZE = 100
MAX_LENGTH = 2500

from pathlib import Path
path = Path(__file__).parent.absolute()

#where you want to save any models you may train
MODEL_DIR = path / 'models/'

# DATA_DIR = '/data/corpora/mimic/experiments/caml-mimic/mimicdata/'
DATA_DIR = path / 'mimicdata/'


MIMIC_3_DIR = DATA_DIR / 'mimic3'
MIMIC_2_DIR = DATA_DIR / 'mimic2'
