from path import Path

ROOT = Path(__file__).parent
DATAFOLDER = ROOT / 'data'
OMNIGLOTFOLDER = DATAFOLDER / 'omniglot-py'
WEIGHTSFOLDER = ROOT / 'model_weights'

if not WEIGHTSFOLDER.exists():
    WEIGHTSFOLDER.mkdir()