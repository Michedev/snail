from path import Path

ROOT = Path(__file__).parent
DATAFOLDER = ROOT / 'data'
OMNIGLOTFOLDER = DATAFOLDER / 'omniglot-py'
MINIIMAGENETFOLDER = DATAFOLDER / 'miniimagenet'
WEIGHTSFOLDER = ROOT / 'model_weights'

if not WEIGHTSFOLDER.exists():
    WEIGHTSFOLDER.mkdir()