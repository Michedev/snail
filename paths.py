from path import Path

ROOT = Path(__file__).parent
DATAFOLDER = ROOT / 'data'
OMNIGLOTFOLDER = DATAFOLDER / 'omniglot-py'
MINIIMAGENETFOLDER = DATAFOLDER / 'miniimagenet'
WEIGHTSFOLDER = ROOT / 'model_weights'

if not WEIGHTSFOLDER.exists():
    WEIGHTSFOLDER.mkdir()

TRAIN_MINIIMAGENET = MINIIMAGENETFOLDER / 'train'
PRETRAIN: Path = WEIGHTSFOLDER / 'embedding-pretraining'
if not PRETRAIN.exists():
    PRETRAIN.mkdir()
PRETRAINED_EMBEDDING_PATH = PRETRAIN / 'embedding_miniimagenet.pth'
PRETRAINED_EMBEDDING_CLASSIFIER_PATH = PRETRAIN / 'embedding_classifier_miniimagenet.pth'
