from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "assets"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGENET_LABELS_PATH = DATA_DIR / "imagenet_class_index.json"
SAMPLE_IMAGE_PATH = DATA_DIR / "..."
SAMPLE_CLRP_IMAGE_PATH = DATA_DIR / "..."

X_BATCH_PATH = DATA_DIR / "x_batch.pt"
Y_BATCH_PATH = DATA_DIR / "y_batch.pt"
S_BATCH_PATH = DATA_DIR / "s_batch.pt"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
