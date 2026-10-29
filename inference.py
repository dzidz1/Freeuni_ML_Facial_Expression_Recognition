"""Shared inference code for the FER2013 emotion classifier.

app.py and 05_optimization.ipynb both import from this file, so every prediction
goes through exactly the same preprocessing the model saw during evaluation in
04_transfer_learning.ipynb (eval_tf).
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FACE_SIZE = 48    # FER2013 images are 48x48 grayscale
INPUT_SIZE = 96   # the fine-tuned ResNet18 takes 96x96 inputs

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "ResNet18-pretrained.pth"

# Identical to eval_tf in 04_transfer_learning.ipynb
eval_tf = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model(num_classes=len(EMOTIONS)):
    """ResNet18 with a 7-class head: the same architecture as tl_model in notebook 04.

    weights=None because the trained weights come from our own file,
    so there is no need to download the ImageNet weights first.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(weights_path=DEFAULT_WEIGHTS, device="cpu"):
    """Build the model, load the trained weights and switch to eval mode."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No weights at {weights_path}. Download ResNet18-pretrained.pth "
            "from the GitHub Release into the weights/ folder (see README)."
        )
    model = build_model()
    # weights_only=True: load tensors only, never run pickled code from the file
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def preprocess(face):
    """Grayscale face crop (2D uint8 array, any size) -> model input tensor [1, 3, 96, 96].

    The crop is first shrunk to 48x48 so it has the same level of detail as the
    FER2013 training images. Then it goes through the same steps as in training:
    scale to [0, 1] (as in FERDataset), then eval_tf.
    """
    if face.ndim != 2 or face.dtype != np.uint8:
        raise ValueError(f"Expected a 2D uint8 grayscale image, got shape {face.shape}, dtype {face.dtype}")
    if face.shape != (FACE_SIZE, FACE_SIZE):
        face = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(np.ascontiguousarray(face)).float().unsqueeze(0) / 255.0  # [1, 48, 48]
    return eval_tf(x).unsqueeze(0)                                                   # [1, 3, 96, 96]


@torch.no_grad()
def predict(model, face):
    """Return {emotion: probability} for one grayscale face crop, sorted high to low."""
    device = next(model.parameters()).device
    logits = model(preprocess(face).to(device))
    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    return dict(sorted(zip(EMOTIONS, probs), key=lambda kv: kv[1], reverse=True))


if __name__ == "__main__":
    # Quick check: python inference.py path/to/face.jpg
    import sys

    if len(sys.argv) != 2:
        sys.exit("Usage: python inference.py path/to/face_image.jpg")
    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        sys.exit(f"Could not read image: {sys.argv[1]}")
    model = load_model()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    for emotion, p in predict(model, image).items():
        print(f"{emotion:<9}{p:6.1%}")
