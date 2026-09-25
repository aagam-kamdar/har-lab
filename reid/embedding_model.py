"""
reid/embedding_model.py
-----------------------
Modular person Re-ID embedding extractor.
Runs 100% locally on CPU/GPU without cloud APIs.

Uses a lightweight deep CNN (MobileNetV3 backbone) combined with
spatial color histograms to produce an appearance feature vector
sufficient to distinguish different students by body, face, and clothing.
"""

import os
from abc import ABC, abstractmethod
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
import cv2


class BaseReIDModel(ABC):
    """Abstract interface so Re-ID model can be replaced cleanly in the future."""

    @abstractmethod
    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        """
        Takes a BGR person crop (H x W x 3 uint8).
        Returns a 1D L2-normalized float32 embedding vector.
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass


class MobileNetV3ReIDExtractor(torch.nn.Module):
    """
    Torch module wrapping MobileNetV3-Small feature backbone
    with global pooling for Re-ID embedding extraction.
    """
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return torch.nn.functional.normalize(x, p=2, dim=1)


class PersonReIDExtractor(BaseReIDModel):
    """
    Local Person Re-ID Feature Extractor.
    Combines:
      - 576-dim Deep CNN features (MobileNetV3-Small)
      - 96-dim Upper/Lower body HSV spatial color histograms
    Result: 672-dimensional L2-normalized appearance vector.
    """

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)

        # Standard person Re-ID crop resolution (height: 224, width: 112 -> 2:1 aspect ratio)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load backbone
        base = models.mobilenet_v3_small(weights=None)
        if weights_path is None:
            # Default to project reid/weights/ path
            weights_path = os.path.join(os.path.dirname(__file__), "weights", "mobilenet_v3_small.pth")

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            base.load_state_dict(state_dict)
        else:
            # Fallback if local weights file not yet copied
            base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        self.model = MobileNetV3ReIDExtractor(base).to(self.device).eval()
        self._dim = 576 + 96  # 672 dimensions

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        """
        Extract normalized embedding from a person crop.
        """
        if person_crop is None or person_crop.size == 0 or person_crop.shape[0] < 12 or person_crop.shape[1] < 12:
            return np.zeros(self._dim, dtype=np.float32)

        # 1. Deep CNN feature
        tensor = self.transform(person_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cnn_feat = self.model(tensor).cpu().numpy()[0]

        # 2. Upper and lower body HSV spatial color histogram
        try:
            hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]
            mid = h // 2
            upper = hsv[:mid, :]
            lower = hsv[mid:, :]
            # 12 hue bins, 4 saturation bins = 48 bins per region
            hist_u = cv2.calcHist([upper], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
            hist_l = cv2.calcHist([lower], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
            hist = np.concatenate([hist_u, hist_l])
            norm = np.linalg.norm(hist)
            if norm > 1e-6:
                hist = hist / norm
            else:
                hist = np.zeros_like(hist)
        except Exception:
            hist = np.zeros(96, dtype=np.float32)

        # 3. Concatenate and L2-normalize
        combined = np.concatenate([cnn_feat * 0.82, hist * 0.18]).astype(np.float32)
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 1e-6:
            combined = combined / combined_norm
        return combined
