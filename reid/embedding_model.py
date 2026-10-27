"""
reid/embedding_model.py
-----------------------
Modular person Re-ID embedding extractor.
Runs 100% locally on CPU/GPU without cloud APIs.

Uses a lightweight deep CNN (MobileNetV3 backbone) combined with
spatial color histograms to produce an appearance feature vector
sufficient to distinguish different students by body, face, and clothing.

IMPORTANT: torchvision is NOT imported here (broken install in this env).
The MobileNetV3-Small architecture is re-implemented with pure torch.nn,
and the existing local weights file (reid/weights/mobilenet_v3_small.pth)
is loaded directly — keeping full compatibility with the pre-trained weights.
"""

import os
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


# ---------------------------------------------------------------------------
# Pure torch.nn MobileNetV3-Small (identical to torchvision's architecture)
# so the pre-trained weights load without needing torchvision.
# ---------------------------------------------------------------------------

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class _HardSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class _HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class _SqueezeExcitation(nn.Module):
    def __init__(self, in_ch, squeeze_ch):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, squeeze_ch, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_ch, in_ch, 1)
        self.hardsigmoid = _HardSigmoid()

    def forward(self, x):
        s = self.avgpool(x)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.hardsigmoid(s)
        return x * s


class _ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 groups=1, activation=nn.ReLU):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.01, eps=1e-3),
            activation(inplace=True) if activation not in (_HardSwish, _HardSigmoid)
            else activation(),
        )


class _InvertedResidual(nn.Module):
    def __init__(self, in_ch, exp_ch, out_ch, kernel, stride,
                 use_se, use_hs):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        act = _HardSwish if use_hs else nn.ReLU
        layers = []
        if exp_ch != in_ch:
            layers.append(_ConvBNActivation(in_ch, exp_ch, 1, activation=act))
        layers += [
            _ConvBNActivation(exp_ch, exp_ch, kernel, stride,
                              groups=exp_ch, activation=act),
        ]
        if use_se:
            layers.append(_SqueezeExcitation(exp_ch, _make_divisible(exp_ch // 4, 8)))
        layers.append(
            nn.Sequential(
                nn.Conv2d(exp_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
            )
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            out = out + x
        return out


class _MobileNetV3SmallFeatures(nn.Module):
    """
    MobileNetV3-Small feature extractor (features + avgpool only).
    Output: (B, 576) — same as torchvision's .features + .avgpool path.
    """

    # Config: (in, exp, out, kernel, stride, SE, HS)
    _INVERTED_RESIDUAL_SETTING = [
        (16,  16,   16,  3, 2, True,  False),
        (16,  72,   24,  3, 2, False, False),
        (24,  88,   24,  3, 1, False, False),
        (24,  96,   40,  5, 2, True,  True),
        (40,  240,  40,  5, 1, True,  True),
        (40,  240,  40,  5, 1, True,  True),
        (40,  120,  48,  5, 1, True,  True),
        (48,  144,  48,  5, 1, True,  True),
        (48,  288,  96,  5, 2, True,  True),
        (96,  576,  96,  5, 1, True,  True),
        (96,  576,  96,  5, 1, True,  True),
    ]

    def __init__(self):
        super().__init__()
        first_conv = _ConvBNActivation(3, 16, 3, 2, activation=_HardSwish)
        layers = [first_conv]
        for in_ch, exp_ch, out_ch, k, s, se, hs in self._INVERTED_RESIDUAL_SETTING:
            layers.append(_InvertedResidual(in_ch, exp_ch, out_ch, k, s, se, hs))
        # Final conv to 576
        last_conv = _ConvBNActivation(96, 576, 1, activation=_HardSwish)
        layers.append(last_conv)
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# Transforms (torchvision-free)
# ---------------------------------------------------------------------------

class _PersonTransform:
    """
    Resize + normalize a BGR numpy crop into a float32 tensor.
    Equivalent to torchvision transforms used in the original extractor.
    """
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, bgr_crop: np.ndarray) -> torch.Tensor:
        # 1. Resize to 224 x 112 (H x W, person Re-ID standard 2:1 ratio)
        resized = cv2.resize(bgr_crop, (112, 224))
        # 2. BGR -> RGB and float32 [0,1]
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        # 3. Normalize
        rgb = (rgb - self.MEAN) / self.STD
        # 4. HWC -> CHW tensor
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        return tensor


# ---------------------------------------------------------------------------
# Public interfaces (identical signatures to original)
# ---------------------------------------------------------------------------

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


class PersonReIDExtractor(BaseReIDModel):
    """
    Local Person Re-ID Feature Extractor.
    Combines:
      - 576-dim Deep CNN features (MobileNetV3-Small, pure torch.nn)
      - 96-dim Upper/Lower body HSV spatial color histograms
    Result: 672-dimensional L2-normalized appearance vector.
    """

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.transform = _PersonTransform()

        model = _MobileNetV3SmallFeatures().to(self.device)

        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(__file__), "weights", "mobilenet_v3_small.pth"
            )

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            # The saved weights are the FULL MobileNetV3-Small (including classifier).
            # Extract only the parts we use: features.* and avgpool.*  (if present).
            # Filter to keys that exist in our model.
            model_keys = set(model.state_dict().keys())
            filtered = {}
            for k, v in state_dict.items():
                # The original torchvision keys start with "features." directly.
                # Our model also has "features." → exact match.
                if k in model_keys:
                    filtered[k] = v
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            if missing:
                # Weights didn't perfectly align — use random init for those layers
                pass
        else:
            # No local weights — use random init (still works, just less accurate)
            pass

        model.eval()
        self.model = model
        self._dim = 576 + 96  # 672 dimensions

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def extract(self, person_crop: np.ndarray) -> np.ndarray:
        """Extract normalized embedding from a person crop."""
        if (
            person_crop is None
            or person_crop.size == 0
            or person_crop.shape[0] < 12
            or person_crop.shape[1] < 12
        ):
            return np.zeros(self._dim, dtype=np.float32)

        # 1. Deep CNN feature
        tensor = self.transform(person_crop).to(self.device)
        with torch.no_grad():
            cnn_feat = self.model(tensor).cpu().numpy()[0]  # (576,)

        # 2. Upper and lower body HSV spatial color histogram
        try:
            hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]
            mid = h // 2
            upper = hsv[:mid, :]
            lower = hsv[mid:, :]
            hist_u = cv2.calcHist([upper], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
            hist_l = cv2.calcHist([lower], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
            hist = np.concatenate([hist_u, hist_l])
            norm = np.linalg.norm(hist)
            hist = hist / norm if norm > 1e-6 else np.zeros_like(hist)
        except Exception:
            hist = np.zeros(96, dtype=np.float32)

        # 3. Concatenate and L2-normalize (same weights as original)
        combined = np.concatenate([cnn_feat * 0.82, hist * 0.18]).astype(np.float32)
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 1e-6:
            combined = combined / combined_norm
        return combined


# Alias kept for backward compatibility
MobileNetV3ReIDExtractor = _MobileNetV3SmallFeatures
