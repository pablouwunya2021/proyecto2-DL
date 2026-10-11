"""
Extractor de caracteristicas para la prueba de vecinos mas cercanos.

Intenta usar una CNN preentrenada (SqueezeNet de torchvision). Si no hay acceso
a los pesos (sin internet), cae a un descriptor clasico reproducible basado en
histograma de color + bordes a baja resolucion, para que la prueba de vecinos
funcione siempre y de forma determinista.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.mode = "pixel"
        self.model = None
        try:
            import torchvision.models as M
            weights = M.SqueezeNet1_1_Weights.IMAGENET1K_V1
            net = M.squeezenet1_1(weights=weights)
            self.model = net.features.to(device).eval()
            self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            self.mode = "cnn-squeezenet"
        except Exception as e:  # sin internet / sin pesos
            print(f"[features] CNN preentrenada no disponible ({type(e).__name__}); "
                  f"uso descriptor de color+bordes.")

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) en [-1,1] -> (B, D) normalizado L2."""
        x = x.to(self.device)
        if self.mode.startswith("cnn"):
            x01 = (x.clamp(-1, 1) + 1) / 2
            x01 = F.interpolate(x01, size=224, mode="bilinear", align_corners=False)
            x01 = (x01 - self._mean) / self._std
            feats = self.model(x01)
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        else:
            feats = self._color_edge(x)
        return F.normalize(feats, dim=1)

    def _color_edge(self, x: torch.Tensor) -> torch.Tensor:
        x01 = (x.clamp(-1, 1) + 1) / 2
        # histograma de color grueso (8 niveles por canal) + mapa de bordes 8x8
        B = x01.size(0)
        bins = torch.clamp((x01 * 8).long(), 0, 7)
        hist = torch.zeros(B, 3 * 8, device=x01.device)
        for c in range(3):
            for b in range(8):
                hist[:, c * 8 + b] = (bins[:, c] == b).float().mean(dim=(1, 2))
        gray = x01.mean(1, keepdim=True)
        gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        edge = (gx.abs().mean(dim=(1, 2, 3), keepdim=True).squeeze(-1).squeeze(-1))
        edges_small = F.adaptive_avg_pool2d((gx.abs()[:, :, :-1, :] + gy.abs()[:, :, :, :-1]),
                                            8).flatten(1)
        return torch.cat([hist, edge, edges_small], dim=1)
