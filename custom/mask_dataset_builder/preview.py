import numpy as np


def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.35):
    rgb = np.asarray(rgb, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    color = np.asarray(color, dtype=np.float32)
    overlay = rgb.copy()
    overlay[mask] = ((1.0 - alpha) * overlay[mask]) + (alpha * color)
    return np.clip(overlay, 0, 255).astype(np.uint8)
