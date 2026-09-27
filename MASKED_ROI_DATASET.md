# Masked-ROI dataset — where it lives now

The masked-ROI dataset (972 episodes, ~3.75M frames: 180x180 RGB frames plus
binary ROI masks) is split across two places, and no longer unpacked on this
machine.

| what | where | visibility |
|---|---|---|
| Frames (83.1 GiB) | `Stefanobraghetto/dreamer4-masked-roi-180` on HuggingFace | public |
| Masks (4.2 GiB) | `Stefanobraghetto/dreamer4-masked-roi-180-masks` on HuggingFace | **private** |
| Packed local archive, both | `/media/stefano/ExternalSSD/PhD/drqv2/masked-roi-180-tars/` | — |

## Why it was split

The combined dataset is 104.3 GB, which exceeds the 100 GB private-storage
allowance of a free Hugging Face account, so it could not be made private at
all. It was also measured to be 98.3% reproducible: the published 180x180
frames are a **bit-exact bilinear resize** of the released 224x224 Dreamer4
frames (mean absolute difference 0.00 against
`raw-dreamer4/expert/walker-stand-1.png`, frame 2004 — a resize, not a crop).

So the frames stay public and reproducible, and the only genuinely original
output — the binary ROI masks — sits in a private repo. The private dataset's
card documents the frame-derivation recipe in full.

## Why the local copy is packed

The external drive is exFAT with a **128 KiB cluster**. Unpacked, the dataset
is 7.5M small files, so every frame and every mask would occupy a full cluster:
84.6 GiB of data would have consumed **915 GiB** on that drive. Packed as one
tar per episode — 1,944 files, the same layout as the HuggingFace copy — it
takes 90 GiB.

## Format

Identical to the HuggingFace dataset, so the two are interchangeable:

```
<level>/<episode>.tar              frames+masks, uncompressed
<level>/<episode>.metadata.json    sidecar
```

Inside a tar, frame *N* is `NNNNNN.frame.png` and its mask is
`NNNNNN.mask.png`. Levels: `expert` (111), `mixed-small` (111),
`mixed-large` (750).

```python
from huggingface_hub import snapshot_download
snapshot_download("Stefanobraghetto/dreamer4-masked-roi-180",
                  repo_type="dataset", local_dir="dreamer4_masked_dataset_all37_180")
```

or, from the local archive:

```bash
tar -xf /media/stefano/ExternalSSD/PhD/drqv2/masked-roi-180-tars/expert/acrobot-swingup-0.tar -C <dir>
```

## What produced it, and what was dropped

`custom/mask_predictor/predict_dreamer4.py` writes the masks and, from
`frames/` + `masks/`, also the derived `masked_frames/`, `overlay_frames/` and
`*_strip.png` contact sheets. Those derived files (135.8 GiB) were deleted on
2026-09-27: the script recreates them in seconds, and
`tools/pack_and_upload_hf.py` in the dreamer4 repositories never uploads them.

Deletion manifests (every removed path, with sizes):

- `/media/stefano/ExternalSSD/PhD/manifests/masked-derived-delete-2026-09-27.tsv`
- `/media/stefano/ExternalSSD/PhD/manifests/checkpoint-prune-2026-09-27.tsv`
