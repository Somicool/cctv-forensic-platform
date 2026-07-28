"""OSNet person re-identification embeddings.

Produces a 512-d appearance embedding per person crop so the SAME person can be
matched across different cameras (cross-camera tracking) even without a clear
face. Uses torchreid's FeatureExtractor (OSNet x1_0).

Note: currently uses ImageNet-pretrained OSNet weights (generic). Swapping in
Market-1501-trained weights (config.REID_MODEL + a model_path) would sharpen
re-ID accuracy; the pipeline is identical either way.

CLI self-test (needs the Task-4 person crops):
    python -m app.ingestion.reid_embedder
"""
from __future__ import annotations

import numpy as np

from .. import config

_extractor = None


def get_extractor():
    """Lazy-load the OSNet feature extractor once."""
    global _extractor
    if _extractor is None:
        try:
            from torchreid.reid.utils import FeatureExtractor
        except ImportError:                       # older torchreid layout
            from torchreid.utils import FeatureExtractor
        _extractor = FeatureExtractor(
            model_name=config.REID_MODEL, model_path="", device=config.DEVICE
        )
    return _extractor


def embed_persons(images, batch_size: int = 32) -> np.ndarray:
    """Encode person crops (paths or ndarrays) -> [N, 512] float32, normalised."""
    if not images:
        return np.zeros((0, config.REID_DIM), dtype="float32")
    ext = get_extractor()
    out = []
    for i in range(0, len(images), batch_size):
        feats = ext(images[i:i + batch_size])          # torch tensor [B, 512]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.detach().cpu().numpy().astype("float32"))
    return np.concatenate(out, axis=0)


def embed_person(image) -> np.ndarray:
    return embed_persons([image])[0]


if __name__ == "__main__":
    test_dir = config.CROP_DIR / "test"
    person_crops = sorted(str(p) for p in test_dir.glob("*_person.jpg")) if test_dir.exists() else []
    if len(person_crops) < 2:
        print(f"Need >=2 person crops in {test_dir}; run: python -m app.ingestion.detector")
    else:
        embs = embed_persons(person_crops)
        print("embeddings shape:", embs.shape)
        self_sim = float(embs[0] @ embs[0])
        cross_sim = float(embs[0] @ embs[1])
        print(f"self-sim(0,0)={self_sim:.3f}  cross-sim(0,1)={cross_sim:.3f}")
        print("pairwise similarity matrix:")
        print(np.round(embs @ embs.T, 2))
