"""OpenCLIP embeddings.

Turns images AND text into the same 512-d normalised vector space, so a text
query and an image crop can be compared directly (cosine similarity). This is
the heart of the descriptive search: encode the query text, find the closest
image vectors.

Also used for whole-frame embeddings (scene-level search) and as a fallback
visual embedding for vehicle image-search.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import config

_model = None
_preprocess = None
_tokenizer = None
_clip_mean = None            # normalization pulled from the preprocess transform
_clip_std = None
_norm_dev = None             # cached (mean, std) tensors on the device


def get_clip():
    """Lazy-load the CLIP model, preprocess transform and tokenizer once."""
    global _model, _preprocess, _tokenizer, _clip_mean, _clip_std
    if _model is None:
        import open_clip
        import torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED, device=config.DEVICE
        )
        model.eval()
        _model = model
        _preprocess = preprocess
        _tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
        # Pull the EXACT normalization used by the transform so the fast GPU
        # preprocess path stays numerically faithful to embed_images().
        for tr in getattr(preprocess, "transforms", []):
            if hasattr(tr, "mean") and hasattr(tr, "std"):
                _clip_mean = tuple(float(x) for x in tr.mean)
                _clip_std = tuple(float(x) for x in tr.std)
        if _clip_mean is None:                       # OpenAI CLIP defaults
            _clip_mean = (0.48145466, 0.4578275, 0.40821073)
            _clip_std = (0.26862954, 0.26130258, 0.27577711)
    return _model, _preprocess, _tokenizer


_CLIP_SIZE = 224


def _letterbox_224(bgr):
    """Pad a BGR crop to square (neutral grey) then resize to 224 - equivalent
    to _pad_to_square + CLIP Resize/CenterCrop, but with fast cv2 ops."""
    h, w = bgr.shape[:2]
    if h != w:
        s = max(h, w)
        canvas = np.full((s, s, 3), 114, np.uint8)
        y0, x0 = (s - h) // 2, (s - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = bgr
        bgr = canvas
        src = s
    else:
        src = h
    interp = cv2.INTER_AREA if src > _CLIP_SIZE else cv2.INTER_CUBIC   # AREA ~ antialias on shrink
    return cv2.resize(bgr, (_CLIP_SIZE, _CLIP_SIZE), interpolation=interp)


def embed_crops(crops, batch_size: int = 64) -> np.ndarray:
    """Fast crop embedding: crops are in-memory BGR ndarrays (NO disk decode).
    Letterbox+resize on CPU with cv2 (C-fast), then do BGR->RGB, /255 and CLIP
    normalization on the GPU in batches - minimises CPU preprocessing and keeps
    the GPU busy. Output matches embed_images() closely (validated by parity)."""
    import torch
    global _norm_dev
    if not crops:
        return np.zeros((0, config.CLIP_DIM), dtype="float32")
    model, _, _ = get_clip()
    if _norm_dev is None:
        mean = torch.tensor(_clip_mean, device=config.DEVICE).view(1, 3, 1, 1)
        std = torch.tensor(_clip_std, device=config.DEVICE).view(1, 3, 1, 1)
        _norm_dev = (mean, std)
    mean, std = _norm_dev
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i + batch_size]
            arr = np.stack([_letterbox_224(c) for c in batch])       # [B,224,224,3] BGR uint8
            t = torch.from_numpy(arr).to(config.DEVICE).permute(0, 3, 1, 2).float().div_(255.0)
            t = t[:, [2, 1, 0], :, :]                                # BGR -> RGB
            t = (t - mean) / std
            feats = model.encode_image(t)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy().astype("float32"))
            if config.LOW_VRAM and config.DEVICE == "cuda":
                torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def _to_pil(img) -> Image.Image:
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):                # OpenCV BGR ndarray
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img.convert("RGB") if isinstance(img, Image.Image) else img


def _pad_to_square(pil: Image.Image) -> Image.Image:
    """Letterbox a PIL image to a square canvas (neutral grey) so CLIP's
    Resize+CenterCrop keeps the WHOLE object instead of clipping a tall person's
    head/feet or a wide vehicle's ends."""
    w, h = pil.size
    if w == h:
        return pil
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (114, 114, 114))
    canvas.paste(pil, ((s - w) // 2, (s - h) // 2))
    return canvas


def _prep(img) -> Image.Image:
    pil = _to_pil(img)
    return _pad_to_square(pil) if config.CLIP_PAD_SQUARE else pil


def embed_images(images, batch_size: int = 32) -> np.ndarray:
    """Encode a list of images (paths / ndarrays / PIL) -> [N, 512] float32,
    L2-normalised. Processed in batches to respect the 4GB GPU."""
    import torch
    model, preprocess, _ = get_clip()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([preprocess(_prep(im)) for im in batch]).to(config.DEVICE)
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy().astype("float32"))
            if config.LOW_VRAM and config.DEVICE == "cuda":
                torch.cuda.empty_cache()
    if not out:
        return np.zeros((0, config.CLIP_DIM), dtype="float32")
    return np.concatenate(out, axis=0)


def embed_texts(texts) -> np.ndarray:
    """Encode a list of text strings -> [N, 512] float32, L2-normalised."""
    import torch
    model, _, tokenizer = get_clip()
    with torch.no_grad():
        tokens = tokenizer(list(texts)).to(config.DEVICE)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


def embed_image(img) -> np.ndarray:
    """Single image -> [512] float32."""
    return embed_images([img])[0]


def embed_text(text: str) -> np.ndarray:
    """Single text -> [512] float32."""
    return embed_texts([text])[0]


if __name__ == "__main__":
    # Sanity check: a bus crop should score higher on "a photo of a bus"
    # than on "a photo of a cat".
    from ultralytics.utils import ASSETS
    bus_crop = config.CROP_DIR / "test" / "bus_det000_bus.jpg"
    img = str(bus_crop if bus_crop.exists() else ASSETS / "bus.jpg")
    ie = embed_image(img)
    te = embed_texts(["a photo of a bus", "a photo of a cat", "a photo of a person"])
    sims = te @ ie
    for label, s in zip(["bus", "cat", "person"], sims):
        print(f"  sim(image, '{label}') = {s:.3f}")
